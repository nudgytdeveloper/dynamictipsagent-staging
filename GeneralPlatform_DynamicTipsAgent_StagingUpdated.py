import base64
import difflib
import pymongo
from google import genai
from google.genai import types as genai_types
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import io
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse, unquote, quote, urlsplit, urlunsplit
from bson import ObjectId

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pypdf import PdfReader

# Render captures stdout through a pipe, where Python block-buffers it: every
# print of a request used to surface in one clump at the end, so the per-stage
# timing lines below would say nothing about *when* each stage ran.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

MONGO_URI              = os.environ.get("MONGO_URI")
DB_NAME                = os.environ.get("DB_NAME", "your_db_name")
TRANSCRIPTS_COLLECTION = os.environ.get("TRANSCRIPTS_COLLECTION", "transcripts")
TIPS_COLLECTION        = os.environ.get("TIPS_COLLECTION", "tips")
# Needed to resolve the service's uploaded SOP: simulation -> service_level -> file URL.
SIMULATIONS_COLLECTION    = os.environ.get("SIMULATIONS_COLLECTION", "simulations")
SERVICE_LEVELS_COLLECTION = os.environ.get("SERVICE_LEVELS_COLLECTION", "service_levels")
GEMINI_API_KEY         = os.environ.get("GEMINI_API_KEY")

# ============================================================================
# Response time ("Optimize speed of AI Tips Agent").
#
# Staging measured a 9.4 s median / 12.4 s p90 per /tips call (2026-08-25 →
# 08-31), the same with or without an SOP file — so the cost was not the
# download but the model: gemini-2.5-flash reasons before it answers unless
# told not to, and for a single ≤500-character tip that reasoning was most of
# the wait. The budget below turns it off. It is env-tunable so it can be
# raised from the Render dashboard, without a deploy, if tip quality ever
# needs it back. The legacy google-generativeai SDK could not set this at
# all, which is why the agent now uses google-genai.
#
# 2026-10-07: the replacement API key's project cannot call gemini-2.5-flash
# ("no longer available to new users"), so the default moved to
# gemini-3.8-flash. A negative budget leaves thinking_config off the request
# entirely, for a model that refuses a budget of 0.
# ============================================================================
GEMINI_MODEL             = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_THINKING_BUDGET   = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))
# Bounds a hung generation; the old client would wait on it indefinitely.
# The learner never waits this long: see TIPS_HEDGE_SECONDS below.
GEMINI_TIMEOUT_SECONDS   = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "20"))
GEMINI_TEMPERATURE       = 0.5
# Thinking tokens (when a budget is set) are charged against this cap too.
GEMINI_MAX_OUTPUT_TOKENS = 3000

# ============================================================================
# OpenAI fallback (2026-10-07). Gemini failures left the learner with "Failed
# to generate guidance" — first a free-tier quota (20 requests a day), then a
# model the replacement key could not call. When Gemini errors or returns no
# text, the same prompt (and SOP PDF) goes to OpenAI instead. Unset
# OPENAI_API_KEY and the agent behaves exactly as before.
# ============================================================================
OPENAI_API_KEY           = os.environ.get("OPENAI_API_KEY")
OPENAI_FALLBACK_MODEL    = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS   = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "20"))

# ============================================================================
# Hedged request (2026-10-28, Changi trial: "loading time should not be more
# than 10 seconds"). From 2026-10-26 gemini-3.8-flash mostly answered 503 or
# hung until the 60 s timeout, and only then did the fallback start, so a tip
# took 45-83 s. Gemini normally answers in 3-4 s, so if it has not answered
# within TIPS_HEDGE_SECONDS (or fails sooner) OpenAI is asked as well and the
# first tip back wins. The slower call is abandoned, not awaited.
# ============================================================================
TIPS_HEDGE_SECONDS     = float(os.environ.get("TIPS_HEDGE_SECONDS", "5"))
TIPS_MODEL_POOL_SIZE   = int(os.environ.get("TIPS_MODEL_POOL_SIZE", "16"))
_model_pool            = ThreadPoolExecutor(max_workers=TIPS_MODEL_POOL_SIZE)

client                 = pymongo.MongoClient(MONGO_URI)
db                     = client[DB_NAME]
transcripts_collection = db[TRANSCRIPTS_COLLECTION]
tips_collection        = db[TIPS_COLLECTION]
simulations_collection    = db[SIMULATIONS_COLLECTION]
service_levels_collection = db[SERVICE_LEVELS_COLLECTION]

_gemini_client: Optional[genai.Client] = None
_gemini_client_lock = threading.Lock()


def gemini_client() -> genai.Client:
    """One shared client, built on first use so a missing key fails the request, not the boot."""
    global _gemini_client
    with _gemini_client_lock:
        if _gemini_client is None:
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY is not set")
            _gemini_client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=genai_types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000),
            )
        return _gemini_client


_openai_client: Optional[OpenAI] = None
_openai_client_lock = threading.Lock()


def openai_client() -> OpenAI:
    """One shared OpenAI client for the fallback, built on first use."""
    global _openai_client
    with _openai_client_lock:
        if _openai_client is None:
            _openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS, max_retries=1)
        return _openai_client


def generate_with_openai(prompt: str, sop_document: Optional[Dict[str, Any]] = None) -> str:
    """The Gemini prompt, answered by OpenAI. A PDF SOP rides along as a file part."""
    content: List[Dict[str, Any]] = []
    if sop_document and sop_document.get("data"):
        encoded = base64.b64encode(sop_document["data"]).decode("ascii")
        content.append({"type": "file", "file": {
            "filename": "sop.pdf",
            "file_data": f"data:{sop_document['mime_type']};base64,{encoded}",
        }})
    content.append({"type": "text", "text": prompt})
    response = openai_client().chat.completions.create(
        model=OPENAI_FALLBACK_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=GEMINI_TEMPERATURE,
        max_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


def response_text(response: Any) -> str:
    """Text of a Gemini response, or "" — never an exception.

    `response.text` is None when the candidate carries no text part (a safety
    block, or a MAX_TOKENS stop where the budget went to thinking); walk the
    parts before giving up so a partial answer is not thrown away.
    """
    try:
        text = response.text
        if text:
            return text.strip()
    except Exception as e:
        print(f"[DEBUG] Gemini response exposed no .text ({e}); reading parts")

    chunks = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


def finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return str(getattr(reason, "name", reason) or "unknown")


def ask_gemini(contents: List[Any]) -> str:
    """Gemini's answer; raises on an error or an empty answer."""
    response = gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            thinking_config=(genai_types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
                             if GEMINI_THINKING_BUDGET >= 0 else None),
        ),
    )
    text = response_text(response)
    if not text:
        raise RuntimeError(f"Gemini returned no text (finish_reason={finish_reason(response)})")
    return text


def ask_openai(prompt: str, sop_document: Optional[Dict[str, Any]]) -> str:
    """OpenAI's answer to the same prompt; raises on an error or an empty answer."""
    text = generate_with_openai(prompt, sop_document)
    if not text:
        raise RuntimeError("OpenAI returned no text")
    return text


def first_answer(prompt: str, contents: List[Any],
                 sop_document: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str], str]:
    """(text, error, model) from the first model to answer; see TIPS_HEDGE_SECONDS.

    Gemini goes first. OpenAI joins when Gemini fails, or has not answered
    within the hedge window. Without OPENAI_API_KEY this is a plain Gemini call.
    """
    gemini = _model_pool.submit(ask_gemini, contents)
    wait([gemini], timeout=TIPS_HEDGE_SECONDS)
    if gemini.done() and gemini.exception() is None:
        return gemini.result(), None, GEMINI_MODEL

    pending: Dict[Future, str] = {gemini: GEMINI_MODEL}
    if OPENAI_API_KEY:
        reason = f"Gemini: {gemini.exception()}"[:300] if gemini.done() else \
                 f"Gemini has not answered in {TIPS_HEDGE_SECONDS:g}s"
        print(f"{reason} — asking {OPENAI_FALLBACK_MODEL} too")
        pending[_model_pool.submit(ask_openai, prompt, sop_document)] = OPENAI_FALLBACK_MODEL

    errors: List[str] = []
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            model = pending.pop(future)
            if future.exception() is None:
                return future.result(), None, model
            errors.append(f"{model}: {future.exception()}")
    return "", "; ".join(errors), OPENAI_FALLBACK_MODEL if OPENAI_API_KEY else GEMINI_MODEL


# ============================================================================
# SOP / reference documents (task 307, "Tips based on SOP or documents").
#
# The platform lets an admin upload a PDF during service / template creation.
# The file's URL lands on the service's `service_levels` document:
#   sop_tips_generation_file     -> read HERE, as the basis for tip generation
#   sop_transcript_summary_file  -> read by the summary agent (challenger-services)
#
# The field is optional: with no SOP configured (or an unusable one) this agent
# generates exactly the same generic communication-coaching tips it always has.
# ============================================================================
SNAPSHOT_KEY          = "dependency_snapshot"
SIM_SERVICE_LEVEL_KEY = "service_level"
SOP_TIPS_FILE_KEY     = "sop_tips_generation_file"

# The field may hold a bare URL string, an object, or a list of either, so probe
# the usual URL-bearing keys rather than assuming one shape.
SOP_FILE_URL_KEYS = ("url", "file_url", "fileUrl", "src", "location", "path")

SOP_DOWNLOAD_TIMEOUT_SECONDS = 30
SOP_MAX_BYTES     = 15 * 1024 * 1024   # keep the request under Gemini's inline-data cap
SOP_PDF_MIME      = "application/pdf"
SOP_PDF_EXTENSIONS  = (".pdf",)
SOP_TEXT_EXTENSIONS = (".txt", ".md")
# A PDF SOP is sent as its text when it has at least this much (2026-10-28).
# Both models read an attached PDF page by page as images on every request:
# Changi's 89-page, 4 MB RTS deck holds only ~15k characters of text, yet with
# it attached Gemini took 11-14 s and gpt-4o-mini ~23 s. Below the threshold
# (a scanned PDF) the file itself is still attached.
SOP_PDF_MIN_TEXT_CHARS = int(os.environ.get("SOP_PDF_MIN_TEXT_CHARS", "500"))


def _first_file_url(value: Any) -> Optional[str]:
    """Pull a usable URL out of a `sop_*_file` field of unknown shape."""
    if not value:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in SOP_FILE_URL_KEYS:
            url = _first_file_url(value.get(key))
            if url:
                return url
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            url = _first_file_url(item)
            if url:
                return url
    return None


def get_simulation(simulation_id: str) -> Optional[Dict[str, Any]]:
    """Look the simulation up by ObjectId, falling back to a string _id."""
    try:
        sim = simulations_collection.find_one({"_id": ObjectId(simulation_id)})
    except Exception:
        sim = None
    if sim is None:
        sim = simulations_collection.find_one({"_id": simulation_id})
    return sim


def get_sop_tips_file_url(simulation: Dict[str, Any]) -> Optional[str]:
    """Resolve the SOP file the tips should be based on.

    Prefers the simulation's `dependency_snapshot` (what the service looked like
    when the simulation ran) and falls back to the live `service_levels` doc.
    """
    snapshot = simulation.get(SNAPSHOT_KEY)
    if isinstance(snapshot, dict):
        url = _first_file_url(snapshot.get(SOP_TIPS_FILE_KEY))
        if url:
            return url
        snapshot_service_level = snapshot.get(SIM_SERVICE_LEVEL_KEY)
        if isinstance(snapshot_service_level, dict):
            url = _first_file_url(snapshot_service_level.get(SOP_TIPS_FILE_KEY))
            if url:
                return url

    service_level = simulation.get(SIM_SERVICE_LEVEL_KEY)
    if not service_level:
        return None

    try:
        query = {"_id": ObjectId(str(service_level))}
    except Exception:
        query = {"_id": service_level}

    service_level_doc = service_levels_collection.find_one(query)
    if not service_level_doc:
        print(f"No {SERVICE_LEVELS_COLLECTION} doc for service_level {service_level}")
        return None

    return _first_file_url(service_level_doc.get(SOP_TIPS_FILE_KEY))


# --- Continuous Learning (Block D) consumer helper — pasted verbatim from continous-learning-agent/README.md ---
CL_INSTRUCTIONS_COLLECTION = os.environ.get("CL_INSTRUCTIONS_COLLECTION", "agent_learner_instructions")
CL_BEHAVIOR_CHANGES_COLLECTION = "agent_behavior_changes"
CL_SIM_LEARNER_KEYS = ("learner", "user", "learner_id", "user_id")
CL_ACTIVE_STATUSES = ("approved", "applied")
CL_STATUS_APPLIED = "applied"
CL_AGENT_NAME = "tips"
FIELD_CL_INSTRUCTION_ID = "cl_instruction_id"
FIELD_CL_INSTRUCTION_VERSION = "cl_instruction_version"
FIELD_CL_STAGE = "cl_stage"
CL_ABSENT_BLOCK = """
            No prior-session background is available for this learner; treat this as a standalone session and do not refer to previous sessions.
            """


def get_learner_instruction(simulation):
    """Active Continuous-Learning instruction for this simulation's learner, or None. Never raises."""
    try:
        learner_id = next((simulation.get(k) for k in CL_SIM_LEARNER_KEYS if simulation and simulation.get(k) is not None), None)
        if learner_id is None:
            return None
        ids = [learner_id, str(learner_id)]
        try:
            ids.append(ObjectId(str(learner_id)))
        except Exception:
            pass
        doc = db[CL_INSTRUCTIONS_COLLECTION].find_one(
            {"learner_id": {"$in": ids}, "status": {"$in": list(CL_ACTIVE_STATUSES)}}, sort=[("version", -1)])
        if not doc or not doc.get("instruction"):
            return None
        target = doc.get("for_service_level")
        if target and simulation.get("service_level") is not None and str(target) != str(simulation.get("service_level")):
            return None  # written for another scenario; the worker refreshes it when this one starts
        return doc
    except Exception as e:
        print(f"[DEBUG] CL instruction lookup failed: {e}")
        return None


def learner_instruction_block(doc):
    """Prompt block with an explicit empty case (same discipline as tips_instruction)."""
    if doc and doc.get("instruction"):
        hint = (doc.get("per_agent") or {}).get(CL_AGENT_NAME) or ""
        return f"""
            Background on this learner from their previous sessions (private; follow it, never quote or mention it):
            ---
            {doc["instruction"]}
            {hint}
            ---
            """
    return CL_ABSENT_BLOCK


def cl_provenance(doc):
    return {FIELD_CL_INSTRUCTION_ID: doc.get("_id") if doc else None,
            FIELD_CL_INSTRUCTION_VERSION: doc.get("version") if doc else None,
            FIELD_CL_STAGE: doc.get("stage") if doc else None}


def cl_mark_applied(doc, simulation_id):
    """Audit trail: this agent used the instruction for this simulation. Never raises."""
    if not doc:
        return
    try:
        now = datetime.now(timezone.utc)
        db[CL_INSTRUCTIONS_COLLECTION].update_one({"_id": doc["_id"], "applied_at": None}, {"$set": {"applied_at": now}})
        db[CL_INSTRUCTIONS_COLLECTION].update_one(
            {"_id": doc["_id"]},
            {"$set": {"status": CL_STATUS_APPLIED},
             "$push": {"applied": {"agent": CL_AGENT_NAME, "simulation_id": str(simulation_id), "at": now}}})
        db[CL_BEHAVIOR_CHANGES_COLLECTION].update_one(
            {"instruction_id": doc["_id"], "agent": CL_AGENT_NAME, "applied_at": None},
            {"$set": {"applied_at": now, "applied_simulation_id": str(simulation_id)}})
    except Exception as e:
        print(f"[DEBUG] CL mark-applied failed: {e}")
# --- end Continuous Learning consumer helper ---


def _encode_url(url: str) -> str:
    """Percent-encode the path/query of a URL.

    SOP filenames routinely contain spaces (e.g. "Tips SOP Sample .pdf").
    urllib rejects those outright with http.client.InvalidURL, so encode before
    requesting. `safe="%"` keeps any already-encoded sequence from being
    double-encoded.
    """
    parts = urlsplit(url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        quote(parts.path, safe="/%"),
        quote(parts.query, safe="=&%"),
        parts.fragment,
    ))


def pdf_text(payload: bytes) -> str:
    """The PDF's extractable text, or "" for a scanned or unreadable file. Never raises."""
    try:
        reader = PdfReader(io.BytesIO(payload))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as e:
        print(f"Could not read text from SOP PDF: {e}")
        return ""


def fetch_sop_document(url: str) -> Optional[Dict[str, Any]]:
    """Download the SOP file and return a part Gemini can consume.

    Returns {"url", "text"} for plain text or a PDF with enough text, {"url",
    "mime_type", "data"} for a PDF without, or None if the file is unreachable,
    oversized, or an unsupported type.
    Never raises: a bad SOP file must not cost the simulation its tips.
    """
    try:
        path = unquote(urlparse(url).path).lower()
    except Exception as e:
        print(f"Unparseable SOP URL {url!r}: {e}")
        return None

    is_pdf  = path.endswith(SOP_PDF_EXTENSIONS)
    is_text = path.endswith(SOP_TEXT_EXTENSIONS)
    if not is_pdf and not is_text:
        print(f"Unsupported SOP file type for {url} — skipping SOP grounding")
        return None

    try:
        request_url = _encode_url(url)
    except Exception:
        request_url = url

    try:
        with urllib.request.urlopen(request_url, timeout=SOP_DOWNLOAD_TIMEOUT_SECONDS) as response:
            # Read one byte past the cap so an oversized file is detected rather
            # than silently truncated into a malformed PDF.
            payload = response.read(SOP_MAX_BYTES + 1)
    except Exception as e:
        print(f"Could not download SOP file {url}: {e}")
        return None

    if not payload:
        print(f"SOP file {url} was empty")
        return None
    if len(payload) > SOP_MAX_BYTES:
        print(f"SOP file {url} exceeds {SOP_MAX_BYTES} bytes — skipping SOP grounding")
        return None

    if is_pdf:
        text = pdf_text(payload)
        if len(text) >= SOP_PDF_MIN_TEXT_CHARS:
            return {"url": url, "text": text}
        print(f"SOP file {url} has {len(text)} characters of text — attaching the PDF itself")
        return {"url": url, "mime_type": SOP_PDF_MIME, "data": payload}

    try:
        text = payload.decode("utf-8").strip()
    except UnicodeDecodeError as e:
        print(f"SOP file {url} is not valid UTF-8 text: {e}")
        return None
    if not text:
        print(f"SOP file {url} contained no text")
        return None
    return {"url": url, "text": text}


# ── SOP download cache ────────────────────────────────────────────────────
# Every /tips call for a service used to download its SOP again — the same
# PDF, up to 15 MB, on every request. The file is cached per URL for a
# bounded time so a session's second and later tips skip the download. An
# unusable result (unreachable, oversized, wrong type) is cached too, but
# only briefly, so a transient outage does not cost ten minutes of SOP-less
# tips. Keyed on the URL alone: an admin re-uploading under the *same*
# filename is served the old file until the entry expires.
SOP_CACHE_TTL_SECONDS         = int(os.environ.get("SOP_CACHE_TTL_SECONDS", "600"))
SOP_CACHE_FAILURE_TTL_SECONDS = int(os.environ.get("SOP_CACHE_FAILURE_TTL_SECONDS", "60"))
SOP_CACHE_MAX_ENTRIES         = 32

_sop_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_sop_cache_lock = threading.Lock()


def get_sop_document(url: str) -> Optional[Dict[str, Any]]:
    """`fetch_sop_document`, memoised per URL. Same contract: never raises."""
    now = time.monotonic()
    with _sop_cache_lock:
        entry = _sop_cache.get(url)
        if entry and now < entry[0]:
            return entry[1]

    document = fetch_sop_document(url)
    ttl = SOP_CACHE_TTL_SECONDS if document else SOP_CACHE_FAILURE_TTL_SECONDS

    with _sop_cache_lock:
        if len(_sop_cache) >= SOP_CACHE_MAX_ENTRIES:
            for key in [k for k, (expires_at, _) in _sop_cache.items() if expires_at <= now]:
                del _sop_cache[key]
        if len(_sop_cache) >= SOP_CACHE_MAX_ENTRIES:
            del _sop_cache[min(_sop_cache, key=lambda k: _sop_cache[k][0])]
        _sop_cache[url] = (now + ttl, document)
    return document


def resolve_sop_document(simulation_id: str,
                         simulation: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """(file URL, Gemini-ready document) for the simulation's service, either half None if absent."""
    sop_file_url = get_sop_tips_file_url(simulation)
    if not sop_file_url:
        print(f"No SOP file configured for simulation {simulation_id}")
        return None, None
    print(f"SOP file for simulation {simulation_id}: {sop_file_url}")
    sop_document = get_sop_document(sop_file_url)
    if not sop_document:
        print(f"SOP file unusable for simulation {simulation_id}; falling back to a generic tip")
    return sop_file_url, sop_document


# ============================================================================
# Transcript anchors (task 295, "Tips reflected against Transcript").
#
# The platform is to show each tip beside the transcript line it is about, and
# let trainers edit, delete and comment on a tip. Neither was possible: a stored
# tip carried no id and no reference to the transcript. Placing tips by time
# does not work either — every tip reads the whole conversation so far, so a
# tip issued at minute five can be about the learner's first message.
#
# So the learner's messages are labelled in the prompt ([U1], [U2], …) and the
# model names the one its tip is about on a trailing ANCHOR line, which is
# stripped before the tip is stored. The model's pick is checked against the
# real messages; failing that, a phrase the tip quotes is matched against
# them; failing that, the tip is anchored to the learner's latest message.
# ============================================================================
TRANSCRIPT_FROM_TYPE_KEY  = "from_type"
TRANSCRIPT_TEXT_KEY       = "dialogue_value"
FROM_TYPE_USER            = "User"
FROM_TYPE_CHARACTER       = "Character"
SPEAKER_USER              = "HUMAN USER"
SPEAKER_CHARACTER         = "AI CHARACTER"
LEARNER_LABEL_PREFIX      = "U"

FIELD_TIP_ID               = "_id"
FIELD_ANCHOR_TRANSCRIPT_ID = "anchor_transcript_id"
FIELD_ANCHOR_QUOTE         = "anchor_quote"
FIELD_ANCHOR_METHOD        = "anchor_method"
FIELD_LAST_TRANSCRIPT_ID   = "last_transcript_id"

# The platform finds a simulation's tips by an ObjectId `simulation_id`, and
# shows the notification bell for a tip whose `is_requested` is false (the
# learner did not ask for it). The agent used to store the id as a string, so
# older documents are keyed by a string and new tips are appended to whichever
# document already exists.
FIELD_SIMULATION_ID        = "simulation_id"
FIELD_IS_REQUESTED         = "is_requested"

ANCHOR_METHOD_MODEL       = "model"        # the model's ANCHOR line named a real learner message
ANCHOR_METHOD_QUOTE_MATCH = "quote_match"  # a phrase quoted in the tip was found in a learner message
ANCHOR_METHOD_LATEST      = "latest"       # neither: the learner's latest message

# `ANCHOR: U3 | "exact excerpt"` — the excerpt is optional.
ANCHOR_LINE_PATTERN  = re.compile(r'^\s*\**\s*ANCHOR\s*\**\s*:\s*\**\s*\[?U(\d+)\]?\s*(?:\|\s*(.*?))?\s*$',
                                  re.IGNORECASE | re.MULTILINE)
# Any [U3] label the model leaks into the tip itself.
LEARNER_LABEL_PATTERN = re.compile(r'\s*\[U\d+\]')
# Phrases a tip quotes: "…" or “…”.
TIP_QUOTE_PATTERN     = re.compile(r'"([^"]+)"|“([^”]+)”')
# Quoted fragments shorter than this match too much to place a tip by when
# searching every learner message. Inside the message the model named, any
# length will do: "hello" is the whole of many opening messages.
MIN_QUOTE_MATCH_CHARS = 8
MIN_NAMED_QUOTE_CHARS = 1
QUOTE_EDGE_CHARS      = " \t\n\"'“”‘’.…,;:!?"
# One character for one character, so a match position in the folded text is
# the same position in the original.
QUOTE_FOLD_TABLE      = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
# The model sometimes fixes the learner's typos while quoting them ("will that
# work" for "will taht work"). Inside the named message, a run of the learner's
# words this similar to the excerpt is taken as the quote, in their spelling.
FUZZY_QUOTE_MIN_RATIO = 0.85


def learner_label(index: int) -> str:
    return f"{LEARNER_LABEL_PREFIX}{index}"


def excerpt_words(excerpt: Optional[str]) -> List[str]:
    return (excerpt or "").translate(QUOTE_FOLD_TABLE).strip(QUOTE_EDGE_CHARS).split()


def find_excerpt(excerpt: Optional[str], text: str,
                 min_chars: int = MIN_QUOTE_MATCH_CHARS) -> Optional[str]:
    """The span of `text` that `excerpt` quotes, as whole words, ignoring case,
    quote style and spacing; None if it is not there. Returns the original
    wording, so the platform can highlight it with a plain substring search."""
    words = excerpt_words(excerpt)
    if not text or not words or len(" ".join(words)) < min_chars:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    if re.match(r"\w", words[0]):
        pattern = r"(?<!\w)" + pattern
    if re.search(r"\w$", words[-1]):
        pattern = pattern + r"(?!\w)"
    match = re.search(pattern, text.translate(QUOTE_FOLD_TABLE), re.IGNORECASE)
    return text[match.start():match.end()] if match else None


def find_excerpt_fuzzy(excerpt: Optional[str], text: str) -> Optional[str]:
    """The run of words in `text` most like `excerpt`, in the learner's own
    spelling, when it is at least FUZZY_QUOTE_MIN_RATIO alike; else None."""
    words = excerpt_words(excerpt)
    target = " ".join(words).lower()
    if not text or len(target) < MIN_QUOTE_MATCH_CHARS:
        return None
    tokens = list(re.finditer(r"\S+", text.translate(QUOTE_FOLD_TABLE)))
    best_ratio, best_span = 0.0, None
    for size in {max(1, len(words) - 1), len(words), len(words) + 1}:
        for first in range(0, len(tokens) - size + 1):
            start, end = tokens[first].start(), tokens[first + size - 1].end()
            candidate = " ".join(t.group() for t in tokens[first:first + size]).strip(QUOTE_EDGE_CHARS)
            ratio = difflib.SequenceMatcher(None, target, candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio, best_span = ratio, (start, end)
    if best_span is None or best_ratio < FUZZY_QUOTE_MIN_RATIO:
        return None
    start, end = best_span
    while start < end and text[start] in QUOTE_EDGE_CHARS:
        start += 1
    while end > start and text[end - 1] in QUOTE_EDGE_CHARS:
        end -= 1
    return text[start:end] or None


def split_anchor_line(tip_text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """(tip without ANCHOR lines or [U#] labels, label named, excerpt given)."""
    matches = list(ANCHOR_LINE_PATTERN.finditer(tip_text))
    label = excerpt = None
    if matches:
        last = matches[-1]
        label = learner_label(int(last.group(1)))
        excerpt = (last.group(2) or "").strip() or None
    cleaned = LEARNER_LABEL_PATTERN.sub("", ANCHOR_LINE_PATTERN.sub("", tip_text))
    tip = "\n\n".join(block.strip() for block in cleaned.split("\n\n") if block.strip())
    return tip, label, excerpt


def quoted_phrases(tip_text: str) -> List[str]:
    return [a or b for a, b in TIP_QUOTE_PATTERN.findall(tip_text)]


def str_or_none(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def resolve_anchor(tip_text: str, label: Optional[str], excerpt: Optional[str],
                   learner_messages: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Which learner message a tip is about, and the words it quotes from it.

    `learner_messages` maps each prompt label to its transcript document, in
    conversation order. Returns the anchor fields to store with the tip, all
    None when the conversation has no learner message to anchor to.
    """
    none = {FIELD_ANCHOR_TRANSCRIPT_ID: None, FIELD_ANCHOR_QUOTE: None, FIELD_ANCHOR_METHOD: None}
    if not learner_messages:
        return none

    def anchor(message: Dict[str, Any], method: str, quote: Optional[str]) -> Dict[str, Any]:
        return {FIELD_ANCHOR_TRANSCRIPT_ID: message.get("_id"),
                FIELD_ANCHOR_QUOTE: quote,
                FIELD_ANCHOR_METHOD: method}

    def quote_in(message: Dict[str, Any], min_chars: int = MIN_QUOTE_MATCH_CHARS) -> Optional[str]:
        text = message.get(TRANSCRIPT_TEXT_KEY) or ""
        for phrase in ([excerpt] if excerpt else []) + quoted_phrases(tip_text):
            found = find_excerpt(phrase, text, min_chars)
            if found:
                return found
        return None

    named = learner_messages.get(label) if label else None
    if named is not None:
        quote = (quote_in(named, MIN_NAMED_QUOTE_CHARS)
                 or find_excerpt_fuzzy(excerpt, named.get(TRANSCRIPT_TEXT_KEY) or ""))
        if quote:
            return anchor(named, ANCHOR_METHOD_MODEL, quote)

    # The tip's own quote, found verbatim in another message, outweighs a label
    # with nothing to back it. Latest message first: a learner who repeats a
    # phrase is most likely being coached on the time they just said it.
    for message in reversed(list(learner_messages.values())):
        quote = quote_in(message)
        if quote:
            return anchor(message, ANCHOR_METHOD_QUOTE_MATCH, quote)

    if named is not None:
        return anchor(named, ANCHOR_METHOD_MODEL, None)
    return anchor(list(learner_messages.values())[-1], ANCHOR_METHOD_LATEST, None)


app = FastAPI(title="Dynamic Tips API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TipsRequest(BaseModel):
    simulation_id: str
    # True when the learner asked for the tip, false when the platform
    # generated it unprompted (the notification bell shows for those).
    is_requested: bool = False

class TipsResponse(BaseModel):
    simulation_id: str
    tips: str
    transcript_count: int
    saved_to_db: bool
    timestamp: str
    error: str | None = None
    # Provenance: which SOP (if any) the tip was based on. Defaulted so existing
    # callers that ignore these fields keep working unchanged.
    sop_applied: bool = False
    sop_file_url: str | None = None
    # Whether a Continuous Learning instruction for this learner shaped the tip.
    cl_applied: bool = False
    # Wall-clock time the agent spent on this request, so response time can be
    # read off the payload instead of inferred from the caller's stopwatch.
    duration_ms: int | None = None
    # Which model wrote the tip — the Gemini model, or the OpenAI fallback.
    model: str | None = None
    # Task 295: the stored tip's id, and the transcript line it is about.
    tip_id: str | None = None
    anchor_transcript_id: str | None = None
    anchor_quote: str | None = None
    anchor_method: str | None = None
    is_requested: bool = False

class DynamicTips:

    # ── 1. Fetch all transcripts for a simulation ──────────────────────────
    def get_transcripts_by_simulation(self, simulation_id: str) -> List[Dict[Any, Any]]:
        """
        FIX: The ObjectId was being constructed but never used in the query.
        We now query with the ObjectId so MongoDB can match the stored reference type.
        We also fall back to the raw string if no results are found, to handle
        collections where simulation is stored as a plain string.
        """
        try: 
            object_id = ObjectId(simulation_id)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid simulation_id format: '{simulation_id}' is not a valid ObjectId"
            )

        # Try ObjectId first (the most common storage format for MongoDB references)
        results = list(
            transcripts_collection
            .find({"simulation": object_id})
            .sort("created_at", pymongo.ASCENDING)
        )

        # Fall back to string match if nothing found
        if not results:
            results = list(
                transcripts_collection
                .find({"simulation": simulation_id})
                .sort("created_at", pymongo.ASCENDING)
            )

        return results
 
    # ── 2. Combine transcripts into a single readable string ───────────────
    def combine_transcripts(self, transcripts: List[Dict[Any, Any]]) -> Tuple[str, Dict[str, Dict[Any, Any]]]:
        """(prompt transcript, {label: transcript doc} for the learner's messages).

        Each learner message is labelled [U1], [U2], … so the model can say
        which one its tip is about (task 295).
        """
        combined_text = ""
        learner_messages: Dict[str, Dict[Any, Any]] = {}
        for transcript in transcripts:
            from_type = transcript.get(TRANSCRIPT_FROM_TYPE_KEY, "Unknown")
            text = transcript.get(TRANSCRIPT_TEXT_KEY, "")

            if from_type == FROM_TYPE_USER:
                label = learner_label(len(learner_messages) + 1)
                learner_messages[label] = transcript
                combined_text += f"[{label}] {SPEAKER_USER}: {text}\n"
            elif from_type == FROM_TYPE_CHARACTER:
                combined_text += f"{SPEAKER_CHARACTER}: {text}\n"
            else:
                combined_text += f"{from_type}: {text}\n"

        return combined_text, learner_messages
 
    # ── 3. Call Gemini to generate tips ────────────────────────────────────
    def generate_tips(self, transcript: str,
                      sop_document: Optional[Dict[str, Any]] = None,
                      learner_instruction: Optional[Dict[str, Any]] = None) -> Tuple[str, str | None, str]:
        """Return (tips, error, model that answered)."""
        model_used = GEMINI_MODEL
        try:
            # When the service has an SOP / best-practice document attached, the
            # tip is drawn from that document's specific instructions instead of
            # generic communication coaching.
            sop_instruction = ""
            if sop_document:
                if sop_document.get("text"):
                    sop_body = f"""
            The standard operating procedure (SOP) for this service is:
            ---
            {sop_document['text']}
            ---
            """
                else:
                    sop_body = """
            The standard operating procedure (SOP) for this service is the attached document.
            """
                sop_instruction = f"""{sop_body}
            That SOP is the standard the learner is expected to meet. Prioritise it:
            - Prefer a tip about an SOP instruction the learner did not follow over a generic communication tip.
            - State what the SOP requires, using its own wording where it helps.
            - Give the learner the specific phrasing or action the SOP calls for.
            - Ignore any part of the SOP the conversation gave them no occasion to apply.
            - If they followed the SOP throughout, fall back to a general communication tip.
            """

            prompt = f"""
            You are a communication coach helping learners in Singapore improve their interpersonal effectiveness.
            {sop_instruction}
            {learner_instruction_block(learner_instruction)}
            Analyze ONLY the "HUMAN USER" messages in this conversation transcript:
            {transcript}

            Identify specific moments where the learner's communication style could be adjusted for better rapport, clarity, or empathy — then give immediately actionable tips they can apply in their very next message or real-world conversation.

            For each tip:
            1. Quote or reference the exact message or pattern you observed
            2. Explain what effect it likely had on the other person
            3. Give a concrete alternative phrasing or behavior they can use right away

            Provide 1 tip. Format each tip as a single self-contained block separated by a blank line:

            🔍 What I noticed: [specific behavior]
            💬 Why it matters: [impact on the listener/relationship]
            ✅ Try this instead: [ready-to-use alternative]

            Focus on: tone, word choice, empathy signals, question framing, and clarity of intent.
            Do NOT evaluate the "AI CHARACTER" messages.
            Do NOT add any preamble, heading, or closing text outside the tip blocks.
            Always keep each tip length to 500 characters or less.

            Each "HUMAN USER" message is labelled [U1], [U2], and so on. Never write these labels in the tip itself.
            After the tip, on its own final line, name the one message the tip is about and copy the exact words from it that you are commenting on, character for character, keeping any spelling mistakes (never correct or reword them):
            ANCHOR: U<number> | "<exact words from that message>"
            """

            # A PDF SOP rides along as inline data; a text SOP is already in the prompt.
            contents: List[Any] = []
            if sop_document and sop_document.get("data"):
                contents.append(genai_types.Part.from_bytes(
                    data=sop_document["data"], mime_type=sop_document["mime_type"]))
            contents.append(prompt)

            tips_text, model_error, model_used = first_answer(prompt, contents, sop_document)
            if not tips_text:
                return "", f"Error generating tips: {model_error}", model_used

            # FIX: Split on double newlines so each full tip block is one list item,
            # rather than splitting on every newline which fragments tips into individual lines.
            tips = "\n\n".join(block.strip() for block in tips_text.split("\n\n") if block.strip())
            return tips, None, model_used

        except Exception as e:
            # FIX: return a str, not []. TipsResponse.tips is typed `str`, so the
            # old `return []` made every generation failure a pydantic validation
            # error (HTTP 500) instead of a clean error payload.
            return "", f"Error generating tips: {e}", model_used
 
    # ── 4. Persist tips to MongoDB ─────────────────────────────────────────
    def save_tips_to_mongodb(self, simulation_id: str, tips: List[str],
                             sop_file_url: Optional[str] = None,
                             sop_applied: bool = False,
                             learner_instruction: Optional[Dict[str, Any]] = None,
                             anchor: Optional[Dict[str, Any]] = None,
                             last_transcript_id: Any = None,
                             is_requested: bool = False) -> List[ObjectId]:
        """Append the tips to the simulation's document; the new tips' ids, or [] on failure."""
        try:
            timestamp   = datetime.now()
            # Each entry records the SOP it came from, so a tip can be traced back
            # to the document that produced it. The `_id` lets the platform edit,
            # delete or comment on one tip, and the anchor fields place it on the
            # transcript (task 295).
            tip_entries = [{FIELD_TIP_ID: ObjectId(),
                            "tip": tip, "timestamp": timestamp,
                            FIELD_IS_REQUESTED: is_requested,
                            "sop_file_url": sop_file_url,
                            "sop_applied": sop_applied,
                            **cl_provenance(learner_instruction),
                            **(anchor or {}),
                            FIELD_LAST_TRANSCRIPT_ID: last_transcript_id} for tip in tips]

            simulation_oid = ObjectId(simulation_id)
            result = tips_collection.update_one(
                {FIELD_SIMULATION_ID: {"$in": [simulation_oid, simulation_id]}},
                {
                    "$push": {"dynamictips": {"$each": tip_entries}},
                    "$setOnInsert": {
                        FIELD_SIMULATION_ID: simulation_oid,
                        "created_at": timestamp,
                    },
                },
                upsert=True,
            )
            return [entry[FIELD_TIP_ID] for entry in tip_entries] if result.acknowledged else []

        except Exception as e:
            print(f"Error saving tips to MongoDB: {e}")
            return []

    # ── 5. Orchestrate the full pipeline ───────────────────────────────────
    def process_simulation(self, simulation_id: str,
                           background_tasks: Optional[BackgroundTasks] = None,
                           is_requested: bool = False) -> Dict[str, Any]:
        """Generate, persist and return one tip for the simulation.

        The reads that do not depend on each other run side by side, and the
        Continuous Learning audit writes (three updates that only record that
        the instruction was used) run after the response has gone out when a
        `background_tasks` is supplied. Each stage's time is logged in one
        line per request, so a slow call can be attributed at a glance.
        """
        started = time.perf_counter()
        timings: Dict[str, int] = {}

        def lap(stage: str) -> None:
            timings[stage] = int((time.perf_counter() - started) * 1000) - sum(timings.values())

        # Transcripts and the simulation document are independent lookups.
        with ThreadPoolExecutor(max_workers=2) as pool:
            transcripts_future = pool.submit(self.get_transcripts_by_simulation, simulation_id)
            simulation_future  = pool.submit(get_simulation, simulation_id)
            transcripts = transcripts_future.result()
            simulation  = simulation_future.result()
        lap("lookup")

        if not transcripts:
            return {
                "simulation_id": simulation_id,
                "tips": "",
                "transcript_count": 0,
                "saved_to_db": False,
                "timestamp": datetime.now().isoformat(),
                "error": "No transcripts found for this simulation.",
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }

        # Optional SOP grounding and Continuous Learning context. A missing
        # simulation, missing field, or unusable file is not an error — the tip
        # is simply generated the generic way. The two are independent of each
        # other, so the CL lookup overlaps the SOP download.
        sop_document = None
        sop_file_url = None
        learner_instruction = None
        if simulation:
            with ThreadPoolExecutor(max_workers=2) as pool:
                instruction_future = pool.submit(get_learner_instruction, simulation)
                sop_future         = pool.submit(resolve_sop_document, simulation_id, simulation)
                learner_instruction        = instruction_future.result()
                sop_file_url, sop_document = sop_future.result()
            if learner_instruction:
                print(f"CL instruction for simulation {simulation_id}: {learner_instruction.get('stage')} v{learner_instruction.get('version')}")
        else:
            print(f"Simulation {simulation_id} not found in {SIMULATIONS_COLLECTION}; "
                  f"generating a generic tip")
        lap("context")

        combined, learner_messages = self.combine_transcripts(transcripts)
        tips, error, model_used = self.generate_tips(combined, sop_document, learner_instruction)
        lap("gemini")

        anchor: Dict[str, Any] = {}
        if tips:
            tips, label, excerpt = split_anchor_line(tips)
            anchor = resolve_anchor(tips, label, excerpt, learner_messages)

        tip_ids: List[ObjectId] = []
        if tips:
            tip_ids = self.save_tips_to_mongodb(
                simulation_id, [tips],
                sop_file_url=sop_file_url,
                sop_applied=bool(sop_document),
                learner_instruction=learner_instruction,
                anchor=anchor,
                last_transcript_id=transcripts[-1].get("_id"),
                is_requested=is_requested,
            )
        saved = bool(tip_ids)
        if saved and learner_instruction:
            if background_tasks is not None:
                background_tasks.add_task(cl_mark_applied, learner_instruction, simulation_id)
            else:
                cl_mark_applied(learner_instruction, simulation_id)
        lap("save")

        duration_ms = int((time.perf_counter() - started) * 1000)
        print(f"Tips for simulation {simulation_id}: {duration_ms}ms "
              f"(lookup={timings['lookup']} context={timings['context']} "
              f"gemini={timings['gemini']} save={timings['save']}) "
              f"transcripts={len(transcripts)} sop={bool(sop_document)} cl={bool(learner_instruction)} "
              f"model={model_used} thinking_budget={GEMINI_THINKING_BUDGET} "
              f"anchor={anchor.get(FIELD_ANCHOR_METHOD)} requested={is_requested}"
              + (f" error={error}" if error else ""))

        return {
            "simulation_id":    simulation_id,
            "tips":             tips,
            "transcript_count": len(transcripts),
            "saved_to_db":      saved,
            "timestamp":        datetime.now().isoformat(),
            "error":            error,
            "sop_applied":      bool(sop_document),
            "sop_file_url":     sop_file_url,
            "cl_applied":       bool(learner_instruction),
            "duration_ms":      duration_ms,
            "model":            model_used,
            "tip_id":           str(tip_ids[0]) if tip_ids else None,
            "anchor_transcript_id": str_or_none(anchor.get(FIELD_ANCHOR_TRANSCRIPT_ID)),
            "anchor_quote":     anchor.get(FIELD_ANCHOR_QUOTE),
            "anchor_method":    anchor.get(FIELD_ANCHOR_METHOD),
            "is_requested":     is_requested,
        }

advisor = DynamicTips()

@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/tips", response_model=TipsResponse)
def get_tips(request: TipsRequest, background_tasks: BackgroundTasks):
    try:
        if not request.simulation_id:
            raise HTTPException(status_code=400, detail="simulation_id is required.")

        result = advisor.process_simulation(request.simulation_id, background_tasks,
                                            is_requested=request.is_requested)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"UNHANDLED ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tips/{simulation_id}", response_model=TipsResponse)
def get_tips_by_id(simulation_id: str, background_tasks: BackgroundTasks,
                   is_requested: bool = False):
    """
    Convenience POST endpoint — same behaviour as /tips but with
    simulation_id in the URL path instead of the request body, and
    is_requested as a query parameter.
    """
    result = advisor.process_simulation(simulation_id, background_tasks,
                                        is_requested=is_requested)

    if result.get("error") and not result.get("tips"):
        raise HTTPException(status_code=502, detail=result["error"])
 
    return result


# ── Debug endpoint: check how simulation_id is stored in MongoDB ──────────
@app.get("/debug/raw/{simulation_id}")
def debug_raw(simulation_id: str):
    results = {}

    count_string = transcripts_collection.count_documents({"simulation": simulation_id})
    results["match_as_string"] = count_string 

    try:
        oid = ObjectId(simulation_id)
        count_oid = transcripts_collection.count_documents({"simulation": oid})
        results["match_as_objectid"] = count_oid 
    except Exception as e:
        results["objectid_error"] = str(e)
    
    return results 

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)