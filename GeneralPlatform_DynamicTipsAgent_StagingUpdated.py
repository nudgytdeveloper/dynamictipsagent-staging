import pymongo
from google import genai
from google.genai import types as genai_types
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import os
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
# ============================================================================
GEMINI_MODEL             = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_THINKING_BUDGET   = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))
# Bounds a hung generation; the old client would wait on it indefinitely.
GEMINI_TIMEOUT_SECONDS   = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "60"))
GEMINI_TEMPERATURE       = 0.5
# Thinking tokens (when a budget is set) are charged against this cap too.
GEMINI_MAX_OUTPUT_TOKENS = 3000

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


def fetch_sop_document(url: str) -> Optional[Dict[str, Any]]:
    """Download the SOP file and return a part Gemini can consume.

    Returns {"url", "mime_type", "data"} for a PDF, {"url", "text"} for plain
    text, or None if the file is unreachable, oversized, or an unsupported type.
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


app = FastAPI(title="Dynamic Tips API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TipsRequest(BaseModel):
    simulation_id: str

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
    def combine_transcripts(self, transcripts: List[Dict[Any, Any]]) -> str:
        combined_text = ""
        for transcript in transcripts:
            from_type = transcript.get("from_type", "Unknown")
 
            if from_type == "User":
                speaker = "HUMAN USER"
            elif from_type == "Character":
                speaker = "AI CHARACTER"
            else:
                speaker = from_type
 
            text = transcript.get("dialogue_value", "")
            combined_text += f"{speaker}: {text}\n"
 
        return combined_text
 
    # ── 3. Call Gemini to generate tips ────────────────────────────────────
    def generate_tips(self, transcript: str,
                      sop_document: Optional[Dict[str, Any]] = None,
                      learner_instruction: Optional[Dict[str, Any]] = None) -> Tuple[str, str | None]:
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
            """

            # A PDF SOP rides along as inline data; a text SOP is already in the prompt.
            contents: List[Any] = []
            if sop_document and sop_document.get("data"):
                contents.append(genai_types.Part.from_bytes(
                    data=sop_document["data"], mime_type=sop_document["mime_type"]))
            contents.append(prompt)

            response = gemini_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
                ),
            )

            tips_text = response_text(response)
            if not tips_text:
                return "", f"Error generating tips: Gemini returned no text (finish_reason={finish_reason(response)})"

            # FIX: Split on double newlines so each full tip block is one list item,
            # rather than splitting on every newline which fragments tips into individual lines.
            tips = "\n\n".join(block.strip() for block in tips_text.split("\n\n") if block.strip())
            return tips, None

        except Exception as e:
            # FIX: return a str, not []. TipsResponse.tips is typed `str`, so the
            # old `return []` made every generation failure a pydantic validation
            # error (HTTP 500) instead of a clean error payload.
            return "", f"Error generating tips: {e}"
 
    # ── 4. Persist tips to MongoDB ─────────────────────────────────────────
    def save_tips_to_mongodb(self, simulation_id: str, tips: List[str],
                             sop_file_url: Optional[str] = None,
                             sop_applied: bool = False,
                             learner_instruction: Optional[Dict[str, Any]] = None) -> bool:
        try:
            timestamp   = datetime.now()
            # Each entry records the SOP it came from, so a tip can be traced back
            # to the document that produced it.
            tip_entries = [{"tip": tip, "timestamp": timestamp,
                            "sop_file_url": sop_file_url,
                            "sop_applied": sop_applied,
                            **cl_provenance(learner_instruction)} for tip in tips]

            result = tips_collection.update_one(
                {"simulation_id": simulation_id},
                {
                    "$push": {"dynamictips": {"$each": tip_entries}},
                    "$setOnInsert": {
                        "simulation_id": simulation_id,
                        "created_at": timestamp,
                    },
                },
                upsert=True,
            )
            return result.acknowledged

        except Exception as e:
            print(f"Error saving tips to MongoDB: {e}")
            return False

    # ── 5. Orchestrate the full pipeline ───────────────────────────────────
    def process_simulation(self, simulation_id: str,
                           background_tasks: Optional[BackgroundTasks] = None) -> Dict[str, Any]:
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

        combined    = self.combine_transcripts(transcripts)
        tips, error = self.generate_tips(combined, sop_document, learner_instruction)
        lap("gemini")

        saved = False
        if tips:
            saved = self.save_tips_to_mongodb(
                simulation_id, [tips],
                sop_file_url=sop_file_url,
                sop_applied=bool(sop_document),
                learner_instruction=learner_instruction,
            )
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
              f"model={GEMINI_MODEL} thinking_budget={GEMINI_THINKING_BUDGET}"
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

        result = advisor.process_simulation(request.simulation_id, background_tasks)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"UNHANDLED ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tips/{simulation_id}", response_model=TipsResponse)
def get_tips_by_id(simulation_id: str, background_tasks: BackgroundTasks):
    """
    Convenience POST endpoint — same behaviour as /tips but with
    simulation_id in the URL path instead of the request body.
    """
    result = advisor.process_simulation(simulation_id, background_tasks)

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