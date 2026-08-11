import pymongo
import google.generativeai as genai
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import os
import urllib.request
import urllib.error
from urllib.parse import urlparse, unquote
from bson import ObjectId

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MONGO_URI              = os.environ.get("MONGO_URI")
DB_NAME                = os.environ.get("DB_NAME", "your_db_name")
TRANSCRIPTS_COLLECTION = os.environ.get("TRANSCRIPTS_COLLECTION", "transcripts")
TIPS_COLLECTION        = os.environ.get("TIPS_COLLECTION", "tips")
# Needed to resolve the service's uploaded SOP: simulation -> service_level -> file URL.
SIMULATIONS_COLLECTION    = os.environ.get("SIMULATIONS_COLLECTION", "simulations")
SERVICE_LEVELS_COLLECTION = os.environ.get("SERVICE_LEVELS_COLLECTION", "service_levels")
GEMINI_API_KEY         = os.environ.get("GEMINI_API_KEY")

client                 = pymongo.MongoClient(MONGO_URI)
db                     = client[DB_NAME]
transcripts_collection = db[TRANSCRIPTS_COLLECTION]
tips_collection        = db[TIPS_COLLECTION]
simulations_collection    = db[SIMULATIONS_COLLECTION]
service_levels_collection = db[SERVICE_LEVELS_COLLECTION]

genai.configure(api_key=GEMINI_API_KEY)


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
        with urllib.request.urlopen(url, timeout=SOP_DOWNLOAD_TIMEOUT_SECONDS) as response:
            # Read one byte past the cap so an oversized file is detected rather
            # than silently truncated into a malformed PDF.
            payload = response.read(SOP_MAX_BYTES + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
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

class DynamicTips:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")
 
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
                      sop_document: Optional[Dict[str, Any]] = None) -> Tuple[str, str | None]:
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
            if sop_document and sop_document.get("data"):
                content = [
                    {"mime_type": sop_document["mime_type"], "data": sop_document["data"]},
                    prompt,
                ]
            else:
                content = prompt

            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=3000,
                ),
            )

            tips_text = response.text.strip()

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
                             sop_applied: bool = False) -> bool:
        try:
            timestamp   = datetime.now()
            # Each entry records the SOP it came from, so a tip can be traced back
            # to the document that produced it.
            tip_entries = [{"tip": tip, "timestamp": timestamp,
                            "sop_file_url": sop_file_url,
                            "sop_applied": sop_applied} for tip in tips]

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
    def process_simulation(self, simulation_id: str) -> Dict[str, Any]:
        transcripts = self.get_transcripts_by_simulation(simulation_id)

        if not transcripts:
            return {
                "simulation_id": simulation_id,
                "tips": "",
                "transcript_count": 0,
                "saved_to_db": False,
                "timestamp": datetime.now().isoformat(),
                "error": "No transcripts found for this simulation.",
            }

        # Optional SOP grounding. A missing simulation, missing field, or unusable
        # file is not an error — the tip is simply generated the generic way.
        sop_document = None
        sop_file_url = None
        simulation = get_simulation(simulation_id)
        if simulation:
            sop_file_url = get_sop_tips_file_url(simulation)
            if sop_file_url:
                print(f"SOP file for simulation {simulation_id}: {sop_file_url}")
                sop_document = fetch_sop_document(sop_file_url)
                if not sop_document:
                    print(f"SOP file unusable for simulation {simulation_id}; "
                          f"falling back to a generic tip")
            else:
                print(f"No SOP file configured for simulation {simulation_id}")
        else:
            print(f"Simulation {simulation_id} not found in {SIMULATIONS_COLLECTION}; "
                  f"generating a generic tip")

        combined    = self.combine_transcripts(transcripts)
        tips, error = self.generate_tips(combined, sop_document)

        saved = False
        if tips:
            saved = self.save_tips_to_mongodb(
                simulation_id, [tips],
                sop_file_url=sop_file_url,
                sop_applied=bool(sop_document),
            )

        return {
            "simulation_id":    simulation_id,
            "tips":             tips,
            "transcript_count": len(transcripts),
            "saved_to_db":      saved,
            "timestamp":        datetime.now().isoformat(),
            "error":            error,
            "sop_applied":      bool(sop_document),
            "sop_file_url":     sop_file_url,
        }

advisor = DynamicTips() 

@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok"}
 

@app.post("/tips", response_model=TipsResponse)
def get_tips(request: TipsRequest):
    try:
        if not request.simulation_id:
            raise HTTPException(status_code=400, detail="simulation_id is required.")
        
        result = advisor.process_simulation(request.simulation_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"UNHANDLED ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tips/{simulation_id}", response_model=TipsResponse)
def get_tips_by_id(simulation_id: str):
    """
    Convenience POST endpoint — same behaviour as /tips but with
    simulation_id in the URL path instead of the request body.
    """
    result = advisor.process_simulation(simulation_id)
 
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
    uvicorn.run("dynamic_tips_api:app", host="0.0.0.0", port=port)