import pymongo
import google.generativeai as genai
from datetime import datetime 
from typing import List, Dict, Any, Tuple
import os 
from bson import ObjectId 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel 

MONGO_URI              = os.environ.get("MONGO_URI")
DB_NAME                = os.environ.get("DB_NAME", "your_db_name")
TRANSCRIPTS_COLLECTION = os.environ.get("TRANSCRIPTS_COLLECTION", "transcripts")
TIPS_COLLECTION        = os.environ.get("TIPS_COLLECTION", "tips")
GEMINI_API_KEY         = os.environ.get("GEMINI_API_KEY")

client                 = pymongo.MongoClient(MONGO_URI)
db                     = client[DB_NAME]
transcripts_collection = db[TRANSCRIPTS_COLLECTION]
tips_collection        = db[TIPS_COLLECTION]
 
genai.configure(api_key=GEMINI_API_KEY)

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
    tips: List[str]
    transcript_count: int 
    saved_to_db: bool 
    timestamp: str
    error: str | None = None

class DynamicTips:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")
 
    # ── 1. Fetch all transcripts for a simulation ──────────────────────────
    def get_transcripts_by_simulation(self, simulation_id: str) -> List[Dict[Any, Any]]:
        try: 
            object_id = ObjectId(simulation_id)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid simulation_id format: '{simulation_id}' is not a valid ObjectId")
        return list(
            transcripts_collection
            .find({"simulation": simulation_id})
            .sort("created_at", pymongo.ASCENDING)
        )
 
    # ── 2. Combine transcripts into a single readable string ───────────────
    def combine_transcripts(self, transcripts: List[Dict[Any, Any]]) -> str:
        combined_text = ""
        for transcript in transcripts:
            from_type = transcript.get("from_type", "Unknown")
 
            if from_type == "User":
                speaker = "HUMAN USER"
            elif from_type == "Character":
                speaker = "AI CHARACTER"      # ← fixed: was `==` instead of `=`
            else:
                speaker = from_type
 
            text = transcript.get("dialogue_value", "")
            combined_text += f"{speaker}: {text}\n"
 
        return combined_text
 
    # ── 3. Call Gemini to generate tips ────────────────────────────────────
    def generate_tips(self, transcript: str) -> Tuple[List[str], str | None]:
        try:
            prompt = f"""
            You are a communication coach helping learners in Singapore improve their interpersonal effectiveness.

            Analyze ONLY the "HUMAN USER" messages in this conversation transcript:
            {transcript}

            Identify specific moments where the learner's communication style could be adjusted for better rapport, clarity, or empathy — then give immediately actionable tips they can apply in their very next message or real-world conversation.

            For each tip:
            1. Quote or reference the exact message or pattern you observed
            2. Explain what effect it likely had on the other person
            3. Give a concrete alternative phrasing or behavior they can use right away

            Provide 3–5 tips. Format each as:

            🔍 What I noticed: [specific behavior or phrasing]
            💬 Why it matters: [impact on the listener/relationship]
            ✅ Try this instead: [ready-to-use alternative]

            Focus on: tone, word choice, empathy signals, question framing, and clarity of intent.
            Do NOT evaluate the "AI CHARACTER" messages.
            """

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=3000,
                ),
            )
 
            tips_text = response.text.strip()
            tips = [tip.strip() for tip in tips_text.split("\n") if tip.strip()]
            return tips, None
 
        except Exception as e:
            return [], f"Error generating tips: {e}"
 
    # ── 4. Persist tips to MongoDB ─────────────────────────────────────────
    def save_tips_to_mongodb(self, simulation_id: str, tips: List[str]) -> bool:
        try:
            timestamp   = datetime.now()
            tip_entries = [{"tip": tip, "timestamp": timestamp} for tip in tips]
 
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
                "tips": [],
                "transcript_count": 0,
                "saved_to_db": False,
                "timestamp": datetime.now().isoformat(),
                "error": "No transcripts found for this simulation.",
            }
 
        combined   = self.combine_transcripts(transcripts)
        tips, error = self.generate_tips(combined)
 
        saved = False
        if tips:
            saved = self.save_tips_to_mongodb(simulation_id, tips)
 
        return {
            "simulation_id":    simulation_id,
            "tips":             tips,
            "transcript_count": len(transcripts),
            "saved_to_db":      saved,
            "timestamp":        datetime.now().isoformat(),
            "error":            error,
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
        # This logs the full traceback to Render logs instead of crashing
        import traceback
        print(f"UNHANDLED ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/tips/{simulation_id}", response_model=TipsResponse)
def get_tips_by_id(simulation_id: str):
    """
    Convenience GET endpoint — same behaviour, simulation_id in the URL path.
    Useful for simple button clicks that just fire a GET request.
    """
    result = advisor.process_simulation(simulation_id)
 
    if result.get("error") and not result.get("tips"):
        raise HTTPException(status_code=502, detail=result["error"])
 
    return result

@app.get("/debug/raw/{simulation_id}")
def debug_raw(simulation_id: str):
    from bson import ObjectId

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
    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 locally
    uvicorn.run("dynamic_tips_api:app", host="0.0.0.0", port=port)