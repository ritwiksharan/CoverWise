import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import ConversationalOrchestrator

# ADK orchestrator is optional - requires google-adk package
try:
    from agents.adk_orchestrator import ADKOrchestrator
    adk_orchestrator = ADKOrchestrator()
    ADK_AVAILABLE = True
    print("Google ADK loaded successfully")
except Exception as e:
    print("ADK not available: " + str(e))
    adk_orchestrator = None
    ADK_AVAILABLE = False

app = FastAPI(title="CoverWise - AI Health Insurance Advisor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conv_orchestrator = ConversationalOrchestrator()

FRONTEND_PATH = "frontend" if os.path.exists("frontend") else "../frontend"


class ChatRequest(BaseModel):
    user_id: str
    message: str

class StartRequest(BaseModel):
    user_id: str

class UserProfile(BaseModel):
    user_id: str
    zip_code: str
    age: int
    income: float
    household_size: int
    drugs: list[str] = []
    doctors: list[str] = []
    utilization: str = "sometimes"
    tobacco_use: bool = False
    is_premium: bool = False
    message: Optional[str] = None

class IntakeMessage(BaseModel):
    user_id: str
    session_id: str
    message: str

class IntakeStart(BaseModel):
    user_id: str
    session_id: str


# ── CONVERSATIONAL INTAKE ──────────────────────────────────────────────────

@app.post("/api/start")
async def start(req: StartRequest):
    try:
        return await conv_orchestrator.start(req.user_id)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        return await conv_orchestrator.chat(req.user_id, req.message)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset(req: StartRequest):
    return conv_orchestrator.reset(req.user_id)


# ── ADK DEEP ANALYSIS ─────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(profile: UserProfile):
    try:
        if not ADK_AVAILABLE or adk_orchestrator is None:
            raise HTTPException(status_code=503, detail="ADK not available locally. Deploy to Cloud Run to use this endpoint.")
        result = await adk_orchestrator.analyze(profile.dict())
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── ADK INTAKE ────────────────────────────────────────────────────────────

@app.post("/api/intake/start")
async def intake_start(req: IntakeStart):
    try:
        from agents.intake_agent import start_session
        return await start_session(req.user_id, req.session_id)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/intake/message")
async def intake_message(req: IntakeMessage):
    try:
        from agents.intake_agent import send_message
        return await send_message(req.user_id, req.session_id, req.message)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── UTILITY ───────────────────────────────────────────────────────────────

@app.get("/api/memory/{user_id}")
async def get_memory(user_id: str):
    from memory.mem0_client import get_user_memories
    return {"memories": get_user_memories(user_id)}

@app.get("/api/cache/stats")
async def cache_stats():
    from cache.cache_manager import get_cache_stats
    return get_cache_stats()

@app.get("/api/health")
async def health():
    return {"status": "ok", "adk_available": ADK_AVAILABLE}

app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
