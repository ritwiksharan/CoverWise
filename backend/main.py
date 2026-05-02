import os
import json
import asyncio
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from agents.adk_orchestrator import ADKOrchestrator
from agents.intake_agent import start_session, send_message as intake_send

app = FastAPI(title="CoverWise - AI Health Insurance Advisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = ADKOrchestrator()

FRONTEND_PATH = "frontend" if os.path.exists("frontend") else "../frontend"


class UserProfile(BaseModel):
    user_id: str
    zip_code: str
    age: int
    income: float
    household_size: int
    drugs: list[str] = []
    doctors: list[str] = []
    utilization: str = "sometimes"   # rarely / sometimes / frequently / chronic
    tobacco_use: bool = False
    is_premium: bool = False
    message: Optional[str] = None


class ChatMessage(BaseModel):
    user_id: str
    message: str
    profile: Optional[UserProfile] = None


class IntakeStart(BaseModel):
    user_id: str
    session_id: str


class IntakeMessage(BaseModel):
    user_id: str
    session_id: str
    message: str


# ── ANALYSIS ──────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(profile: UserProfile):
    try:
        result = await orchestrator.analyze(profile.dict())
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(msg: ChatMessage):
    try:
        result = await orchestrator.chat(
            msg.user_id, msg.message, msg.profile.dict() if msg.profile else None
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── CONVERSATIONAL INTAKE (Google ADK) ────────────────────────────────────────

@app.post("/api/intake/start")
async def intake_start(req: IntakeStart):
    """Start a new ADK conversational intake session."""
    try:
        result = await start_session(req.user_id, req.session_id)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intake/message")
async def intake_message(req: IntakeMessage):
    """Send a message to the intake agent; returns reply + profile when ready."""
    try:
        result = await intake_send(req.user_id, req.session_id, req.message)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── UTILITY ───────────────────────────────────────────────────────────────────

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
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
