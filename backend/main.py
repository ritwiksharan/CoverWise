import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import ConversationalOrchestrator

app = FastAPI(title="CoverWise")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

orchestrator = ConversationalOrchestrator()

class ChatRequest(BaseModel):
    user_id: str
    message: str

class StartRequest(BaseModel):
    user_id: str

@app.post("/api/start")
async def start(req: StartRequest):
    try:
        return await orchestrator.start(req.user_id)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        return await orchestrator.chat(req.user_id, req.message)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset(req: StartRequest):
    return orchestrator.reset(req.user_id)

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/cache/stats")
async def cache_stats():
    from cache.cache_manager import get_cache_stats
    return get_cache_stats()

FRONTEND_PATH = "frontend" if os.path.exists("frontend") else "../frontend"
app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
