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
from auth.router import router as auth_router
from auth.db import init_db

app = FastAPI(title="CoverWise - AI Health Insurance Advisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth")

orchestrator = ADKOrchestrator()
init_db()

FRONTEND_PATH = "frontend" if os.path.exists("frontend") else "../frontend"


@app.on_event("startup")
async def _seed_formulary_background():
    """Pre-seed formulary RAG store for the most common issuers at startup."""
    def _seed():
        try:
            from rag.formulary_store import seed_issuer
            # BCBS IL (36096) is the highest-volume issuer with a working MRF endpoint
            seed_issuer("36096", blocking=False)
        except Exception as e:
            print(f"[startup] formulary seed error: {e}")
    import threading
    threading.Thread(target=_seed, daemon=True).start()


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


class SpecialtySearchRequest(BaseModel):
    user_id: str
    condition: str
    zip_code: str
    state: str = ""
    plan_ids: list[str] = []
    is_premium: bool = False


class ProcedureCostRequest(BaseModel):
    user_id: str
    procedure_key: str
    plans: list = []


class HospitalSearchRequest(BaseModel):
    user_id: str
    name: str
    state: str
    city: str = ""
    zip_code: str = ""
    plan_ids: list[str] = []


class InsuranceQARequest(BaseModel):
    user_id: str
    question: str
    is_premium: bool = False


class DoctorSearchRequest(BaseModel):
    name: str
    city: str = ""
    state: str = ""
    zip_code: str = ""


class PlanProvidersRequest(BaseModel):
    plan_id: str
    zip_code: str
    specialty: str = "Internal Medicine"
    limit: int = 20


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


# ── SPECIALIST SEARCH ─────────────────────────────────────────────────────────

@app.post("/api/specialty-search")
async def specialty_search(req: SpecialtySearchRequest):
    """Find local specialists for a condition and check coverage on user's plans."""
    try:
        from agents.tools import find_specialists_for_condition
        from tools.gov_apis import get_fips_from_zip, _fips_to_state
        state = req.state
        if not state:
            fips = get_fips_from_zip(req.zip_code)
            state = _fips_to_state(fips) if fips else "US"
        result = await asyncio.to_thread(
            find_specialists_for_condition,
            req.condition, req.zip_code, state, req.plan_ids
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── PROCEDURE COST ESTIMATOR ──────────────────────────────────────────────────

@app.post("/api/procedure-cost")
async def procedure_cost(req: ProcedureCostRequest):
    """Estimate patient out-of-pocket cost for a procedure across plans."""
    try:
        from tools.gov_apis import estimate_procedure_oop
        result = await asyncio.to_thread(estimate_procedure_oop, req.procedure_key, req.plans)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── HOSPITAL SEARCH ───────────────────────────────────────────────────────────

@app.get("/api/hospitals/nearby/{zip_code}")
async def hospitals_nearby(zip_code: str, plan_ids: str = ""):
    """Return hospitals near a ZIP code (location-based, no name required)."""
    try:
        from tools.gov_apis import search_hospitals_nearby, check_doctor_in_plan_network
        hospitals = await asyncio.to_thread(search_hospitals_nearby, zip_code)
        pids = [p.strip() for p in plan_ids.split(",") if p.strip()][:3]
        results = []
        for h in hospitals:
            npi = h.get("npi")
            network_status = {}
            if npi and pids:
                for pid in pids:
                    net = await asyncio.to_thread(
                        check_doctor_in_plan_network, pid, str(npi), zip_code
                    )
                    network_status[pid] = net
            results.append({**h, "network_status": network_status})
        return {"hospitals": results, "zip_code": zip_code}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hospital-search")
async def hospital_search(req: HospitalSearchRequest):
    """Search for hospitals by name and check in-network status."""
    try:
        from tools.gov_apis import search_hospitals, check_doctor_in_plan_network, get_fips_from_zip, _fips_to_state
        state = req.state
        if not state and req.zip_code:
            fips = get_fips_from_zip(req.zip_code)
            state = _fips_to_state(fips) if fips else "US"

        hospitals = await asyncio.to_thread(search_hospitals, req.name, state, req.city)

        # Check network status for up to 3 plans
        results = []
        for h in hospitals:
            npi = h.get("npi")
            network_status = {}
            if npi and req.plan_ids:
                for pid in req.plan_ids[:3]:
                    net = await asyncio.to_thread(
                        check_doctor_in_plan_network, pid, str(npi), req.zip_code or ""
                    )
                    network_status[pid] = net
            results.append({**h, "network_status": network_status})

        return {"hospitals": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── DOCTOR SEARCH ─────────────────────────────────────────────────────────────

@app.post("/api/doctor-search")
async def doctor_search(req: DoctorSearchRequest):
    """
    Look up a doctor by name via the NPPES NPI Registry.
    Returns NPI, specialty, city/state, phone, credential, and up to 3 candidate matches.
    Optionally pass city/state/zip to narrow results.
    """
    try:
        from tools.gov_apis import lookup_npi_registry, get_fips_from_zip, _fips_to_state
        state = req.state
        if not state and req.zip_code:
            fips = get_fips_from_zip(req.zip_code)
            state = _fips_to_state(fips) if fips else ""
        result = await asyncio.to_thread(lookup_npi_registry, req.name, req.city, state)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── PLAN PROVIDERS ────────────────────────────────────────────────────────────

@app.post("/api/plan-providers")
async def plan_providers(req: PlanProvidersRequest):
    """
    Return providers for a plan.
    Fetches the plan's issuer name and provider directory URL from CMS,
    then returns NPPES providers for the requested specialty near the ZIP.
    Network membership must be confirmed via the insurer's directory URL.
    """
    try:
        from tools.gov_apis import get_plan_providers
        result = await asyncio.to_thread(
            get_plan_providers, req.plan_id, req.zip_code, req.specialty, req.limit
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── INSURANCE Q&A (ADK agent — Gemini decides which tools to call) ────────────

@app.post("/api/insurance-qa")
async def insurance_qa(req: InsuranceQARequest):
    """
    Answer free-form health insurance questions using a Google ADK agent.
    Gemini 2.0 Flash decides which tool(s) to call and synthesises the answer.
    Restricted to health insurance topics only.
    """
    try:
        from agents.insurance_qa_agent import get_agent, ADK_AVAILABLE

        if not ADK_AVAILABLE:
            return {
                "answer": "The AI advisor is temporarily unavailable. Please try again shortly.",
                "tool_calls": [],
                "data": {},
                "tool_used": "none",
            }

        agent = get_agent()
        result = await agent.ask(req.user_id, req.question)
        return {
            "answer": result["answer"],
            "tool_calls": result.get("tool_calls", []),
            "data": {},
            "tool_used": ", ".join(result.get("tool_calls", [])) or "gemini",
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "answer": f"Sorry, I encountered an error: {e}",
            "data": {},
            "tool_used": "error",
            "tool_calls": [],
        }


# ── UTILITY ───────────────────────────────────────────────────────────────────

@app.get("/api/memory/{user_id}")
async def get_memory(user_id: str):
    from memory.mem0_client import get_user_memories
    return {"memories": get_user_memories(user_id)}


@app.get("/api/cache/stats")
async def cache_stats():
    from cache.cache_manager import get_cache_stats
    return get_cache_stats()


@app.get("/api/formulary/stats")
async def formulary_stats():
    """Return stats on the RAG formulary index."""
    from rag.formulary_store import get_stats
    return await asyncio.to_thread(get_stats)


@app.post("/api/formulary/seed/{issuer_id}")
async def formulary_seed(issuer_id: str):
    """
    Seed the formulary RAG store for a 5-digit issuer ID (e.g. 36096 for BCBS IL).
    Downloads and indexes the insurer's machine-readable formulary JSON.
    Runs in the background — check /api/formulary/stats to see progress.
    """
    from rag.formulary_store import seed_issuer, ISSUER_MRF_URLS
    if issuer_id not in ISSUER_MRF_URLS:
        raise HTTPException(status_code=404, detail=f"Issuer {issuer_id} not in known MRF index")
    seed_issuer(issuer_id, blocking=False)
    return {"status": "seeding_started", "issuer_id": issuer_id}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
