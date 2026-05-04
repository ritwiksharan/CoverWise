import os
import asyncio
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

# ── Core conversational orchestrator (always available) ───────────────────────
from agents.orchestrator import ConversationalOrchestrator

# ── ADK orchestrator (optional - needs google-adk) ────────────────────────────
try:
    from agents.adk_orchestrator import ADKOrchestrator
    adk_orchestrator = ADKOrchestrator()
    ADK_AVAILABLE = True
    print("Google ADK loaded successfully")
except Exception as e:
    print("ADK not available: " + str(e))
    adk_orchestrator = None
    ADK_AVAILABLE = False

# ── Insurance Q&A agent (optional - needs google-adk) ────────────────────────
try:
    from agents.insurance_qa_agent import get_agent as get_qa_agent
    QA_AVAILABLE = ADK_AVAILABLE
except Exception as e:
    print("QA agent not available: " + str(e))
    QA_AVAILABLE = False

app = FastAPI(title="CoverWise - AI Health Insurance Advisor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

conv_orchestrator = ConversationalOrchestrator()
FRONTEND_PATH = "frontend" if os.path.exists("frontend") else "../frontend"


# ── Startup: pre-seed formulary RAG (background, non-blocking) ────────────────
@app.on_event("startup")
async def seed_formulary():
    def _seed():
        try:
            from rag.formulary_store import seed_issuer
            seed_issuer("36096", blocking=False)
        except Exception as e:
            print("[startup] formulary seed skipped: " + str(e))
    import threading
    threading.Thread(target=_seed, daemon=True).start()


# ── Models ────────────────────────────────────────────────────────────────────
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
    drugs: List[str] = []
    doctors: List[str] = []
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

class QARequest(BaseModel):
    user_id: str
    question: str
    is_premium: bool = False

class SpecialtySearchRequest(BaseModel):
    user_id: str
    condition: str
    zip_code: str
    state: str = ""
    plan_ids: List[str] = []
    is_premium: bool = False

class ProcedureCostRequest(BaseModel):
    user_id: str
    procedure_key: str
    plans: List[dict] = []

class HospitalSearchRequest(BaseModel):
    user_id: str
    name: str
    state: str = ""
    city: str = ""
    zip_code: str = ""
    plan_ids: List[str] = []


# ── Conversational intake ─────────────────────────────────────────────────────
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


# ── ADK intake ────────────────────────────────────────────────────────────────
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


# ── ADK deep analysis ─────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(profile: UserProfile):
    try:
        if not ADK_AVAILABLE or adk_orchestrator is None:
            raise HTTPException(status_code=503, detail="ADK not available locally.")
        result = await adk_orchestrator.analyze(profile.dict())
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Insurance Q&A (Bhuvi's ADK agent) ────────────────────────────────────────
@app.post("/api/insurance-qa")
async def insurance_qa(req: QARequest):
    try:
        if not QA_AVAILABLE:
            return {"answer": "QA agent not available.", "tool_calls": [], "tool_used": "none"}
        agent = get_qa_agent()
        result = await agent.ask(req.user_id, req.question)
        return {"answer": result["answer"], "tool_calls": result.get("tool_calls", []),
                "tool_used": ", ".join(result.get("tool_calls", [])) or "gemini"}
    except Exception as e:
        traceback.print_exc()
        return {"answer": "Sorry, I encountered an error: " + str(e), "tool_calls": [], "tool_used": "error"}


# ── Specialty search ──────────────────────────────────────────────────────────
@app.post("/api/specialty-search")
async def specialty_search(req: SpecialtySearchRequest):
    try:
        from tools.gov_apis import (
            get_fips_from_zip, _fips_to_state, map_condition_to_specialty,
            search_providers_by_specialty, get_doctor_quality_score,
            check_doctor_in_plan_network
        )
        fips = get_fips_from_zip(req.zip_code)
        state = req.state or (_fips_to_state(fips) if fips else "TX")
        specialty_info = map_condition_to_specialty(req.condition)
        providers = search_providers_by_specialty(specialty_info["taxonomy_desc"], state, limit=5)
        enriched = []
        for p in providers[:5]:
            quality = get_doctor_quality_score(str(p.get("npi", ""))) if p.get("npi") else {}
            networks = {}
            if p.get("npi"):
                for pid in req.plan_ids[:3]:
                    networks[pid] = check_doctor_in_plan_network(pid, str(p["npi"]), req.zip_code)
            enriched.append({**p, "mips_score": quality.get("mips_score"),
                              "telehealth": quality.get("telehealth", False),
                              "network_status": networks})
        return {"condition": req.condition, "specialty": specialty_info.get("specialty", req.condition),
                "providers": enriched}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Procedure cost estimator ──────────────────────────────────────────────────
PROCEDURES = {
    "knee_replacement": ("Knee Replacement", 35000, "Surgery"),
    "hip_replacement": ("Hip Replacement", 32000, "Surgery"),
    "appendectomy": ("Appendectomy", 18000, "Surgery"),
    "back_surgery": ("Spinal Fusion", 45000, "Surgery"),
    "heart_bypass": ("Coronary Bypass", 95000, "Cardiac"),
    "colonoscopy": ("Colonoscopy", 3500, "Diagnostic"),
    "mri": ("MRI Scan", 2600, "Diagnostic"),
    "ct_scan": ("CT Scan", 2000, "Diagnostic"),
    "er_visit": ("ER Visit", 3200, "Emergency"),
    "ambulance": ("Ambulance", 1800, "Emergency"),
    "childbirth_vaginal": ("Vaginal Birth", 12000, "Maternity"),
    "childbirth_csection": ("C-Section", 20000, "Maternity"),
    "inpatient_3day": ("3-Day Hospital Stay", 22000, "Inpatient"),
}
COINSURANCE = {"Bronze": 0.40, "Silver": 0.30, "Gold": 0.20, "Platinum": 0.10}

@app.post("/api/procedure-cost")
async def procedure_cost(req: ProcedureCostRequest):
    proc = PROCEDURES.get(req.procedure_key)
    if not proc:
        raise HTTPException(status_code=400, detail="Unknown procedure.")
    name, total_cost, category = proc
    results = []
    for plan in req.plans[:5]:
        ded = plan.get("deductible", 0) or 0
        oop_max = plan.get("oop_max", 9450) or 9450
        metal = plan.get("metal_level", "Silver")
        coins = COINSURANCE.get(metal, 0.30)
        net_premium = plan.get("premium_w_credit") or plan.get("premium_after_subsidy") or plan.get("premium", 0)
        after_ded = max(0, total_cost - ded)
        patient_oop = min(ded + after_ded * coins, oop_max)
        results.append({
            "plan_name": plan.get("name", "Unknown"), "metal_level": metal,
            "net_premium": round(net_premium), "deductible": ded,
            "coinsurance_pct": round(coins * 100),
            "patient_oop": round(patient_oop),
            "insurance_pays": round(max(0, total_cost - patient_oop)),
        })
    results.sort(key=lambda x: x["patient_oop"])
    return {"procedure": name, "total_cost": total_cost, "category": category, "results": results}


# ── Hospital search ───────────────────────────────────────────────────────────
@app.get("/api/hospitals/nearby/{zip_code}")
async def hospitals_nearby(zip_code: str, plan_ids: str = ""):
    try:
        from tools.gov_apis import search_providers_by_specialty, get_fips_from_zip, _fips_to_state, check_doctor_in_plan_network
        fips = get_fips_from_zip(zip_code)
        state = _fips_to_state(fips) if fips else "TX"
        hospitals = search_providers_by_specialty("282N00000X", state, limit=5)
        pids = [p for p in plan_ids.split(",") if p][:3]
        enriched = []
        for h in hospitals[:5]:
            networks = {}
            if h.get("npi") and pids:
                for pid in pids:
                    networks[pid] = check_doctor_in_plan_network(pid, str(h["npi"]), zip_code)
            enriched.append({**h, "network_status": networks})
        return {"hospitals": enriched}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hospital-search")
async def hospital_search(req: HospitalSearchRequest):
    try:
        from tools.gov_apis import lookup_npi_registry, check_doctor_in_plan_network, get_fips_from_zip, _fips_to_state
        state = req.state
        if not state and req.zip_code:
            fips = get_fips_from_zip(req.zip_code)
            state = _fips_to_state(fips) if fips else ""
        result = lookup_npi_registry(req.name, state=state, entity_type="2")
        hospitals = result if isinstance(result, list) else ([result] if result.get("npi") else [])
        enriched = []
        for h in hospitals[:5]:
            networks = {}
            if h.get("npi") and req.plan_ids:
                for pid in req.plan_ids[:3]:
                    networks[pid] = check_doctor_in_plan_network(pid, str(h["npi"]), req.zip_code)
            enriched.append({**h, "network_status": networks})
        return {"hospitals": enriched}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Utility ───────────────────────────────────────────────────────────────────
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
    try:
        from rag.formulary_store import get_stats
        return get_stats()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/health")
async def health():
    return {"status": "ok", "adk_available": ADK_AVAILABLE, "qa_available": QA_AVAILABLE}

app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
