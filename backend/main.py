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

def _route_insurance_question(question: str) -> tuple[str, dict]:
    """Fallback keyword router used only when ADK is unavailable."""
    q = question.lower()
    if "compare" in q or " vs " in q or " versus " in q:
        # Try to extract tickers from the question
        import re
        # Common ticker patterns: 2-5 uppercase letters surrounded by spaces/punctuation
        tickers_found = re.findall(
            r"\b(ALL|PGR|TRV|CB|MET|HIG|CI|UNH|CVS|ELV|HUM|AFL|PRU|LNC|UNM|GL|RLI|CINF|"
            r"Allstate|Progressive|Travelers|Chubb|MetLife|Hartford|Cigna|UnitedHealth|"
            r"Aetna|Elevance|Humana|Aflac|Prudential|Lincoln|Unum|Torchmark|RLI|Cincinnati)\b",
            question,
            re.IGNORECASE,
        )
        # Map company names to tickers
        name_to_ticker = {
            "allstate": "ALL", "progressive": "PGR", "travelers": "TRV",
            "chubb": "CB", "metlife": "MET", "hartford": "HIG",
            "cigna": "CI", "unitedhealth": "UNH", "aetna": "CVS",
            "elevance": "ELV", "humana": "HUM", "aflac": "AFL",
            "prudential": "PRU", "lincoln": "LNC", "unum": "UNM",
            "torchmark": "GL", "cincinnati": "CINF",
        }
        resolved = []
        for t in tickers_found:
            upper = t.upper()
            mapped = name_to_ticker.get(t.lower(), upper)
            if mapped not in resolved:
                resolved.append(mapped)

        if len(resolved) < 2:
            # Fallback: pick two common insurers
            resolved = ["ALL", "PGR"]
        return "compare_insurers", {"tickers": resolved[:4]}

    # Financial / earnings queries
    if any(kw in q for kw in [
        "financial", "financials", "earnings", "revenue", "profit",
        "ratio", "ticker", "stock", "sec", "edgar", "10-k", "annual report",
        "net income", "assets", "liabilities",
    ]):
        # Extract a ticker from the question
        import re
        ticker_match = re.search(
            r"\b(ALL|PGR|TRV|CB|MET|HIG|CI|UNH|CVS|ELV|HUM|AFL|PRU|LNC|UNM|GL|RLI|CINF)\b",
            question,
            re.IGNORECASE,
        )
        # Also try company names
        name_to_ticker = {
            "allstate": "ALL", "progressive": "PGR", "travelers": "TRV",
            "chubb": "CB", "metlife": "MET", "hartford": "HIG",
            "cigna": "CI", "unitedhealth": "UNH", "aetna": "CVS",
            "elevance": "ELV", "humana": "HUM", "aflac": "AFL",
            "prudential": "PRU", "lincoln": "LNC", "unum": "UNM",
            "torchmark": "GL", "cincinnati": "CINF",
        }
        ticker = None
        if ticker_match:
            ticker = ticker_match.group(1).upper()
        else:
            for name, t in name_to_ticker.items():
                if name in q:
                    ticker = t
                    break
        ticker = ticker or "PGR"
        return "query_insurer_financials", {"ticker": ticker, "years": 3}

    # Risk / ZIP score queries
    if any(kw in q for kw in ["risk", "zip", "score", "zipcode", "flood risk"]):
        import re
        zip_match = re.search(r"\b(\d{5})\b", question)
        zip_code = zip_match.group(1) if zip_match else "77002"
        return "risk_score", {"zip_code": zip_code}

    # Flood / FEMA / disaster queries (default)
    state_abbrevs = [
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
        "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
        "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
        "TX","UT","VT","VA","WA","WV","WI","WY",
    ]
    import re
    state = ""
    for abbrev in state_abbrevs:
        if re.search(r"\b" + abbrev + r"\b", question):
            state = abbrev
            break
    # Also catch full state names — a small subset
    state_names = {
        "texas": "TX", "florida": "FL", "california": "CA",
        "louisiana": "LA", "new york": "NY", "north carolina": "NC",
        "south carolina": "SC", "georgia": "GA", "ohio": "OH",
    }
    if not state:
        for name, abbrev in state_names.items():
            if name in q:
                state = abbrev
                break
    state = state or "TX"
    # County extraction — naive: look for "in <County> county"
    county_match = re.search(r"in\s+(\w+)\s+county", question, re.IGNORECASE)
    county = county_match.group(1).upper() if county_match else ""
    return "query_flood_claims", {"state": state, "county": county, "limit": 100}


@app.post("/api/insurance-qa")
async def insurance_qa(req: InsuranceQARequest):
    """
    Answer free-form insurance questions using an ADK agent.
    Gemini 2.0 Flash reads the question, decides which tool(s) to call
    (flood claims, insurer financials, risk score, compare insurers),
    calls them with the right arguments, and synthesises the answer.
    """
    import json as _json

    try:
        from agents.insurance_qa_agent import get_agent, ADK_AVAILABLE

        if ADK_AVAILABLE:
            agent = get_agent()
            result = await agent.ask(req.user_id, req.question)

            if result.get("error") == "adk_unavailable":
                raise RuntimeError("ADK unavailable")

            return {
                "answer": result["answer"],
                "tool_calls": result.get("tool_calls", []),
                "data": {},
                "tool_used": ", ".join(result.get("tool_calls", [])) or "gemini",
            }

        # ── Fallback: keyword router + direct Gemini synthesis ────────────────
        tool_name, tool_kwargs = _route_insurance_question(req.question)
        from tools import insurance_mcp_tools as _tools
        fn = getattr(_tools, tool_name)
        tool_result = await asyncio.to_thread(fn, **tool_kwargs)

        answer_text = ""
        try:
            import google.genai as genai
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=(
                    "You are an insurance data analyst. Answer this question using ONLY "
                    "the provided data. Be specific, cite numbers, use markdown.\n\n"
                    f"Question: {req.question}\n\n"
                    f"Data:\n{_json.dumps(tool_result, indent=2)}"
                ),
            )
            answer_text = response.text
        except Exception:
            answer_text = f"**Data retrieved** (AI synthesis unavailable)\n\n```json\n{_json.dumps(tool_result, indent=2)}\n```"

        return {"answer": answer_text, "data": tool_result, "tool_used": tool_name, "tool_calls": [tool_name]}

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


@app.get("/api/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
