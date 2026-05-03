
"""
ADK Orchestrator - Google ADK powered expert analysis.
"""
import os
import asyncio
import traceback
from typing import List, Dict, Any, Optional

try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import ToolContext
    from google.genai.types import Content, Part
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    ToolContext = None
    Content = None
    Part = None

from agents.tools import (
    get_location_info, get_subsidy_estimate, find_plans,
    check_medication_coverage, verify_doctors, get_market_risks
)
from memory.mem0_client import build_memory_context, store_user_profile
from cache.cache_manager import get_cache_stats

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "coverwise-local")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
APP_NAME = "CoverWise"

ORCHESTRATOR_INSTRUCTION = """You are the CoverWise Expert Analysis Agent.
Your goal is to provide a high-fidelity, multi-pillar insurance analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS PILLARS — COMPARATIVE DEPTH REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ## 🛡️ Financial Pillar (Risk vs Reward)
   • Compare the **Fixed Costs** (Annual Premiums) vs **Variable Risks** (Deductible/OOP Max).
   • Calculate the "Breakeven Point": How many doctor visits until Plan A becomes cheaper than Plan B?
   • Explain the **Actuarial Value** (Bronze=60%, Silver=70%, Gold=80%) and what it means for their specific income.

2. ## 💊 Medical Pillar (Benefit Depth)
   • Don't just list drugs. Analyze the **Formulary Strategy**. 
   • If a drug is Tier 3, explain the "Step Therapy" or "Prior Auth" requirements found in the tool data.
   • Suggest specific **Generic Savings** (e.g., "Switching to the generic version of Drug X saves you $120/mo").

3. ## 🏥 Network Pillar (Provider Access)
   • Explicitly confirm if **every** doctor provided is In-Network.
   • For Out-of-Network doctors, calculate the estimated "Balance Billing" risk.
   • Mention the **MIPS Quality Score** for each doctor (e.g., "Dr. Smith has a 95/100 quality rating").

4. ## 🌐 Market Pillar (Local Context)
   • Analyze **HRSA Shortage Scores**. If primary care is scarce in their ZIP, emphasize plans with **$0 Telehealth**.
   • Confirm SEP status with the exact deadline date.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREMIUM TIER RULES (is_premium: true)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Provide **3x more detail** in the comparisons.
• Include a **5-Year HSA Wealth Forecast** table for HDHP plans.
• Provide a **Side-by-Side Benefit Table** for the Top 3 plans.
• Explain **CSR variants (73/87/94)** in detail—don't just say they qualify; show the deductible drop.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURE & FORMATTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ALWAYS use Markdown tables for comparisons. Start lines with `|`.
• Use `**` for all currency amounts.
• Tone: Objective, mathematical, and authoritative.
"""

async def run_full_analysis_parallel(tool_context) -> dict:
    """
    Run all domain-specific tools in parallel waves and return the consolidated data.
    """
    profile = tool_context.state.get("profile", {})
    zip_code = profile.get("zip_code")
    income = profile.get("income", 50000)
    age = profile.get("age", 35)
    household_size = profile.get("household_size", 1)
    tobacco_use = profile.get("tobacco_use", False)
    is_premium = profile.get("is_premium", False)
    drugs = profile.get("drugs", [])
    doctors = profile.get("doctors", [])

    # Enforce Limits
    if not is_premium:
        drugs = drugs[:1]
        doctors = doctors[:1]

    if not zip_code:
        return {"error": "ZIP code missing in profile state."}

    try:
        # Wave 0: Location
        loc = get_location_info(zip_code)
        state = loc["state"]
        fips = loc["fips"]

        # Wave 1: Subsidy & Plans
        subsidy, plans = await asyncio.gather(
            asyncio.to_thread(get_subsidy_estimate, income, age, household_size, zip_code, tobacco_use),
            asyncio.to_thread(find_plans, zip_code, age, income, tobacco_use)
        )
        # Premium shows 10, Free shows 3
        plan_limit = 10 if is_premium else 3
        plan_ids = [p["id"] for p in plans[:plan_limit]] if plans else []

        # Wave 2: Detailed Checks
        meds, docs, risks = await asyncio.gather(
            asyncio.to_thread(check_medication_coverage, drugs, plan_ids),
            asyncio.to_thread(verify_doctors, doctors, state, zip_code, plan_ids),
            asyncio.to_thread(get_market_risks, zip_code, state)
        )

        # Calculate True Annual Cost & Premium Features
        monthly_credit = subsidy.get("monthly_aptc", 0)
        oop_weight = {"rarely": 0.1, "sometimes": 0.25, "frequently": 0.5, "chronic": 0.8}
        oop_factor = oop_weight.get(profile.get("utilization", "sometimes"), 0.25)
        
        # Build plan-drug coverage map
        drug_coverage_map = {}
        for c in meds.get("coverage_details", []):
            pid = c.get("plan_id")
            rxcui = str(c.get("rxcui", ""))
            if pid not in drug_coverage_map: drug_coverage_map[pid] = {}
            drug_coverage_map[pid][rxcui] = c

        processed_plans = []
        for p in plans[:plan_limit]:
            pid = p.get("id")
            net_monthly = max(0, p.get("premium", 0) - monthly_credit)
            annual_premiums = net_monthly * 12
            
            # Estimate drug costs
            est_drugs = 0
            pa_required = False
            for d in meds.get("resolved_drugs", []):
                rxcui = str(d.get("rxcui", ""))
                c = drug_coverage_map.get(pid, {}).get(rxcui, {})
                if c.get("coverage") == "Covered":
                    tier = c.get("drug_tier", "Tier 1")
                    copay = 10 if "1" in tier else 30 if "2" in tier else 60
                    est_drugs += copay * 12
                    if c.get("prior_authorization"): pa_required = True
                elif c.get("coverage") == "NotCovered":
                    est_drugs += 500 * 12
            
            est_oop = p.get("deductible", 0) * oop_factor
            p["premium_w_credit"] = round(net_monthly, 2)
            p["true_annual_cost"] = round(annual_premiums + est_oop + est_drugs, 2)
            p["est_annual_drug_cost"] = round(est_drugs, 2)
            p["pa_warning"] = pa_required

            # HSA Tax Savings (Premium)
            if is_premium and p.get("hsa_eligible"):
                tax_rate = 0.22 if income > 45000 else 0.12
                annual_tax_save = 4150 * tax_rate # Individual limit
                p["hsa_tax_savings"] = round(annual_tax_save, 2)
                p["hsa_5yr_growth"] = round(4150 * 5 * 1.07, 2) # Simple 7% growth

            processed_plans.append(p)

        # Consolidate
        analysis_data = {
            "location": loc,
            "subsidy": subsidy,
            "plans": sorted(processed_plans, key=lambda x: x.get("true_annual_cost", 999999)),
            "medication_coverage": meds,
            "doctor_verification": docs,
            "market_risks": risks,
            "cache_stats": get_cache_stats(),
            "is_premium": is_premium
        }
        
        # Update state directly
        tool_context.state["analysis_data"] = analysis_data
        return {"status": "Analysis complete", "plan_count": len(plans), "data": analysis_data}
    except Exception as e:
        traceback.print_exc()
        return {"status": "Error", "message": str(e)}

class ADKOrchestrator:
    def __init__(self):
        self._runner = None
        if ADK_AVAILABLE:
            self._session_service = InMemorySessionService()
        else:
            self._session_service = None

    def _ensure_runner(self):
        if self._runner: return
        agent = Agent(
            name="insurance_expert",
            model="gemini-2.0-flash",
            instruction=ORCHESTRATOR_INSTRUCTION,
            tools=[run_full_analysis_parallel]
        )
        self._runner = Runner(agent=agent, app_name=APP_NAME, session_service=self._session_service)

    async def analyze(self, profile: dict) -> dict:
        self._ensure_runner()
        user_id = profile.get("user_id", "anonymous")
        session_id = user_id
        
        try:
            await self._session_service.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id,
                state={"profile": profile}
            )
        except Exception:
            session = await self._session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
            session.state["profile"] = profile

        memory_context = build_memory_context(user_id) or ""
        prompt_text = (
            f"Perform a multi-pillar insurance analysis. "
            f"User Profile: {profile}. Context: {memory_context}"
        )
        msg = Content(role="user", parts=[Part(text=prompt_text)])
        
        reply = ""
        async for event in self._runner.run_async(user_id=user_id, session_id=session_id, new_message=msg):
            if hasattr(event, "is_final_response") and event.is_final_response():
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            reply += part.text
        
        session = await self._session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        data = session.state.get("analysis_data", {})
        store_user_profile(user_id, profile)
        
        return {
            "route": data.get("subsidy", {}).get("is_medicaid_eligible") and "medicaid" or "subsidized",
            "profile": {
                "fpl_percentage": data.get("subsidy", {}).get("fpl_percentage"),
                "route_reason": "Based on income and location analysis."
            },
            "recommendation": reply,
            "plans": data.get("plans", []),
            "subsidy": data.get("subsidy", {}),
            "drugs": data.get("medication_coverage", {}),
            "doctors": data.get("doctor_verification", {}),
            "risks": data.get("market_risks", {}),
            "cache_stats": data.get("cache_stats", {}),
            "memory_used": bool(memory_context)
        }

    async def chat(self, user_id: str, message: str, profile: Optional[dict] = None) -> dict:
        self._ensure_runner()
        session_id = user_id
        
        try:
            await self._session_service.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id,
                state={"profile": profile or {}}
            )
        except Exception:
            pass

        # Retrieve context from previous analysis
        session = await self._session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        analysis_data = session.state.get("analysis_data", {})
        user_profile = session.state.get("profile", profile or {})

        context_msg = (
            f"SYSTEM CONTEXT: You are helping a user with their insurance choices. "
            f"User Profile: {user_profile}. "
            f"Analysis Data Summary: {len(analysis_data.get('plans', []))} plans found. "
            f"Subsidy: ${analysis_data.get('subsidy', {}).get('monthly_aptc', 0)}/mo. "
            f"Drugs checked: {len(analysis_data.get('medication_coverage', {}).get('resolved_drugs', []))}. "
            f"Market Risks: {analysis_data.get('market_risks', {})}. "
            f"Please answer based on this specific data."
        )

        msg = Content(role="user", parts=[Part(text=f"{context_msg}\n\nUSER MESSAGE: {message}")])
        
        reply = ""
        async for event in self._runner.run_async(user_id=user_id, session_id=session_id, new_message=msg):
            if hasattr(event, "is_final_response") and event.is_final_response():
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            reply += part.text
        
        return {"reply": reply, "memory_used": False}
