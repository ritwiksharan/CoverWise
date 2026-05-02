
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
Your goal is to provide a comprehensive, data-driven health insurance recommendation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPORT STRUCTURE — YOU MUST FOLLOW THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ## 🏆 Best Value Selection
   Explicitly pick the #1 plan based on **True Annual Cost**. 
   Show the math: (Net Premium * 12) + Est. OOP + Drug Costs = **$Total**.
   Explain exactly why this plan won (e.g. "Lowest out-of-pocket exposure for a chronic user").

2. ## 📊 Plan Comparison Table
   Create a Markdown table comparing the Top 3 plans across these metrics:
   | Plan | Net Premium | Deductible | True Annual Cost | Top Benefit |
   |------|-------------|------------|------------------|-------------|
   | ...  | ...         | ...        | ...              | ...         |

3. ## 💰 Financial Optimization
   Discuss the **APTC subsidy** and **CSR eligibility**. If the user is CSR-eligible, explain why Silver is superior to Gold/Bronze using specific dollar comparisons.
4. ## 💊 Medication & Doctor Notes
   Confirm coverage for each entered drug/doctor based on tool data.
5. ## ⚠️ Local Market Alerts
   Address HRSA shortages or SEP status.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Use Markdown tables for plan comparisons.
• Use bold text for all dollar amounts.
• Never invent data. If a tool returns "Unknown", say "Data not provided by insurer".
• Maintain a professional, expert advisor tone.
"""

async def run_full_analysis_parallel(tool_context: ToolContext) -> dict:
# ... (rest of function unchanged, just need to keep the code structure)
    """
    Run all domain-specific tools in parallel waves and return the consolidated data.
    """
    profile = tool_context.state.get("profile", {})
    zip_code = profile.get("zip_code")
    income = profile.get("income", 50000)
    age = profile.get("age", 35)
    household_size = profile.get("household_size", 1)
    tobacco_use = profile.get("tobacco_use", False)
    drugs = profile.get("drugs", [])
    doctors = profile.get("doctors", [])

    if not zip_code:
        return {"error": "ZIP code missing in profile state."}

    try:
        print(f"DEBUG: run_full_analysis_parallel called for ZIP {zip_code}")
        # Wave 0: Location
        loc = get_location_info(zip_code)
        state = loc["state"]
        fips = loc["fips"]

        # Wave 1: Subsidy & Plans
        subsidy, plans = await asyncio.gather(
            asyncio.to_thread(get_subsidy_estimate, income, age, household_size, zip_code, tobacco_use),
            asyncio.to_thread(find_plans, zip_code, age, income, tobacco_use)
        )
        plan_ids = [p["id"] for p in plans[:10]] if plans else []

        # Wave 2: Detailed Checks
        meds, docs, risks = await asyncio.gather(
            asyncio.to_thread(check_medication_coverage, drugs, plan_ids),
            asyncio.to_thread(verify_doctors, doctors, state, zip_code, plan_ids),
            asyncio.to_thread(get_market_risks, zip_code, state)
        )

        # Calculate True Annual Cost for UI
        monthly_credit = subsidy.get("monthly_aptc", 0)
        oop_weight = {"rarely": 0.1, "sometimes": 0.25, "frequently": 0.5, "chronic": 0.8}
        oop_factor = oop_weight.get(profile.get("utilization", "sometimes"), 0.25)
        
        # Build plan-drug coverage map for easier cost calculation
        drug_coverage_map = {}
        for c in meds.get("coverage_details", []):
            pid = c.get("plan_id")
            rxcui = str(c.get("rxcui", ""))
            if pid not in drug_coverage_map: drug_coverage_map[pid] = {}
            drug_coverage_map[pid][rxcui] = c

        for p in plans:
            pid = p.get("id")
            net_monthly = max(0, p.get("premium", 0) - monthly_credit)
            annual_premiums = net_monthly * 12
            
            # Estimate drug costs
            est_drugs = 0
            for d in meds.get("resolved_drugs", []):
                rxcui = str(d.get("rxcui", ""))
                c = drug_coverage_map.get(pid, {}).get(rxcui, {})
                if c.get("coverage") == "Covered":
                    tier = c.get("drug_tier", "Tier 1")
                    copay = 10 if "1" in tier else 30 if "2" in tier else 60
                    est_drugs += copay * 12
                elif c.get("coverage") == "NotCovered":
                    est_drugs += 500 * 12
            
            est_oop = p.get("deductible", 0) * oop_factor
            p["premium_w_credit"] = round(net_monthly, 2)
            p["true_annual_cost"] = round(annual_premiums + est_oop + est_drugs, 2)
            p["est_annual_drug_cost"] = round(est_drugs, 2)

        # Consolidate
        analysis_data = {
            "location": loc,
            "subsidy": subsidy,
            "plans": sorted(plans, key=lambda x: x.get("true_annual_cost", 999999)),
            "medication_coverage": meds,
            "doctor_verification": docs,
            "market_risks": risks,
            "cache_stats": get_cache_stats()
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
        
        # Create or Get session
        try:
            await self._session_service.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id,
                state={"profile": profile}
            )
        except Exception:
            # If session exists, update profile
            session = await self._session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
            session.state["profile"] = profile

        memory_context = build_memory_context(user_id) or ""
        prompt_text = (
            f"Mandatory: Call 'run_full_analysis_parallel' to retrieve the current insurance data. "
            f"After receiving the data, generate the recommendation report using the required structure. "
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
        
        # Get the consolidated data from state
        session = await self._session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        data = session.state.get("analysis_data", {})
        
        # Store in Mem0
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
        
        # Ensure session exists. Runner.run_async will handle it if create_session is called correctly.
        try:
            await self._session_service.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id,
                state={"profile": profile or {}}
            )
        except Exception:
            pass # Already exists

        memory_context = build_memory_context(user_id) or ""
        msg = Content(role="user", parts=[Part(text=message)])
        
        reply = ""
        async for event in self._runner.run_async(user_id=user_id, session_id=session_id, new_message=msg):
            if hasattr(event, "is_final_response") and event.is_final_response():
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            reply += part.text
        
        return {"reply": reply, "memory_used": bool(memory_context)}
