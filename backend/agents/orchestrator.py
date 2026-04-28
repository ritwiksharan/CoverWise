"""
Orchestrator Agent - Vertex AI version
"""

import os
import asyncio
import json
from typing import Optional
import vertexai
from vertexai.generative_models import GenerativeModel

from agents.sub_agents import (
    profile_agent, subsidy_agent, plan_search_agent,
    drug_check_agent, doctor_check_agent, risk_gaps_agent,
    metal_tier_agent, medicaid_agent
)
from memory.mem0_client import store_user_profile, build_memory_context
from cache.cache_manager import get_cache_stats

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "agenticai-ritwik")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel("gemini-2.0-flash")

SYSTEM_PROMPT = (
    "You are CoverWise, an expert AI health insurance advisor. "
    "Help users find the best ACA plan for their situation. "
    "Be direct, use real numbers, flag subsidy cliffs and CSR opportunities."
)


class OrchestratorAgent:
    def __init__(self):
        self.conversation_histories: dict = {}

    async def run(self, profile: dict) -> dict:
        user_id = profile.get("user_id", "anonymous")
        profile_result = await profile_agent(profile)
        route = profile_result["route"]
        fpl_pct = profile_result["fpl_percentage"]

        if route == "medicaid":
            medicaid_result = await medicaid_agent(profile, fpl_pct)
            store_user_profile(user_id, profile)
            return {
                "route": "medicaid",
                "profile": profile_result,
                "medicaid": medicaid_result,
                "recommendation": medicaid_result["message"],
                "plans": [],
                "cache_stats": get_cache_stats(),
            }

        parallel_tasks = await asyncio.gather(
            subsidy_agent(profile, fpl_pct),
            plan_search_agent(profile),
            doctor_check_agent(profile),
        )
        subsidy_result, plan_result, doctor_result = parallel_tasks
        plans = plan_result.get("plans", [])

        parallel_tasks_2 = await asyncio.gather(
            drug_check_agent(profile, plans),
            risk_gaps_agent(profile, plans, fpl_pct),
            metal_tier_agent(profile, plans, subsidy_result),
        )
        drug_result, risk_result, metal_result = parallel_tasks_2

        memory_context = build_memory_context(user_id) or ""
        recommendation = await self._synthesize_recommendation(
            profile, profile_result, subsidy_result, plans,
            drug_result, doctor_result, risk_result, metal_result,
            memory_context
        )

        store_user_profile(user_id, profile)

        monthly_credit = subsidy_result.get("monthly_credit", 0)
        for plan in plans:
            plan["premium_after_subsidy"] = max(0, plan.get("premium", 0) - monthly_credit)

        return {
            "route": route,
            "profile": profile_result,
            "subsidy": subsidy_result,
            "plans": plans[:5],
            "drugs": drug_result,
            "doctors": doctor_result,
            "risks": risk_result,
            "metal_tier": metal_result,
            "recommendation": recommendation,
            "cache_stats": get_cache_stats(),
            "memory_used": bool(memory_context),
        }

    async def _synthesize_recommendation(
        self, profile, profile_result, subsidy, plans,
        drugs, doctors, risks, metal, memory_context
    ) -> str:
        monthly_credit = subsidy.get("monthly_credit", 0)
        annual_credit = subsidy.get("annual_credit", 0)
        csr_eligible = subsidy.get("csr_eligible", False)
        csr_note = subsidy.get("csr_note") or ""
        fpl = profile_result["fpl_percentage"]
        drugs_list = ", ".join(profile.get("drugs", [])) or "None"
        metal_rec = metal.get("recommendation") or ""
        flags = "\n".join(risks.get("flags", []))
        drug_warnings = ", ".join(drugs.get("warnings", [])) or "None"
        plans_json = json.dumps(plans[:5], indent=2)
        drugs_json = json.dumps(drugs.get("results", []), indent=2)
        memory_context = memory_context or ""

        parts = [
            SYSTEM_PROMPT,
            memory_context,
            "## User Profile",
            "ZIP: " + str(profile.get("zip_code", "")) + "  Age: " + str(profile.get("age", "")) + "  Income: $" + f"{profile.get('income', 0):,.0f}",
            "Household: " + str(profile.get("household_size", "")) + "  Meds: " + drugs_list,
            "FPL: " + str(fpl) + "%",
            "",
            "## Subsidy",
            "Monthly credit: $" + f"{monthly_credit:.2f}" + "  Annual: $" + f"{annual_credit:.2f}",
            "CSR eligible: " + str(csr_eligible) + "  " + csr_note,
            "",
            "## Available Plans",
            plans_json,
            "",
            "## Drug Coverage",
            drugs_json,
            "Warnings: " + drug_warnings,
            "",
            "## Risk Flags",
            flags,
            "",
            "## Metal Tier Guidance",
            metal_rec,
            "",
            "Give top 3 plan recommendations ranked by true annual cost.",
            "Include net premium after subsidy, tradeoffs, drug notes, warnings. Under 400 words.",
        ]
        prompt = "\n".join(str(p) for p in parts)

        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            fallback = plans[0]["name"] if plans else "No plans found"
            price = (plans[0].get("premium", 0) - monthly_credit) if plans else 0
            return "Top recommendation: " + fallback + " at $" + f"{price:.0f}" + "/month. Error: " + str(e)

    async def chat(self, user_id: str, message: str, profile=None) -> dict:
        memory_context = build_memory_context(user_id) or ""
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        self.conversation_histories[user_id].append("User: " + message)
        history_str = "\n".join(self.conversation_histories[user_id][-10:])
        parts = [SYSTEM_PROMPT, memory_context, "Conversation:", history_str, "Assistant:"]
        prompt = "\n".join(str(p) for p in parts)
        try:
            response = model.generate_content(prompt)
            reply = response.text
            self.conversation_histories[user_id].append("Assistant: " + reply)
            return {"reply": reply, "memory_used": bool(memory_context)}
        except Exception as e:
            return {"reply": "Error: " + str(e), "memory_used": False}
