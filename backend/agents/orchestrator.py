"""
CoverWise Conversational Orchestrator
Handles multi-turn chat intake, mem0 pre-fill, profile confirmation, and plan analysis
"""

import os
import json
import asyncio
from typing import Optional
import vertexai
from vertexai.generative_models import GenerativeModel

from agents.sub_agents import (
    subsidy_agent, plan_search_agent, drug_check_agent,
    doctor_check_agent, risk_gaps_agent, metal_tier_agent, medicaid_agent
)
from memory.mem0_client import store_user_profile, build_memory_context, get_user_memories
from cache.cache_manager import get_cache_stats

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "agenticai-ritwik")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel("gemini-2.0-flash")

INTAKE_SYSTEM = (
    "You are CoverWise, a friendly AI health insurance advisor. "
    "Collect the user profile through natural conversation. "
    "You need: zip_code, age, annual_income, household_size, medications (optional), doctors (optional), "
    "utilization (rarely/sometimes/frequently/chronic - how often they use healthcare), "
    "tobacco_use (yes/no), is_premium (yes/no - premium users get deeper HSA forecasts and 3x more detail). "
    "Ask one question at a time. Be warm and conversational. Keep responses to 1-2 sentences. "
    "Ask utilization as: How often do you typically use healthcare? (rarely/sometimes/frequently/chronic) "
    "Ask premium as: Would you like our premium deep analysis with HSA wealth forecasts and detailed benefit comparisons? (free/premium) "
    "Once you have all required fields output EXACTLY this on its own line with no extra text: "
    'PROFILE_COMPLETE:{"zip_code":"77001","age":34,"income":52000,"household_size":2,"drugs":[],"doctors":[],"utilization":"sometimes","tobacco_use":false,"is_premium":false}'
)

ADVISOR_SYSTEM = (
    "You are CoverWise, an expert ACA health insurance advisor. "
    "Answer follow-up questions using the plan data provided. "
    "Be specific with dollar amounts. Keep answers concise unless more detail is requested."
)


class ConversationalOrchestrator:
    def __init__(self):
        self.sessions = {}

    def _get_session(self, user_id):
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "state": "intake",
                "messages": [],
                "profile": None,
                "analysis": None,
            }
        return self.sessions[user_id]

    async def start(self, user_id):
        session = self._get_session(user_id)
        memories = get_user_memories(user_id)
        if memories:
            mem_text = ", ".join(memories[:5])
            prompt = (
                INTAKE_SYSTEM + "\n\nRETURNING USER. Known facts: " + mem_text +
                "\nGreet them, confirm what you know, ask only what may have changed. Start now."
            )
        else:
            prompt = INTAKE_SYSTEM + "\n\nNEW USER. Welcome them warmly and ask for ZIP code first."
        try:
            response = model.generate_content(prompt)
            reply = response.text
        except Exception:
            reply = "Welcome to CoverWise! To find your best health plan, what is your ZIP code?"
        session["messages"].append({"role": "assistant", "content": reply})
        return {"reply": reply, "state": "intake", "returning_user": bool(memories)}

    async def chat(self, user_id, message):
        session = self._get_session(user_id)
        state = session["state"]
        if state == "intake":
            return await self._handle_intake(user_id, session, message)
        elif state == "confirm":
            return await self._handle_confirmation(user_id, session, message)
        elif state == "complete":
            return await self._handle_followup(user_id, session, message)
        return {"reply": "Please refresh to start over.", "state": "error"}

    async def _handle_intake(self, user_id, session, message):
        session["messages"].append({"role": "user", "content": message})
        memories = get_user_memories(user_id)
        mem_ctx = "Known from memory: " + ", ".join(memories[:5]) if memories else ""

        # Build Gemini native chat history (alternating user/model turns)
        from vertexai.generative_models import ChatSession
        chat_history = []
        for m in session["messages"][:-1]:  # exclude current message
            role = "user" if m["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [{"text": m["content"]}]})

        system = (
            INTAKE_SYSTEM + "\n\n" + mem_ctx +
            "\n\nIMPORTANT: Reply ONLY with your next question or response. "
            "Do NOT repeat or summarize the conversation history. "
            "Do NOT prefix your reply with User: or Assistant:. "
            "Just respond naturally as if speaking directly to the user."
        )

        try:
            # Use fresh model call with history
            from vertexai.generative_models import GenerativeModel, Content, Part
            chat_model = GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=system
            )
            # Build history as Content objects
            history_contents = []
            for m in session["messages"][:-1]:
                role = "user" if m["role"] == "user" else "model"
                history_contents.append(Content(role=role, parts=[Part.from_text(m["content"])]))

            chat = chat_model.start_chat(history=history_contents)
            response = chat.send_message(message)
            reply = response.text
        except Exception as e:
            print("Chat error: " + str(e))
            # Fallback to simple prompt
            history = "\n".join(
                ("User: " if m["role"] == "user" else "Assistant: ") + m["content"]
                for m in session["messages"][-8:]
            )
            prompt = (
                INTAKE_SYSTEM + "\n\n" + mem_ctx +
                "\n\nDo NOT repeat conversation history. Just give your next response.\n\n" +
                "Context:\n" + history + "\n\nYour response:"
            )
            try:
                response = model.generate_content(prompt)
                reply = response.text
            except Exception:
                reply = "Could you share your ZIP code, age, annual income, and household size?"

        # Clean up any accidental history echoing
        if "User:" in reply and "Assistant:" in reply:
            # LLM echoed history - extract just the last assistant line
            lines = reply.strip().split("\n")
            for line in reversed(lines):
                if line.startswith("Assistant:"):
                    reply = line.replace("Assistant:", "").strip()
                    break
                elif line and not line.startswith("User:"):
                    reply = line.strip()
                    break

        session["messages"].append({"role": "assistant", "content": reply})
        if "PROFILE_COMPLETE:" in reply:
            profile = self._extract_profile(reply)
            if profile:
                session["profile"] = profile
                session["state"] = "confirm"
                clean = reply.split("PROFILE_COMPLETE:")[0].strip()
                confirm = self._build_confirm_msg(profile)
                return {"reply": clean + "\n\n" + confirm, "state": "confirm", "profile": profile}
        return {"reply": reply, "state": "intake"}

    def _extract_profile(self, text):
        try:
            idx = text.find("PROFILE_COMPLETE:")
            if idx == -1:
                return None
            json_str = text[idx + 17:].strip()
            end = json_str.find("\n")
            if end > 0:
                json_str = json_str[:end]
            profile = json.loads(json_str)
            if all(k in profile for k in ["zip_code", "age", "income", "household_size"]):
                profile.setdefault("drugs", [])
                profile.setdefault("doctors", [])
                return profile
        except Exception as e:
            print("Profile extract error: " + str(e))
        return None

    def _build_confirm_msg(self, p):
        drugs = ", ".join(p.get("drugs", [])) or "None"
        doctors = ", ".join(p.get("doctors", [])) or "None"
        income = float(p.get("income", 0))
        return (
            "Here is your profile - please confirm it looks correct:\n\n" +
            "ZIP Code: " + str(p.get("zip_code", "")) + "\n" +
            "Age: " + str(p.get("age", "")) + "\n" +
            "Annual Income: $" + f"{income:,.0f}" + "\n" +
            "Household Size: " + str(p.get("household_size", "")) + "\n" +
            "Medications: " + drugs + "\n" +
            "Doctors: " + doctors + "\n\n" +
            "Does this look correct? Say yes to analyze your plans, or tell me what to fix."
        )

    async def _handle_confirmation(self, user_id, session, message):
        msg_lower = message.lower().strip()
        positive = ["yes", "correct", "right", "looks good", "confirm", "yep", "yeah", "yup", "ok", "okay", "sure", "perfect", "exactly"]
        if any(w in msg_lower for w in positive):
            profile = session["profile"]
            profile["user_id"] = user_id
            analysis = await self._run_analysis(user_id, profile)
            session["analysis"] = analysis
            session["state"] = "complete"
            store_user_profile(user_id, profile)
            rec = analysis.get("recommendation", "")
            reply_text = "Profile confirmed! Here is your personalized plan analysis.\n\n" + rec if rec else "Profile confirmed! Here is your personalized plan analysis."
            return {"reply": reply_text, "state": "complete", "analysis": analysis}
        else:
            session["messages"].append({"role": "user", "content": message})
            session["state"] = "intake"
            return {"reply": "No problem - what would you like to correct?", "state": "intake"}

    async def _run_analysis(self, user_id, profile):
        from tools.gov_apis import get_state_exchange, calculate_fpl_percentage

        fpl_pct = calculate_fpl_percentage(float(profile["income"]), int(profile["household_size"]))
        state_exchange = get_state_exchange(str(profile["zip_code"]))
        if state_exchange:
            return {"route": "state_exchange", "state_exchange": state_exchange, "fpl_percentage": fpl_pct, "recommendation": state_exchange["message"], "plans": []}
        if fpl_pct < 138:
            med = await medicaid_agent(profile, fpl_pct)
            return {"route": "medicaid", "fpl_percentage": fpl_pct, "medicaid": med, "recommendation": med["message"], "plans": []}

        # Try ADK orchestrator first (available on Cloud Run)
        try:
            from agents.adk_orchestrator import ADKOrchestrator, ADK_AVAILABLE
            if ADK_AVAILABLE:
                adk = ADKOrchestrator()
                result = await adk.analyze(profile)
                result["fpl_percentage"] = fpl_pct
                return result
        except Exception as e:
            print("ADK analysis failed, falling back to basic: " + str(e))

        # Fallback: basic parallel sub-agents
        wave1 = await asyncio.gather(subsidy_agent(profile, fpl_pct), plan_search_agent(profile), doctor_check_agent(profile))
        subsidy, plan_result, doctors = wave1
        plans = plan_result.get("plans", [])
        wave2 = await asyncio.gather(drug_check_agent(profile, plans), risk_gaps_agent(profile, plans, fpl_pct), metal_tier_agent(profile, plans, subsidy))
        drugs, risks, metal = wave2
        monthly_credit = subsidy.get("monthly_credit", 0)
        for plan in plans:
            plan["premium_after_subsidy"] = max(0, plan.get("premium", 0) - monthly_credit)
        memory_context = build_memory_context(user_id) or ""
        recommendation = await self._synthesize(profile, fpl_pct, subsidy, plans, drugs, doctors, risks, metal, memory_context)
        return {
            "route": "subsidized" if fpl_pct <= 400 else "marketplace",
            "fpl_percentage": fpl_pct,
            "subsidy": subsidy,
            "plans": plans[:5],
            "drugs": drugs,
            "doctors": doctors,
            "risks": risks,
            "metal_tier": metal,
            "recommendation": recommendation,
            "cache_stats": get_cache_stats(),
        }

    async def _synthesize(self, profile, fpl_pct, subsidy, plans, drugs, doctors, risks, metal, memory_context):
        monthly_credit = subsidy.get("monthly_credit", 0)
        annual_credit = subsidy.get("annual_credit", 0)
        csr_note = subsidy.get("csr_note") or ""
        drugs_list = ", ".join(profile.get("drugs", [])) or "None"
        flags = "\n".join(risks.get("flags", []))
        drug_warnings = ", ".join(drugs.get("warnings", [])) or "None"
        plans_json = json.dumps(plans[:5], indent=2)
        drugs_json = json.dumps(drugs.get("results", []), indent=2)
        income = float(profile.get("income", 0))
        parts = [
            "You are CoverWise, expert ACA advisor. Give specific personalized recommendations.",
            memory_context,
            "User: ZIP " + str(profile.get("zip_code")) + " Age " + str(profile.get("age")) + " Income $" + f"{income:,.0f}" + " Household " + str(profile.get("household_size")),
            "FPL: " + str(fpl_pct) + "% Monthly subsidy: $" + f"{monthly_credit:.2f}" + " Annual: $" + f"{annual_credit:.2f}",
            "CSR: " + csr_note,
            "Meds: " + drugs_list,
            "Plans:\n" + plans_json,
            "Drug coverage:\n" + drugs_json,
            "Warnings: " + drug_warnings,
            "Risk flags:\n" + flags,
            "Metal rec: " + (metal.get("recommendation") or ""),
            "Give top 3 plans ranked by TRUE annual cost. For each: net premium after subsidy, why it ranks here, drug note, one warning." +
            (" Include a 5-Year HSA Wealth Forecast table for any HSA-eligible plans. Provide breakeven analysis between top plans. Explain CSR variants in detail. Give 3x more detail on financial tradeoffs. No word limit." if profile.get("is_premium") else " Under 400 words."),
        ]
        prompt = "\n".join(str(p) for p in parts)
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return "Top plan: " + (plans[0]["name"] if plans else "N/A") + ". Error: " + str(e)

    async def _handle_followup(self, user_id, session, message):
        analysis = session.get("analysis", {})
        plans_summary = json.dumps(analysis.get("plans", [])[:3], indent=2)
        memory_context = build_memory_context(user_id) or ""
        prompt = (
            ADVISOR_SYSTEM + "\n\n" + memory_context + "\n\n" +
            "Plans:\n" + plans_summary + "\n\nRecommendation:\n" + str(analysis.get("recommendation", "")) +
            "\n\nUser: " + message + "\n\nAnswer:"
        )
        try:
            response = model.generate_content(prompt)
            return {"reply": response.text, "state": "complete"}
        except Exception as e:
            return {"reply": "Error: " + str(e), "state": "complete"}

    def reset(self, user_id):
        if user_id in self.sessions:
            del self.sessions[user_id]
        return {"reply": "Starting fresh! What is your ZIP code?", "state": "intake"}
