"""
Orchestrator Agent - coordinates all sub-agents and calls Gemini for final ranking.
"""

import os
import asyncio
import json
from typing import Optional
from google import genai
from google.genai import types as genai_types

from agents.sub_agents import (
    profile_agent, subsidy_agent, plan_search_agent,
    drug_check_agent, doctor_check_agent, risk_gaps_agent,
    metal_tier_agent, medicaid_agent
)
from memory.mem0_client import store_user_profile, build_memory_context
from cache.cache_manager import get_cache_stats

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "coverwise-local")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

def _make_client():
    if _USE_VERTEXAI:
        return genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

SYSTEM_PROMPT = """\
You are CoverWise, an expert ACA health insurance advisor with deep knowledge of the \
Affordable Care Act, premium tax credits (APTC), Cost Sharing Reductions (CSR), \
HSA-eligible plans, formulary tiers, and Medicare provider enrollment.

You receive REAL data pulled live from the CMS Marketplace API, the CMS Order & Referring \
dataset, and the CMS drugs/covered endpoint. Never invent numbers. Every figure you cite \
must come from the data provided to you.
"""

RECOMMENDATION_PROMPT_TEMPLATE = """\
{system_prompt}

{memory_context}

══════════════════════════════════════════════
SECTION 1 — USER PROFILE (from form input)
══════════════════════════════════════════════
ZIP: {zip_code}  |  Age: {age}  |  Annual Income: ${income:,}  |  Household Size: {household_size}  |  Healthcare Use: {utilization}
FPL: {fpl}%  |  Monthly APTC subsidy (CMS-calculated): ${monthly_credit:.2f}  |  Annual: ${annual_credit:,.0f}
CSR Eligible: {csr_eligible}  {csr_note}
Medications entered: {drugs_list}

══════════════════════════════════════════════
SECTION 2 — ALL AVAILABLE PLANS (real CMS Marketplace data)
══════════════════════════════════════════════
Each line = one real plan. Costs are pre-computed for you:
  • "net/mo" = full premium minus the ${monthly_credit:.2f} APTC subsidy
  • "est. true annual" = (net/mo × 12) + est_OOP + est_drugs
  • "worst-case annual" = (net/mo × 12) + OOP_max + est_drugs

{plans_block}

══════════════════════════════════════════════
SECTION 3 — DRUG COVERAGE (CMS drugs/covered API)
══════════════════════════════════════════════
{drug_block}
Drug warnings: {drug_warnings}

══════════════════════════════════════════════
SECTION 4 — DOCTOR VERIFICATION (CMS Order & Referring dataset)
══════════════════════════════════════════════
{doctor_block}

══════════════════════════════════════════════
SECTION 5 — RISK FLAGS & METAL TIER GUIDANCE
══════════════════════════════════════════════
Risk flags:
{flags}

Metal tier guidance: {metal_rec}

══════════════════════════════════════════════
YOUR TASK
══════════════════════════════════════════════
Produce a structured recommendation that follows EXACTLY this format and style:

---
## 📊 Full Plan Ranking — Lowest to Highest True Annual Cost

Rank every plan from Section 2. For each plan show the math explicitly. Example format:

**#1 — [Plan Name] ([Metal] [Type])**
Issuer: [Issuer Name]
Full premium: $XXX.XX/mo → After $YY subsidy: $ZZZ.ZZ/mo → Annual premiums: $A,AAA
Deductible: $D,DDD | OOP Max: $O,OOO
Est. annual drugs: $G,GGG
True annual (estimated): $A,AAA + $OOP + $G,GGG = $T,TTT  ← use real numbers
Worst-case annual (full OOP): $A,AAA + $O,OOO + $G,GGG = $W,WWW
HSA eligible: Yes/No
Why this rank: [1-2 sentence explanation based on this user's FPL, drugs, age]

**#2 — [Plan Name] ([Metal] [Type])**
... (same format)

(continue for ALL plans)

---
## 🏆 Top 3 Recommended Plans

Pick the 3 best for THIS user considering:
- Their FPL ({fpl}%) and subsidy amount (${monthly_credit:.2f}/mo)
- Whether they are CSR-eligible (Silver plans get enhanced deductibles at {fpl}% FPL)
- Their medications: {drugs_list}
- Their doctors: {doctor_names}

For each of the top 3, write 3-4 sentences explaining:
1. The exact net cost after subsidy
2. Why this metal tier makes sense for their income level
3. Any drug or doctor considerations specific to this plan
4. One tradeoff or risk to be aware of

---
## ⚠️ Flags & Alerts

Address each of these specifically with a 1-2 sentence explanation:
{flags}

If CSR eligible: Explain exactly what CSR means for Silver plans at {fpl}% FPL.
Example: "At 210% FPL, a Silver plan's deductible drops from $4,500 to ~$700 due to CSR — \\
this makes Silver act like Platinum at Silver prices. Always choose Silver if CSR-eligible."

If subsidy cliff warning: Explain the dollar impact.
Example: "Your income is $51,000 — just $1,000 below the 400% FPL cliff at $52,000. \\
If your income rises by $1,001, you lose your entire $95/mo subsidy ($1,140/year)."

If HSA-eligible plans exist: Explain the HSA math.
Example: "A Bronze HSA plan lets you contribute $4,150/year pre-tax (single, 2024). \\
At a 22% tax bracket that saves you $913/year in taxes, offsetting the higher deductible."

---
## 💊 Drug Coverage Notes

For each medication ({drugs_list}), state:
- What the CMS data shows (covered/not covered/data not provided)
- What action the user must take before enrolling
- If data is missing, instruct them to check the plan's formulary_url or call the insurer

---
## 🩺 Doctor Network Notes

For each doctor, state:
- Medicare enrollment status from Section 4
- Why this matters (Medicare Advantage vs. ACA marketplace network)
- What to verify before enrolling

---
## 📋 Action Checklist

Provide a numbered list of concrete steps the user must take, for example:
1. Call [top plan issuer] at their member services line and confirm [drug] is covered under your formulary before Feb 15 open enrollment deadline.
2. Confirm Dr. [name] is in-network for plan [name] — do not rely on the online directory alone.
3. [Any other specific action based on their data]
"""

CHAT_PROMPT_TEMPLATE = """\
{system_prompt}

You are in an ongoing conversation with a user who has already received a CoverWise plan \\
analysis. You have full context about their profile and plan results from memory.

{memory_context}

══════════════════════════════════════════════
HOW TO ANSWER FOLLOW-UP QUESTIONS
══════════════════════════════════════════════

Always ground your answers in specific numbers. Never give vague advice.

EXAMPLE of a GOOD answer:
  User: "Is Ozempic covered under the Blue plan?"
  Bad:  "You should check the formulary for drug coverage."
  Good: "The CMS drugs/covered API returned 'DataNotProvided' for Ozempic on the Blue \\
FocusCare Bronze plan. This means the insurer did not submit formulary data to CMS for 2024. \\
You must call Blue Cross IL directly at their member services line and ask specifically: \\
'Is semaglutide (Ozempic, RxCUI 1991311) covered under plan ID 36096IL0810183 for 2024, \\
and what is the copay?' Do this before enrolling."

EXAMPLE of a GOOD answer:
  User: "Should I pick Silver or Bronze?"
  Bad:  "It depends on your health needs."
  Good: "At your FPL of 210%, you are CSR-eligible. A Silver plan's deductible drops from \\
$4,500 to roughly $800 under CSR. Bronze plan net premium is $120/mo ($1,440/yr) but your \\
deductible is $7,000. Silver plan net premium is $145/mo ($1,740/yr) but your deductible \\
is only $800. If you spend more than $300 on healthcare in a year, Silver is cheaper overall: \\
$1,740 + $800 = $2,540 vs Bronze $1,440 + $7,000 = $8,440 in a bad year. Pick Silver."

EXAMPLE of a GOOD answer:
  User: "What if my income goes up?"
  Bad:  "Your subsidy may change."
  Good: "At $50,000 income you receive $87/mo APTC. The 400% FPL cliff for a household of 1 \\
in IL is $60,240. If your income crosses that, you lose all APTC — that's $1,044/year. \\
At $55,000 your APTC would drop to roughly $52/mo. You can update your income estimate on \\
healthcare.gov mid-year to avoid owing money at tax time."

══════════════════════════════════════════════
CONVERSATION HISTORY
══════════════════════════════════════════════
{history}"""


class OrchestratorAgent:
    def __init__(self):
        self.conversation_histories: dict = {}

    async def run(self, profile: dict) -> dict:
        user_id = profile.get("user_id", "anonymous")

        # Wave 0 — profile must run first (provides fips + state for all other agents)
        profile_result = await profile_agent(profile)
        route = profile_result["route"]
        fpl_pct = profile_result["fpl_percentage"]
        fips = profile_result["fips"]
        state = profile_result["state"]

        # HANDOFF: State-based exchange route
        state_exchange = profile_result.get("state_exchange")
        if state_exchange:
            return {
                "route": "state_exchange",
                "profile": profile_result,
                "state_exchange": state_exchange,
                "recommendation": state_exchange["message"],
                "plans": [],
                "cache_stats": get_cache_stats(),
            }

        if route == "medicaid":
            med = await medicaid_agent(profile, fpl_pct, state)
            store_user_profile(user_id, profile)
            return {
                "route": "medicaid",
                "profile": profile_result,
                "medicaid": med,
                "recommendation": med["message"],
                "plans": [],
                "cache_stats": get_cache_stats(),
            }

        # Wave 1 — parallel: subsidy + plan search (doctor check needs plan list, runs in wave 2)
        subsidy_result, plan_result = await asyncio.gather(
            subsidy_agent(profile, fpl_pct, fips, state),
            plan_search_agent(profile, fips, state),
        )
        plans = plan_result.get("plans", [])

        # Derive actual monthly credit from CMS plan data (premium - premium_w_credit)
        # This overrides the eligibility estimate with plan-specific CMS figures
        if plans:
            cms_credit = round(plans[0]["premium"] - plans[0]["premium_w_credit"], 2)
            if cms_credit > 0:
                subsidy_result["monthly_credit"] = cms_credit
                subsidy_result["annual_credit"] = round(cms_credit * 12, 2)

        monthly_credit = subsidy_result.get("monthly_credit", 0)

        # Enrich profile with state/fips for risk agent
        profile["state"] = state
        profile["fips"] = fips

        # Wave 2 — parallel: drug check + doctor check + risk & gaps + metal tier
        drug_result, doctor_result, risk_result, metal_result = await asyncio.gather(
            drug_check_agent(profile, plans),
            doctor_check_agent(profile, plans),
            risk_gaps_agent(profile, plans, fpl_pct),
            metal_tier_agent(profile, plans, subsidy_result),
        )

        memory_context = build_memory_context(user_id) or ""

        # Apply subsidy to every plan
        for plan in plans:
            plan["premium_after_subsidy"] = round(max(0, plan["premium"] - monthly_credit), 2)

        recommendation = await self._synthesize_recommendation(
            profile, profile_result, subsidy_result, plans,
            drug_result, doctor_result, risk_result, metal_result,
            memory_context
        )

        store_user_profile(user_id, profile)

        return {
            "route": route,
            "profile": profile_result,
            "subsidy": subsidy_result,
            "plans": plans,           # ALL plans — ranked by LLM
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
        utilization = profile.get("utilization", "sometimes")
        metal_rec = metal.get("recommendation") or ""
        flags = "\\n".join(risks.get("flags", []))
        drug_warnings = "\\n".join(drugs.get("warnings", [])) or "None"

        # Pre-compute true annual costs; weight OOP by utilization level
        oop_weight = {"rarely": 0.1, "sometimes": 0.25, "frequently": 0.5, "chronic": 0.8}
        oop_factor = oop_weight.get(utilization, 0.25)
        
        # Rough drug copay estimate by tier
        tier_copays = {"Tier 1": 10, "Tier 2": 30, "Tier 3": 60, "Tier 4": 150, "Tier 5": 300}

        for p in plans:
            pid = p.get("id")
            net_monthly = max(0, p.get("premium", 0) - monthly_credit)
            annual_premiums = round(net_monthly * 12, 2)
            deductible = p.get("deductible", 0)
            oop_max = p.get("oop_max", 0)
            
            # Estimate annual drug costs for this plan
            est_annual_drug_cost = 0
            for d in drugs.get("results", []):
                p_cov = d.get("plan_coverage", {}).get(pid, {})
                if p_cov.get("coverage") == "Covered":
                    tier = p_cov.get("tier")
                    copay = tier_copays.get(tier, 20) if tier else 20
                    est_annual_drug_cost += copay * 12
                elif p_cov.get("coverage") == "NotCovered":
                    est_annual_drug_cost += 500 * 12 # Penalise heavily if not covered
            
            est_oop = round(deductible * oop_factor)
            p["true_annual_cost"] = annual_premiums + est_oop + est_annual_drug_cost
            p["est_annual_drug_cost"] = est_annual_drug_cost
            p["est_oop"] = est_oop

        # Sort plans by true_annual_cost
        plans.sort(key=lambda x: x.get("true_annual_cost", 999999))

        plan_rows = []
        for p in plans:
            net_monthly = max(0, p.get("premium", 0) - monthly_credit)
            annual_premiums = round(net_monthly * 12, 2)
            hsa = " [HSA-eligible]" if p.get("hsa_eligible") else ""
            quality = f" ★{p['quality_rating']}" if p.get("quality_rating") else ""
            
            plan_rows.append(
                f"- [{p['metal_level']} {p['type']}] {p['name']} ({p.get('issuer','')}){quality}{hsa}: "
                f"full premium ${p.get('premium',0):.2f}/mo → net after subsidy ${net_monthly:.2f}/mo "
                f"(${annual_premiums:,.0f}/yr), deductible ${p['deductible']:,}, OOP max ${p['oop_max']:,} | "
                f"est. annual drugs ${p['est_annual_drug_cost']:,} | "
                f"est. true annual ({utilization} user) "
                f"${p['true_annual_cost']:,.0f} | worst-case ${annual_premiums + p['oop_max'] + p['est_annual_drug_cost']:,.0f}"
            )
        plans_block = "\\n".join(plan_rows) if plan_rows else "No plans found."

        # Drug coverage summary — include tier, PA, step therapy
        drug_coverage_lines = []
        for d in drugs.get("results", []):
            pa = f", prior_auth in {d.get('prior_auth_in',0)} plans" if d.get("prior_auth_in") else ""
            st = f", step_therapy in {d.get('step_therapy_in',0)} plans" if d.get("step_therapy_in") else ""
            generics = d.get("generic_alternatives", [])
            gen_note = ""
            if generics:
                gnames = ", ".join(g["generic_name"] for g in generics if g.get("is_generic"))
                if gnames:
                    gen_note = f" | generic alternative: {gnames}"
            drug_coverage_lines.append(
                f"- {d['drug_name']} (rxcui {d.get('rxcui','?')}): "
                f"covered in {d.get('covered_in',0)} plans, "
                f"not covered in {d.get('not_covered_in',0)}, "
                f"data missing in {d.get('data_missing_in',0)}{pa}{st}{gen_note}"
            )
        drug_block = "\\n".join(drug_coverage_lines) or "No medications entered."

        # Doctor verification summary — include NPI, specialty, MIPS, network status
        doctor_lines = []
        for d in doctors.get("results", []):
            if d.get("found"):
                mips = f", MIPS score {d['mips_score']}" if d.get("mips_score") else ""
                spec = f", {d['specialty']}" if d.get("specialty") else ""
                net_summary = ""
                for pid, net in d.get("network_status", {}).items():
                    if net.get("in_network") is False:
                        net_summary += f" | OUT-OF-NETWORK on {pid}"
                    elif net.get("in_network"):
                        acpt = " (not accepting patients)" if net.get("accepting_patients") is False else ""
                        net_summary += f" | in-network on {pid}{acpt}"
                doctor_lines.append(
                    f"- {d.get('name')} (NPI {d.get('npi')}){spec}{mips}{net_summary}"
                )
            else:
                doctor_lines.append(f"- {d.get('searched_name')}: not found in NPPES registry")
        doctor_block = "\\n".join(doctor_lines) or "No doctors entered."

        doctor_names = ", ".join(
            d.get("name", d.get("searched_name", "")) for d in doctors.get("results", [])
        ) or "None entered"

        prompt = RECOMMENDATION_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            memory_context=memory_context,
            zip_code=profile.get("zip_code", ""),
            age=profile.get("age", ""),
            income=profile.get("income", 0),
            household_size=profile.get("household_size", ""),
            utilization=utilization,
            fpl=fpl,
            monthly_credit=monthly_credit,
            annual_credit=annual_credit,
            csr_eligible=csr_eligible,
            csr_note=csr_note,
            drugs_list=drugs_list,
            plans_block=plans_block,
            drug_block=drug_block,
            drug_warnings=drug_warnings,
            doctor_block=doctor_block,
            flags=flags or "None identified.",
            metal_rec=metal_rec,
            doctor_names=doctor_names,
        )

        try:
            client = _make_client()
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text
        except Exception as e:
            return f"Recommendation error: {e}. Top plan by net premium: {plans[0]['name'] if plans else 'None'}"

    async def chat(self, user_id: str, message: str, profile=None) -> dict:
        memory_context = build_memory_context(user_id) or ""
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        self.conversation_histories[user_id].append(f"User: {message}")
        history_str = "\\n".join(self.conversation_histories[user_id][-10:])

        prompt = CHAT_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            memory_context=memory_context if memory_context else "(No prior session data — answer based on the conversation below.)",
            history=history_str,
        ) + "\\n\\nAssistant:"

        try:
            client = _make_client()
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            reply = response.text
            self.conversation_histories[user_id].append(f"Assistant: {reply}")
            return {"reply": reply, "memory_used": bool(memory_context)}
        except Exception as e:
            return {"reply": f"Error: {e}", "memory_used": False}
