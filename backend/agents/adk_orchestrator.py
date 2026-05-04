
"""
ADK Orchestrator — two-phase pipeline:
  Phase 1: parallel data collection (plans, drugs, doctors, risks)
  Phase 2: Gemini 2.0 Flash synthesis with ORCHESTRATOR_INSTRUCTION as system prompt

The synthesis call receives a fully structured text document (not raw JSON) so that
Gemini can apply the 4-pillar analysis, scenario ranking, and Markdown tables correctly.
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
except Exception as _import_err:
    print(f"[ADK import error] {_import_err}")
    ADK_AVAILABLE = False
    class ToolContext:  # type: ignore[no-redef]
        state: dict = {}
    class Content:  # type: ignore[no-redef]
        def __init__(self, **kw): pass
    class Part:  # type: ignore[no-redef]
        @staticmethod
        def from_text(t): return t
        def __init__(self, **kw): pass

try:
    from google import genai
    from google.genai import types as genai_types
    VERTEXAI_AVAILABLE = True
except Exception as _import_err:
    print(f"[ADK import error] {_import_err}")
    VERTEXAI_AVAILABLE = False

USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

from agents.tools import (
    get_location_info, get_subsidy_estimate, find_plans,
    check_medication_coverage, verify_doctors, get_market_risks
)
from tools.gov_apis import get_plan_drug_copays
from memory.mem0_client import build_memory_context, store_user_profile
from cache.cache_manager import get_cache_stats

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "coverwise-local")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
APP_NAME = "CoverWise"

ORCHESTRATOR_INSTRUCTION = """You are the CoverWise Expert Analysis Agent.
Produce a rigorous, mathematically-grounded health insurance recommendation. Explain every concept
(deductibles, OOP max, CSR, actuarial value, HSA) clearly for a layperson while staying numerically precise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STRUCTURE — these sections must appear in this exact order
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Pre-Analysis
Answer the four reasoning questions (Q1–Q4) from the data. Each answer must cite real numbers and
explain the implication for this specific user. Keep each answer concise but substantive.

## Recommendation
Start with a one-sentence summary of the top pick and why it wins for this user's situation.

Then produce the following sub-sections IN ORDER — every one is required:

### EV Ranking
A table ranking all plans by Expected Value score (lowest EV = best). Mark the top pick clearly.
For each plan, explain in a few sentences WHY it ranks where it does — not just what the number is,
but what trade-off it represents relative to the plan above and below it.

### 🛡️ Financial Pillar
Cover these four topics in order:

1. Scenario Costs — a table showing each plan's total cost across the three simulated years
   (Healthy Year = premiums only, Clinical Year = premiums + drugs, Worst Case = premiums + full OOP Max).
   After the table, identify which scenario is most relevant for this user and explain why.

2. Breakeven Analysis — for the top two plan pairs (rank 1 vs 2, rank 1 vs 3), show the arithmetic:
   how much the cheaper plan saves per year in premiums, how much lower the other plan's OOP Max is,
   and how many extra catastrophic events or specialist visits it would take to justify the premium difference.
   Conclude whether that breakeven is realistic given the user's utilization level.

3. Actuarial Value — explain what the metal level's AV percentage means in practical dollar terms
   at this user's income: how much they pay before coverage kicks in, and what percentage of costs
   they absorb themselves.

4. HSA Wealth Forecast — if any plan in the list is HSA-eligible, show a 5-year table of annual
   contributions, running balance, and cumulative federal tax savings. Explain the triple tax advantage
   (deductible contributions, tax-free growth, tax-free qualified withdrawals) briefly.

Side-by-Side Benefit Comparison — a table comparing the top plans across premium, deductible,
OOP Max, and HSA eligibility. Follow with a short plain-language explanation of the most important differences.

### 💊 Medical Pillar
If medications were provided: cover each drug's coverage status, tier, Prior Authorization implications
(what it means in practice: paperwork burden, delay, denial risk, cash cost), Step Therapy requirements,
and any generic alternatives with their savings.
If no medications were provided: state that this pillar is skipped and why.

### 🏥 Network Pillar
If doctors were provided: confirm each doctor's network status by name and NPI, interpret their MIPS
quality score in plain English, and estimate out-of-network cost exposure if applicable.
If no doctors were provided: state that this pillar is skipped and why.

### 🌐 Market Pillar
State the enrollment status, exact deadline, and days remaining. If enrollment is closed,
explain what qualifying life events trigger a Special Enrollment Period and the 60-day window rule.

### Recommendation Summary
A short paragraph (3–5 sentences) that names the top pick, states the single most important trade-off
the user must weigh, and gives one concrete next action they should take.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RANKING RULE (NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the data contains "FINAL PLAN ORDER (NON-NEGOTIABLE)", follow it exactly in every table and paragraph.
Do NOT re-rank or reorder. Your role is explanation only.
When no pre-computed ranking is provided, compute EV using the utilization-adjusted weights from the data.
CSR override: if CSR-eligible, the top Silver plan must be rank 1 regardless of EV arithmetic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Use Markdown tables for all comparisons — never render tabular data as prose.
• Do NOT use asterisk (*) characters anywhere in the output.
• Do NOT invent numbers — every dollar figure must come verbatim from the data provided.
• Tone: authoritative and precise, but accessible — define every insurance term the first time you use it.
"""

# ── Drug cost helpers ────────────────────────────────────────────────────────
# Typical monthly list price per tier — used when plan charges coinsurance, not a flat copay.
_TIER_LIST_PRICE = {
    "GENERIC": 50,          "PREFERRED-GENERIC": 40,     "NON-PREFERRED-GENERIC": 80,
    "PREFERRED-BRAND": 300, "BRAND": 300,                "NON-PREFERRED-BRAND": 500,
    "SPECIALTY": 3000,      "PREFERRED-SPECIALTY": 2500, "NON-PREFERRED-SPECIALTY": 4000,
}

# Last-resort fallback when CMS benefits API returns nothing for a plan
_TIER_COPAY_FALLBACK = {
    "PREFERRED-GENERIC": 10, "GENERIC": 10,     "NON-PREFERRED-GENERIC": 20,
    "PREFERRED-BRAND": 45,   "BRAND": 45,       "NON-PREFERRED-BRAND": 75,
    "SPECIALTY": 100,        "PREFERRED-SPECIALTY": 100, "NON-PREFERRED-SPECIALTY": 150,
}


def _calc_drug_monthly_cost(tier: str, plan_copay_data: dict):
    """Return (monthly_cost_float, display_str) using real CMS cost-sharing or fallback."""
    info = plan_copay_data.get(tier, {})
    if info:
        copay = float(info.get("copay_amount") or 0)
        coins = float(info.get("coinsurance_rate") or 0)
        display = info.get("display_string", "")
        if copay > 0:
            return copay, display or f"${copay:.0f}/mo copay"
        if coins > 0:
            monthly = round(_TIER_LIST_PRICE.get(tier, 200) * coins, 2)
            return monthly, display or f"{coins*100:.0f}% coinsurance (~${monthly:.0f}/mo)"
        return 0.0, display or "$0 (covered)"
    cost = float(_TIER_COPAY_FALLBACK.get(tier, 50))
    return cost, f"~${cost:.0f}/mo (estimated)"


# ── PHASE 1: Data Collection ──────────────────────────────────────────────────

async def _collect_analysis_data(profile: dict) -> dict:
    """
    Parallel data collection — three waves, no AI involved.
    Returns structured dict consumed by _build_synthesis_prompt.
    """
    zip_code  = profile.get("zip_code")
    income    = profile.get("income", 50000)
    age       = profile.get("age", 35)
    household_size = profile.get("household_size", 1)
    tobacco_use    = profile.get("tobacco_use", False)
    is_premium     = profile.get("is_premium", False)
    drugs   = profile.get("drugs", [])
    doctors = profile.get("doctors", [])

    if not is_premium:
        drugs   = drugs[:1]
        doctors = doctors[:1]

    if not zip_code:
        raise ValueError("ZIP code missing in profile")

    # Wave 0 — location
    loc   = get_location_info(zip_code)
    state = loc["state"]

    # Wave 1 — subsidy + plans (parallel)
    subsidy, plans = await asyncio.gather(
        asyncio.to_thread(get_subsidy_estimate, income, age, household_size, zip_code, tobacco_use),
        asyncio.to_thread(find_plans, zip_code, age, income, tobacco_use),
    )
    plan_limit = 10 if is_premium else 3
    plan_ids   = [p["id"] for p in plans[:plan_limit]]

    # Wave 2 — drugs / doctors / risks (parallel, skip empty inputs)
    _empty_meds = {"resolved_drugs": [], "coverage_details": [], "generic_suggestions": {}}
    _empty_docs = {"results": []}

    wave2_tasks = [asyncio.to_thread(get_market_risks, zip_code, state)]
    if drugs:
        wave2_tasks.insert(0, asyncio.to_thread(check_medication_coverage, drugs, plan_ids))
    if doctors:
        wave2_tasks.insert(1 if drugs else 0, asyncio.to_thread(verify_doctors, doctors, state, zip_code, plan_ids, plans[:plan_limit]))

    wave2_results = await asyncio.gather(*wave2_tasks)

    idx = 0
    meds  = wave2_results[idx] if drugs else _empty_meds;  idx += bool(drugs)
    docs  = wave2_results[idx] if doctors else _empty_docs; idx += bool(doctors)
    risks = wave2_results[idx]

    # ── Enrich plans with financial model ────────────────────────────────────
    monthly_credit = subsidy.get("monthly_aptc", 0)
    oop_factor = {"rarely": 0.1, "sometimes": 0.25, "frequently": 0.5, "chronic": 0.8}.get(
        profile.get("utilization", "sometimes"), 0.25
    )

    # rxcui → coverage per plan
    drug_coverage_map: Dict[str, Dict[str, dict]] = {}
    for c in meds.get("coverage_details", []):
        pid   = c.get("plan_id")
        rxcui = str(c.get("rxcui", ""))
        drug_coverage_map.setdefault(pid, {})[rxcui] = c

    processed_plans = []
    for p in plans[:plan_limit]:
        pid = p.get("id")
        net_monthly = max(0, p.get("premium", 0) - monthly_credit)

        # Real drug cost-sharing from CMS plan benefits endpoint
        plan_copay_data = get_plan_drug_copays(pid)

        est_drugs   = 0
        pa_required = False
        drug_detail: List[dict] = []

        for d in meds.get("resolved_drugs", []):
            rxcui = str(d.get("rxcui", ""))
            cov   = drug_coverage_map.get(pid, {}).get(rxcui, {})
            tier  = (cov.get("drug_tier") or "").upper()

            if cov.get("coverage") == "Covered":
                copay, copay_display = _calc_drug_monthly_cost(tier, plan_copay_data)
                est_drugs += copay * 12
                if cov.get("prior_authorization"):
                    pa_required = True
            elif cov.get("coverage") == "NotCovered":
                copay, copay_display = 500.0, "$500/mo (OOP — not covered)"
                est_drugs += 500 * 12
            else:
                copay, copay_display = 0.0, "—"

            drug_detail.append({
                "name":          d.get("name", ""),
                "rxcui":         rxcui,
                "coverage":      cov.get("coverage", "DataNotProvided"),
                "tier":          tier or "—",
                "pa":            bool(cov.get("prior_authorization")),
                "st":            bool(cov.get("step_therapy")),
                "ql":            bool(cov.get("quantity_limit")),
                "copay_mo":      copay,
                "copay_display": copay_display,
                "source":        cov.get("source", "cms"),
                "note":          cov.get("note", ""),
            })

        est_oop = p.get("deductible", 0) * oop_factor
        annual_premium = net_monthly * 12

        # Scenario costs
        healthy_year   = round(annual_premium, 2)
        clinical_year  = round(annual_premium + est_drugs, 2)
        worst_case     = round(annual_premium + p.get("oop_max", 0), 2)
        true_annual    = round(annual_premium + est_oop + est_drugs, 2)

        p.update({
            "premium_w_credit":   round(net_monthly, 2),
            "true_annual_cost":   true_annual,
            "est_annual_drug_cost": round(est_drugs, 2),
            "pa_warning":         pa_required,
            "drug_detail":        drug_detail,
            "scenario_healthy":   healthy_year,
            "scenario_clinical":  clinical_year,
            "scenario_worst":     worst_case,
        })

        if is_premium and p.get("hsa_eligible"):
            hsa_limit  = 8300 if household_size > 1 else 4150
            tax_rate   = 0.22 if income > 45000 else 0.12
            p["hsa_tax_savings"] = round(hsa_limit * tax_rate, 2)
            p["hsa_5yr_growth"]  = round(hsa_limit * 5 * 1.07, 2)

        processed_plans.append(p)

    # ── Risk flags ───────────────────────────────────────────────────────────
    fpl_pct      = subsidy.get("fpl_percentage", 0)
    monthly_aptc = subsidy.get("monthly_aptc", 0)
    risk_flags   = []

    if any(p.get("oop_max", 0) > 8700 for p in processed_plans):
        risk_flags.append("⚠️ Some plans have an OOP max above $8,700 — weigh catastrophic risk carefully.")
    if 390 <= fpl_pct <= 410:
        risk_flags.append(f"🚨 Subsidy cliff: income within 5% of 400% FPL. ~$1,800 more income = lose ${monthly_aptc*12:,.0f}/yr subsidy.")

    hsa_plans_list = [p for p in processed_plans if p.get("hsa_eligible")]
    if hsa_plans_list:
        hsa_limit  = 8300 if household_size > 1 else 4150
        tax_rate   = 0.22 if income > 45000 else 0.12
        risk_flags.append(f"💡 {len(hsa_plans_list)} HSA-eligible plan(s). Max contribution ${hsa_limit:,}/yr → ~${round(hsa_limit*tax_rate):,} federal tax savings.")

    if age < 30:
        risk_flags.append("💡 Under 30: Catastrophic plan option available — $9,450 deductible but lower premiums.")

    sep = risks.get("sep", {})
    if sep.get("in_open_enrollment"):
        risk_flags.append(f"📅 Open enrollment active — {sep.get('days_remaining')} days left (deadline {sep.get('deadline')}).")
    else:
        risk_flags.append(f"📅 {sep.get('message', 'Open enrollment closed.')} SEP triggers on qualifying life events (60-day window).")

    if processed_plans and all(p.get("type", "").upper() == "HMO" for p in processed_plans):
        risk_flags.append("ℹ️ All plans are HMO — confirm doctors are in-network before enrolling.")
    if subsidy.get("is_medicaid_eligible"):
        risk_flags.append("🏥 Income suggests Medicaid eligibility — check healthcare.gov before paying marketplace premiums.")

    csr_variant = subsidy.get("csr_variant")
    if csr_variant:
        csr_labels = {"94": "$0–$500", "87": "$500–$1,500", "73": "$1,500–$3,000"}
        risk_flags.append(
            f"💡 CSR-{csr_variant}: Silver plans get deductibles ~{csr_labels.get(csr_variant, 'reduced')} "
            f"at the same premium. Do NOT choose Gold or Bronze if CSR-eligible."
        )

    return {
        "location":           loc,
        "subsidy":            subsidy,
        "plans":              processed_plans,
        "medication_coverage": meds,
        "doctor_verification": docs,
        "market_risks":       risks,
        "risk_flags":         risk_flags,
        "sep":                sep,
        "cache_stats":        get_cache_stats(),
        "is_premium":         is_premium,
        "drug_coverage_map":  drug_coverage_map,
    }


# ADK tool wrapper — just calls _collect_analysis_data and stores state
async def run_full_analysis_parallel(tool_context: ToolContext) -> dict:
    """ADK tool: collect data and store in session state for chat continuity."""
    profile = tool_context.state.get("profile", {})
    print(f"[orchestrator] run_full_analysis_parallel called for ZIP {profile.get('zip_code')}")
    try:
        data = await _collect_analysis_data(profile)
        tool_context.state["analysis_data"] = data
        return {"status": "Analysis complete", "plan_count": len(data.get("plans", []))}
    except Exception as e:
        traceback.print_exc()
        return {"status": "Error", "message": str(e)}


# ── PHASE 2: Synthesis Prompt Builder ────────────────────────────────────────

def _build_synthesis_prompt(profile: dict, data: dict, ranking: dict = None) -> str:
    """
    Convert structured analysis data into a rich, human-readable document
    that Gemini can reason over using ORCHESTRATOR_INSTRUCTION.
    Every number cited in the recommendation must come from this document.
    """
    subsidy   = data.get("subsidy", {})
    plans     = data.get("plans", [])
    meds      = data.get("medication_coverage", {})
    docs      = data.get("doctor_verification", {})
    risks     = data.get("market_risks", {})
    flags     = data.get("risk_flags", [])
    sep       = data.get("sep", {})
    loc       = data.get("location", {})
    is_prem   = data.get("is_premium", False)

    income         = profile.get("income", 0)
    age            = profile.get("age", 0)
    household_size = profile.get("household_size", 1)
    utilization    = profile.get("utilization", "sometimes")
    fpl_pct        = subsidy.get("fpl_percentage", 0)
    monthly_aptc   = subsidy.get("monthly_aptc", 0)
    csr            = subsidy.get("csr_variant") or "None"
    is_medicaid    = subsidy.get("is_medicaid_eligible", False)

    # Compute OOP range for Q4
    oop_values = [p.get("oop_max", 0) for p in plans if p.get("oop_max", 0) > 0]
    oop_min = min(oop_values) if oop_values else 0
    oop_max_val = max(oop_values) if oop_values else 0
    oop_gap = oop_max_val - oop_min
    oop_as_pct_income = round(oop_gap / income * 100, 1) if income > 0 else 0

    # Dominant scenario label based on utilization
    dominant_scenario = {
        "rarely":     "Healthy Year (premium is almost the entire cost — rarely triggers deductible)",
        "sometimes":  "Clinical Year (moderate visits + drugs; standard EV 0.3/0.4/0.3 weighting)",
        "frequently": "Clinical Year with elevated weight (drug costs and visit copays dominate)",
        "chronic":    "Worst Case + Clinical (catastrophic OOP exposure is the critical factor)",
    }.get(utilization, "Clinical Year")

    lines = [
        "═" * 60,
        "CoverWise — Full Analysis Data Package",
        "═" * 60,
        "",
        "USER PROFILE",
        "─" * 40,
        f"ZIP: {profile.get('zip_code')} ({loc.get('state','')})  |  Age: {age}  |  Income: ${income:,.0f}/yr",
        f"Household: {household_size}  |  Utilization: {utilization.title()}  |  Premium tier: {'Yes' if is_prem else 'No'}",
        f"Medications: {', '.join(profile.get('drugs', [])) or 'None'}",
        f"Doctors:     {', '.join(profile.get('doctors', [])) or 'None'}",
        "",
        "ELIGIBILITY & SUBSIDY",
        "─" * 40,
        f"FPL: {fpl_pct:.1f}%  |  Monthly APTC: ${monthly_aptc:,.0f}/mo (${monthly_aptc*12:,.0f}/yr)",
        f"CSR Variant: {csr}  |  Medicaid eligible: {is_medicaid}",
        "",
        "REASONING CONTEXT — answer Q1–Q4 in ## Pre-Analysis before writing ## Recommendation",
        "─" * 40,
        f"Q1 UTILIZATION: User is '{utilization}'. Dominant scenario → {dominant_scenario}.",
        f"   Which EV scenario weighting is correct for this person, and does it change the plan ranking?",
        (
            f"Q2 CSR: FPL {fpl_pct:.1f}% → CSR-{csr} applies. "
            f"Deductible for CSR-{csr}: {'~$250 (CSR-94)' if csr=='94' else '~$900 (CSR-87)' if csr=='87' else '~$2,500 (CSR-73)' if csr=='73' else 'see ranges'}. "
            f"Compute: is the Silver plan's premium worth the deductible reduction vs the top Bronze?"
            if csr != "None" else
            f"Q2 CSR: FPL {fpl_pct:.1f}% — no CSR. At this FPL, is Bronze genuinely better than Silver? "
            f"Show the actuarial value difference and what it means for this income."
        ),
        (
            f"Q3 SUBSIDY CLIFF: ⚠ Income ${income:,.0f} is within 10% of 400% FPL. "
            f"What happens to the ${monthly_aptc*12:,.0f}/yr subsidy if income rises $2,000? "
            f"Quantify the exact subsidy loss and whether income management is worth considering."
            if 390 <= fpl_pct <= 410 else
            f"Q3 SUBSIDY CLIFF: No cliff risk at FPL {fpl_pct:.1f}%."
        ),
        f"Q4 OOP RANGE: Plans span ${oop_min:,}–${oop_max_val:,} OOP Max (gap: ${oop_gap:,} = {oop_as_pct_income}% of gross income). "
        f"At income ${income:,.0f}, is the ${oop_gap:,} gap between best and worst case meaningful enough to justify higher premiums?",
        "",
    ]

    # ── Plans table ──────────────────────────────────────────────────────────
    lines += [
        f"PLANS ({len(plans)} found)",
        "─" * 40,
        "| # | Plan | Metal | Type | Gross $/mo | After Subsidy | Deductible | OOP Max | Est. Drug/yr | True Annual | HSA |",
        "|---|------|-------|------|-----------|--------------|-----------|--------|------------|------------|-----|",
    ]
    for i, p in enumerate(plans, 1):
        hsa = "✓" if p.get("hsa_eligible") else "—"
        lines.append(
            f"| {i} | {p.get('name','')} | {p.get('metal_level','')} | {p.get('type','')} "
            f"| ${p.get('premium',0):.0f} | **${p.get('premium_w_credit',0):.0f}** "
            f"| ${p.get('deductible',0):,} | ${p.get('oop_max',0):,} "
            f"| ${p.get('est_annual_drug_cost',0):,.0f} | **${p.get('true_annual_cost',0):,.0f}** | {hsa} |"
        )

    # ── Scenario tables ──────────────────────────────────────────────────────
    lines += [
        "",
        "SCENARIO ANALYSIS (pre-computed)",
        "─" * 40,
        "Healthy Year (premiums only, no healthcare use):",
        "| Plan | Monthly | Annual Premium |",
        "|------|---------|---------------|",
    ]
    for p in plans:
        lines.append(f"| {p.get('name','')} | ${p.get('premium_w_credit',0):.0f} | ${p.get('scenario_healthy',0):,.0f} |")

    lines += ["", "Clinical Year (premiums + all drug costs, no deductible hit):"]
    lines += ["| Plan | Annual Premium | Est. Drug Cost | Total |", "|------|--------------|--------------|-------|"]
    for p in plans:
        lines.append(
            f"| {p.get('name','')} | ${p.get('premium_w_credit',0)*12:.0f} "
            f"| ${p.get('est_annual_drug_cost',0):,.0f} | **${p.get('scenario_clinical',0):,.0f}** |"
        )

    lines += ["", "Worst Case (premiums + full OOP Max):"]
    lines += ["| Plan | Annual Premium | OOP Max | Ceiling Cost |", "|------|--------------|--------|-------------|"]
    for p in plans:
        lines.append(
            f"| {p.get('name','')} | ${p.get('premium_w_credit',0)*12:.0f} "
            f"| ${p.get('oop_max',0):,} | **${p.get('scenario_worst',0):,.0f}** |"
        )

    # ── Breakeven helper data ────────────────────────────────────────────────
    if len(plans) >= 2:
        p1, p2 = plans[0], plans[1]
        premium_diff = abs(p1.get("premium_w_credit", 0) - p2.get("premium_w_credit", 0)) * 12
        deduct_diff  = abs(p1.get("deductible", 0) - p2.get("deductible", 0))
        lines += [
            "",
            "BREAKEVEN DATA (use to calculate visits-to-breakeven)",
            "─" * 40,
            f"Plan 1 vs Plan 2 annual premium difference: ${premium_diff:,.0f}",
            f"Plan 1 vs Plan 2 deductible difference:     ${deduct_diff:,.0f}",
            f"Typical specialist visit cost: ~$200–$350 out-of-pocket before deductible",
        ]

    # ── Drug coverage detail ─────────────────────────────────────────────────
    lines += ["", "DRUG COVERAGE DETAIL", "─" * 40]
    resolved = meds.get("resolved_drugs", [])
    cov_list = meds.get("coverage_details", [])
    generics = meds.get("generic_suggestions", {})

    for drug in resolved:
        rxcui = str(drug.get("rxcui", ""))
        lines += [
            f"{drug.get('name','')} — RxCUI {rxcui}  |  "
            f"Route: {drug.get('route','')}  |  Strength: {drug.get('strength','')}",
            "| Plan | Coverage | Tier | Prior Auth | Step Therapy | Qty Limit | Est. Copay/mo | Source |",
            "|------|----------|------|-----------|-------------|---------|-------------|--------|",
        ]
        for p in plans:
            pid   = p.get("id")
            cov   = next((c for c in cov_list if str(c.get("rxcui","")) == rxcui and c.get("plan_id") == pid), {})
            pname = p.get("name", "")[:35]
            tier  = cov.get("drug_tier") or "—"
            pa    = "YES ⚠" if cov.get("prior_authorization") else "no"
            st    = "YES" if cov.get("step_therapy") else "no"
            ql    = "YES" if cov.get("quantity_limit") else "no"
            dd    = next((x for x in p.get("drug_detail",[]) if x.get("rxcui") == rxcui), {})
            if dd.get("coverage") == "Covered":
                copay = dd.get("copay_display") or f"${dd.get('copay_mo',0):.0f}/mo"
            else:
                copay = "—"
            src   = "RAG✓" if cov.get("source") == "rag_formulary" else "CMS"
            note  = " ⚡proxy" if cov.get("note") else ""
            lines.append(f"| {pname} | {cov.get('coverage','—')} | {tier} | {pa} | {st} | {ql} | {copay} | {src}{note} |")

        gen = generics.get(drug.get("name",""), [])
        if gen:
            lines.append(f"  Generic alternatives: {', '.join(gen)}")
        lines.append("")

    # ── Doctor verification ──────────────────────────────────────────────────
    lines += ["DOCTOR VERIFICATION", "─" * 40]
    for doc in docs.get("results", []):
        mips  = f"MIPS {doc.get('mips_score')}/100" if doc.get("mips_score") is not None else "MIPS: not available"
        lines += [
            f"Searched: {doc.get('searched_name','')}",
            f"  Verified: {doc.get('name','')}  |  NPI: {doc.get('npi','')}",
            f"  Specialty: {doc.get('specialty','')}  |  {doc.get('city','')}, {doc.get('state','')}",
            f"  Phone: {doc.get('phone','')}  |  Credential: {doc.get('credential','')}  |  Active: {doc.get('active',True)}",
            f"  {mips}  |  Telehealth: {doc.get('telehealth', False)}",
            "  Network status:",
        ]
        for pid, net in doc.get("network_status", {}).items():
            pname = next((p.get("name","") for p in plans if p.get("id") == pid), pid)
            status = "In-Network ✓" if net.get("in_network") else ("Out-of-Network ✗" if net.get("in_network") is False else "Unknown")
            url_note = f"  → verify: {net.get('verify_url','')}" if net.get("verify_url") else ""
            lines.append(f"    • {pname[:40]}: {status}{url_note}")
        lines.append("")

    # ── Market / SEP ─────────────────────────────────────────────────────────
    lines += [
        "MARKET & RISK DATA",
        "─" * 40,
        f"Enrollment: {'Open — ' + str(sep.get('days_remaining','')) + ' days left (deadline ' + str(sep.get('deadline','')) + ')' if sep.get('in_open_enrollment') else sep.get('message','Closed')}",
        f"Actuarial values: Bronze=60% | Silver=70% | Gold=80% | Platinum=90%",
        f"CSR-94 deductible range: $0–$500  |  CSR-87: $500–$1,500  |  CSR-73: $1,500–$3,000",
        "",
        "RISK FLAGS",
        "─" * 40,
    ]
    for f in flags:
        lines.append(f"  {f}")

    # ── LLM Ranking Agent output (NON-NEGOTIABLE ORDER) ─────────────────────
    if ranking:
        ev      = ranking.get("expected_value_ranking", [])
        top     = ranking.get("top_recommendation", {})
        csr_ov  = ranking.get("csr_override")
        rf      = ranking.get("red_flags", [])
        r_scen  = ranking.get("rankings", {})

        lines += [
            "",
            "╔" + "═" * 58 + "╗",
            "║  FINAL PLAN ORDER (NON-NEGOTIABLE — PHASE 1.5 LLM RANKING) ║",
            "╚" + "═" * 58 + "╝",
            "YOU MUST present plans in this exact order in every table and paragraph.",
            "Do NOT re-rank, re-sort, or reorder — your job is explanation only.",
            "",
        ]

        if ev:
            lines += [
                "EXPECTED VALUE RANKING (definitive order):",
                "| EV Rank | Plan Name | EV Score | Key Reason |",
                "|---------|-----------|---------|-----------|",
            ]
            for r in ev:
                marker = "★ #1 ← TOP PICK" if r.get("rank") == 1 else f"#{r.get('rank')}"
                lines.append(
                    f"| {marker} | {r.get('plan_name','')} | **${r.get('ev_score',0):,.0f}** | {r.get('key_reason','')} |"
                )

        if top:
            lines += [
                "",
                f"TOP RECOMMENDATION: {top.get('plan_name','')}",
                f"Rationale: {top.get('rationale','')}",
            ]

        if csr_ov:
            lines.append(f"CSR OVERRIDE: Silver plan {csr_ov} must be prioritised — CSR deductible reduction applies.")

        for scen_key, scen_label in [
            ("healthy_year",  "Healthy Year"),
            ("clinical_year", "Clinical Year"),
            ("worst_case",    "Worst Case"),
        ]:
            rows = r_scen.get(scen_key, [])
            if rows:
                lines += [
                    "",
                    f"{scen_label} order:",
                    "| Rank | Plan | Annual Cost |",
                    "|------|------|------------|",
                ]
                for r in rows:
                    lines.append(
                        f"| #{r.get('rank')} | {r.get('plan_name','')} | **${r.get('annual_cost',0):,.0f}** |"
                    )

        if rf:
            lines += ["", "Red Flags from Ranking Agent:"]
            for flag in rf:
                lines.append(f"  ⚠ {flag}")

        lines.append("")

    # ── Synthesis instruction ────────────────────────────────────────────────
    lines += [
        "",
        "═" * 60,
        "SYNTHESIS INSTRUCTION",
        "═" * 60,
        "Using ALL data above, produce a rigorous multi-pillar recommendation.",
        "",
        "MANDATORY RESPONSE ORDER:",
        "  1. Start with '## Pre-Analysis' — answer Q1–Q4 from REASONING CONTEXT above.",
        "     Each answer must be 1-2 sentences and show actual numbers from the data.",
        "  2. Then write '## Recommendation' — full 4-pillar analysis.",
        "  Do NOT skip Pre-Analysis. Do NOT merge it into Recommendation.",
        "",
    ]

    if ranking:
        ev_top3 = ranking.get("expected_value_ranking", [])[:3]
        order_str = " → ".join(f"#{r.get('rank')} {r.get('plan_name','')}" for r in ev_top3)
        trade_off = ranking.get("scenario_trade_off", "")
        csr_exp   = ranking.get("csr_explanation", "")
        util_rsn  = ranking.get("utilization_weight_reasoning", "")
        lines += [
            f"MANDATORY PLAN ORDER: {order_str}",
            "Every section, table, and paragraph MUST follow the FINAL PLAN ORDER above.",
            "Start ## Recommendation with a summary table matching the EV Ranking exactly.",
            "",
        ]
        if util_rsn:
            lines += [f"RANKING AGENT UTILIZATION REASONING: {util_rsn}", ""]
        if trade_off:
            lines += [f"RANKING AGENT SCENARIO TRADE-OFF: {trade_off}", ""]
        if csr_exp:
            lines += [f"RANKING AGENT CSR ANALYSIS: {csr_exp}", ""]
    else:
        lines.append("No pre-computed ranking — apply the Simulated Year protocol from your system instructions.")

    lines += [
        "FINANCIAL PILLAR REQUIREMENTS:",
        "  - Scenario table with Healthy/Clinical/Worst for each plan.",
        "  - Breakeven arithmetic: 'Plan A saves $[X]/yr vs Plan B. Plan B deductible is $[Y] lower.",
        "    At $275/specialist visit: breakeven = [Y÷X×275] extra visits.'",
        "  - Translate actuarial value (Bronze 60%) into plain dollars for this income.",
        "",
        "MEDICAL PILLAR REQUIREMENTS:",
        "  - For each drug with Prior Auth: explain what PA means in practical terms (delay, appeal process).",
        "  - For Step Therapy: explain the trial-and-fail process timeline.",
        "  - Quantify generic savings if alternatives are listed.",
        "  - If no drugs provided: write 'No medications provided — skipping Medical Pillar.'",
        "",
        "NETWORK PILLAR REQUIREMENTS:",
        "  - Confirm every doctor's status by name and NPI.",
        "  - Interpret MIPS score (90+ = top 10%, 75-89 = top 25%, <50 = below average).",
        "  - If no doctors provided: write 'No doctors provided — skipping Network Pillar.'",
        "",
        f"{'Apply PREMIUM TIER RULES: 3× detail, HSA wealth forecast table (5-year), side-by-side benefit table.' if is_prem else 'Free tier: cover top 3 plans only.'}",
        "Do NOT invent numbers. Every dollar figure must appear verbatim in the tables above.",
        "═" * 60,
    ]

    return "\n".join(lines)


# ── PHASE 1.5: Python EV Ranking (no LLM call — instant, deterministic) ──────

def _rank_plans_python(data: dict, profile: dict = None) -> dict:
    """
    Compute EV ranking entirely in Python.
    EV = (w_healthy × healthy_year) + (w_clinical × clinical_year) + (w_worst × worst_case).
    Eliminates a full Gemini round-trip; the synthesis agent explains the WHY in its text.
    """
    profile = profile or {}
    plans   = data.get("plans", [])
    subsidy = data.get("subsidy", {})
    flags   = data.get("risk_flags", [])

    if not plans:
        return {}

    utilization = profile.get("utilization", "sometimes")
    w_h, w_c, w_w = {
        "rarely":     (0.5, 0.3, 0.2),
        "sometimes":  (0.3, 0.4, 0.3),
        "frequently": (0.2, 0.5, 0.3),
        "chronic":    (0.15, 0.4, 0.45),
    }.get(utilization, (0.3, 0.4, 0.3))

    scored = sorted(
        [(round(w_h * p["scenario_healthy"] + w_c * p["scenario_clinical"] + w_w * p["scenario_worst"], 2), p)
         for p in plans],
        key=lambda x: x[0],
    )

    ev_ranking = [
        {"rank": i + 1, "plan_id": p.get("id"), "plan_name": p.get("name"),
         "ev_score": ev, "key_reason": ""}
        for i, (ev, p) in enumerate(scored)
    ]

    def scenario_rank(key):
        return [
            {"rank": i + 1, "plan_id": p.get("id"), "plan_name": p.get("name"),
             "annual_cost": round(p.get(key, 0), 2), "reason": ""}
            for i, p in enumerate(sorted(plans, key=lambda p: p.get(key, 0)))
        ]

    top = ev_ranking[0] if ev_ranking else {}

    csr_variant    = subsidy.get("csr_variant")
    csr_override   = None
    csr_explanation = None
    if csr_variant:
        silver = next((p for _, p in scored if p.get("metal_level", "").upper() == "SILVER"), None)
        if silver:
            csr_override    = silver.get("id")
            csr_explanation = (
                f"CSR-{csr_variant} applies — Silver plan deductible is dramatically reduced. "
                "Strongly prefer Silver over Bronze regardless of raw EV."
            )

    util_reasoning = {
        "rarely":     f"'rarely' → premium is almost the entire annual spend. Weights: {w_h}/{w_c}/{w_w} (Healthy/Clinical/Worst).",
        "sometimes":  f"'sometimes' → moderate use expected. Weights: {w_h}/{w_c}/{w_w} (Healthy/Clinical/Worst).",
        "frequently": f"'frequently' → drug costs and visits dominate. Weights: {w_h}/{w_c}/{w_w} (Healthy/Clinical/Worst).",
        "chronic":    f"'chronic' → catastrophic OOP is the primary risk. Weights: {w_h}/{w_c}/{w_w} (Healthy/Clinical/Worst).",
    }.get(utilization, f"Standard weights {w_h}/{w_c}/{w_w} applied.")

    print(f"[ranking] Python EV ranking complete — top: {top.get('plan_name')} EV=${top.get('ev_score')}")

    return {
        "utilization_weight_reasoning": util_reasoning,
        "rankings": {
            "healthy_year":  scenario_rank("scenario_healthy"),
            "clinical_year": scenario_rank("scenario_clinical"),
            "worst_case":    scenario_rank("scenario_worst"),
        },
        "expected_value_ranking": ev_ranking,
        "scenario_trade_off": "",
        "top_recommendation": {
            "plan_id":   top.get("plan_id", ""),
            "plan_name": top.get("plan_name", ""),
            "rationale": (
                f"Lowest EV score of ${top.get('ev_score', 0):,.2f} using {utilization}-adjusted "
                f"weights ({w_h}/{w_c}/{w_w} Healthy/Clinical/Worst)."
            ),
        },
        "csr_override":    csr_override,
        "csr_explanation": csr_explanation,
        "red_flags":       [f for f in flags if any(c in f for c in ("⚠️", "🚨", "⚠"))],
    }


# ── PHASE 2: Gemini Direct Synthesis ─────────────────────────────────────────

async def _synthesize_with_gemini(synthesis_prompt: str, is_premium: bool = False) -> str:
    """
    Call Gemini 2.5 Pro (thinking) with ORCHESTRATOR_INSTRUCTION as system prompt
    and the full structured data document as the user turn.
    Falls back to a data-only summary if Vertex AI is unavailable.
    """
    if not VERTEXAI_AVAILABLE:
        return (
            "AI synthesis unavailable (google-genai not installed). "
            "The full plan, drug, and doctor data is available in the structured sections above."
        )

    try:
        if USE_VERTEXAI:
            client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
        else:
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=synthesis_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=ORCHESTRATOR_INSTRUCTION,
                max_output_tokens=32768,
                temperature=0.7,
            ),
        )
        return response.text.replace("*", "")
    except Exception as e:
        traceback.print_exc()
        return f"Synthesis error: {e}\n\nRaw data has been returned in the structured fields."


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
            model="gemini-2.5-flash",
            instruction=ORCHESTRATOR_INSTRUCTION,
            tools=[run_full_analysis_parallel]
        )
        self._runner = Runner(agent=agent, app_name=APP_NAME, session_service=self._session_service)

    async def analyze(self, profile: dict) -> dict:
        """
        Two-phase agentic pipeline:
          Phase 1   — parallel data collection (Python, no AI — 3 waves of API calls)
          Phase 1.5 — Python EV ranking (instant, deterministic — no LLM call)
          Phase 2   — Gemini synthesis (single LLM call: full 4-pillar recommendation)
        """
        user_id = profile.get("user_id", "anonymous")
        print(f"[orchestrator] analyze() start — user={user_id} ZIP={profile.get('zip_code')}")

        # ── Phase 1: data collection ─────────────────────────────────────────
        data = await _collect_analysis_data(profile)
        print(f"[orchestrator] Phase 1 complete — {len(data.get('plans',[]))} plans collected")

        # ── Phase 1.5: Python EV ranking (instant) ────────────────────────────
        ranking = _rank_plans_python(data, profile)
        if ranking:
            data["llm_ranking"] = ranking
        else:
            print("[orchestrator] Phase 1.5 skipped — no plans to rank")

        # ── Phase 2: synthesis ───────────────────────────────────────────────
        synthesis_prompt = _build_synthesis_prompt(profile, data, ranking=ranking or None)
        print(f"[orchestrator] synthesis prompt built ({len(synthesis_prompt)} chars) — calling Gemini")
        recommendation = await _synthesize_with_gemini(synthesis_prompt, data.get("is_premium", False))
        print(f"[orchestrator] recommendation received ({len(recommendation)} chars)")

        data["recommendation"] = recommendation

        # Store in ADK session for chat follow-ups
        if ADK_AVAILABLE and self._session_service:
            self._ensure_runner()
            try:
                await self._session_service.create_session(
                    app_name=APP_NAME, user_id=user_id, session_id=user_id,
                    state={"profile": profile, "analysis_data": data}
                )
            except Exception:
                try:
                    session = await self._session_service.get_session(
                        app_name=APP_NAME, user_id=user_id, session_id=user_id
                    )
                    session.state["profile"] = profile
                    session.state["analysis_data"] = data
                except Exception:
                    pass

        store_user_profile(user_id, profile)

        sub = data.get("subsidy", {})
        return {
            "route": "medicaid" if sub.get("is_medicaid_eligible") else "subsidized",
            "profile": {
                "fpl_percentage": sub.get("fpl_percentage"),
                "route_reason":   "Based on income and location analysis.",
            },
            "recommendation":       recommendation,
            "llm_ranking":          data.get("llm_ranking", {}),
            "synthesis_prompt_len": len(synthesis_prompt),
            "plans":     data.get("plans", []),
            "subsidy":   sub,
            "drugs":     data.get("medication_coverage", {}),
            "doctors":   data.get("doctor_verification", {}),
            "risks":     {**data.get("market_risks", {}), "flags": data.get("risk_flags", [])},
            "cache_stats": data.get("cache_stats", {}),
            "medication_coverage": data.get("medication_coverage", {}),
        }

    async def chat(self, user_id: str, message: str, profile: Optional[dict] = None, context: str = "") -> dict:
        """
        Follow-up chat: loads prior analysis from session, builds context,
        calls Gemini directly with ORCHESTRATOR_INSTRUCTION + user question.
        """
        # ── Load prior analysis data ──────────────────────────────────────────
        analysis_data: dict = {}
        user_profile  = profile or {}

        if ADK_AVAILABLE and self._session_service:
            self._ensure_runner()
            try:
                session = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=user_id
                )
                analysis_data = session.state.get("analysis_data", {})
                user_profile  = session.state.get("profile", profile or {})
            except Exception:
                pass

        # If no prior analysis, collect minimal data (plans + subsidy only)
        if not analysis_data.get("plans") and user_profile.get("zip_code"):
            try:
                analysis_data = await _collect_analysis_data(user_profile)
            except Exception:
                pass

        # ── Build chat context from synthesis prompt + user question ──────────
        data_doc = _build_synthesis_prompt(user_profile, analysis_data) if analysis_data else ""
        prior_rec = (analysis_data.get("recommendation") or "")[:3000]

        chat_prompt = "\n".join([
            data_doc,
            "",
            "─" * 60,
            f"PRIOR RECOMMENDATION SUMMARY:\n{prior_rec}" if prior_rec else "",
            f"EXTRA CONTEXT (e.g. Employer Plan Comparison run by user):\n{context}" if context else "",
            "─" * 60,
            f"USER FOLLOW-UP QUESTION: {message}",
            "",
            "Answer using ONLY the verified data above. Cite exact plan names, dollar amounts, tiers, "
            "and doctor names. Be concise and direct.",
        ])

        reply = await _synthesize_with_gemini(chat_prompt)
        return {"reply": reply, "memory_used": bool(prior_rec)}
