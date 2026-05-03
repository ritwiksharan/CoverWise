
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
except ImportError:
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
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False

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
AGENTIC PLAN RANKING PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Do NOT simply sort by the lowest premium. Use a "Simulated Year" reasoning logic to rank the Top 3 plans:
1. **The 'Healthy Year' (Low Use)**: Rank by [Monthly Premium * 12]. This is the "Floor" cost.
2. **The 'Clinical Year' (Chronic Care)**: Rank by how well the plan covers the user's specific drugs/doctors. Prioritize $0 copays.
3. **The 'Worst Case' (Catastrophic)**: Rank by [Monthly Premium * 12 + OOP Max]. This is the "Ceiling" risk.

**Final Decision**: Recommend the plan that provides the best **Expected Value** across all three scenarios. If a user is eligible for **CSR (Cost Sharing Reductions)**, you MUST prioritize **Silver** plans and explain the massive deductible drop.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURE & FORMATTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ALWAYS use Markdown tables for comparisons. Start lines with `|`.
• Use `**` for all currency amounts.
• Tone: Objective, mathematical, and authoritative.
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

    # Wave 2 — drugs / doctors / risks (parallel)
    meds, docs, risks = await asyncio.gather(
        asyncio.to_thread(check_medication_coverage, drugs, plan_ids),
        asyncio.to_thread(verify_doctors, doctors, state, zip_code, plan_ids, plans[:plan_limit]),
        asyncio.to_thread(get_market_risks, zip_code, state),
    )

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

    hrsa = risks.get("hrsa", {})
    if hrsa.get("shortage_area"):
        risk_flags.append(f"🏥 {hrsa.get('message', 'HPSA area.')} Prefer PPO/EPO with $0 telehealth.")

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

    # ── Market / HRSA / SEP ──────────────────────────────────────────────────
    hrsa = risks.get("hrsa", {})
    lines += [
        "MARKET & RISK DATA",
        "─" * 40,
        f"HRSA shortage area: {hrsa.get('shortage_area', False)}  |  {hrsa.get('message','')}",
        f"Enrollment: {'Open — ' + str(sep.get('days_remaining','')) + ' days left (deadline ' + str(sep.get('deadline','')) + ')' if sep.get('in_open_enrollment') else sep.get('message','Closed')}",
        f"Actuarial values: Bronze=60% | Silver=70% | Gold=80% | Platinum=90%",
        f"CSR-94 deductible range: $0–$500  |  CSR-87: $500–$1,500  |  CSR-73: $1,500–$3,000",
        "",
        "RISK FLAGS",
        "─" * 40,
    ]
    for f in flags:
        lines.append(f"  {f}")

    # ── LLM Ranking Agent output ─────────────────────────────────────────────
    if ranking:
        import json as _json
        lines += ["", "LLM RANKING AGENT OUTPUT (Phase 1.5 — use this as your starting ranking)", "─" * 40]
        top = ranking.get("top_recommendation", {})
        if top:
            lines.append(f"Top Recommendation: {top.get('plan_name','')} — {top.get('rationale','')}")
        csr_ov = ranking.get("csr_override")
        if csr_ov:
            lines.append(f"CSR Override: {csr_ov}")
        ev = ranking.get("expected_value_ranking", [])
        if ev:
            lines += ["Expected Value Ranking:", "| Rank | Plan | EV Score | Key Reason |", "|------|------|---------|-----------|"]
            for r in ev:
                lines.append(f"| {r.get('rank')} | {r.get('plan_name','')} | ${r.get('ev_score',0):,.0f} | {r.get('key_reason','')} |")
        rf = ranking.get("red_flags", [])
        if rf:
            lines.append("Red Flags from Ranking Agent:")
            for flag in rf:
                lines.append(f"  • {flag}")
        lines.append("")

    # ── Synthesis instruction ────────────────────────────────────────────────
    lines += [
        "",
        "═" * 60,
        "SYNTHESIS INSTRUCTION",
        "═" * 60,
        "Using ALL data above, produce a complete multi-pillar recommendation.",
        "The LLM Ranking Agent (Phase 1.5) has already computed the plan ordering — use it as your starting point.",
        "Follow the exact structure in your system instructions:",
        "  1. Financial Pillar — use the scenario tables and breakeven data above.",
        "  2. Medical Pillar — cite exact tiers, PA requirements, and real copay strings from the drug tables.",
        "  3. Network Pillar — confirm each doctor's verified name, NPI, MIPS score, and network status.",
        "  4. Market Pillar — cite HRSA and enrollment deadline from the data above.",
        "Apply the Agentic Plan Ranking Protocol (Healthy / Clinical / Worst-Case scenarios) — confirm or adjust the LLM agent's ranking with your deeper analysis.",
        f"{'Apply PREMIUM TIER RULES: 3× detail, HSA wealth forecast, side-by-side benefit table.' if is_prem else 'Free tier: focus on the top 3 plans.'}",
        "Do NOT invent any numbers. Every dollar figure must match the tables above exactly.",
        "═" * 60,
    ]

    return "\n".join(lines)


# ── PHASE 1.5: LLM Ranking Agent ─────────────────────────────────────────────

_RANKING_INSTRUCTION = """You are a health insurance plan ranking agent.
Given structured plan data, rank the plans using a Simulated Year model:
  1. Healthy Year — rank by lowest annual premium only
  2. Clinical Year — rank by lowest (premium + estimated drug costs)
  3. Worst Case   — rank by lowest (premium + OOP max)
  4. Expected Value — weighted average (0.3 * healthy + 0.4 * clinical + 0.3 * worst)

Return ONLY valid JSON matching exactly:
{
  "rankings": {
    "healthy_year":   [{"rank": 1, "plan_id": "...", "plan_name": "...", "annual_cost": 0, "reason": "..."}],
    "clinical_year":  [{"rank": 1, "plan_id": "...", "plan_name": "...", "annual_cost": 0, "reason": "..."}],
    "worst_case":     [{"rank": 1, "plan_id": "...", "plan_name": "...", "annual_cost": 0, "reason": "..."}]
  },
  "expected_value_ranking": [{"rank": 1, "plan_id": "...", "plan_name": "...", "ev_score": 0, "key_reason": "..."}],
  "top_recommendation": {"plan_id": "...", "plan_name": "...", "rationale": "..."},
  "csr_override": null,
  "red_flags": []
}
If a Silver plan has CSR, set csr_override to its plan_id and explain the deductible drop in rationale.
red_flags: list any PA/step-therapy issues, OOP cliff risks, or doctor network warnings."""


async def _rank_plans_with_llm(data: dict) -> dict:
    """
    Phase 1.5 — LLM Ranking Agent.
    Calls Gemini with structured JSON output to rank plans across all three scenarios.
    Returns ranking dict embedded in the synthesis prompt so Phase 2 sees reasoned ordering.
    """
    if not VERTEXAI_AVAILABLE:
        return {}

    plans  = data.get("plans", [])
    subsidy = data.get("subsidy", {})
    flags  = data.get("risk_flags", [])

    # Build compact input for the ranking agent
    plan_rows = []
    for p in plans:
        row = {
            "plan_id":      p.get("id"),
            "plan_name":    p.get("name"),
            "metal_level":  p.get("metal_level"),
            "plan_type":    p.get("type"),
            "net_monthly":  p.get("premium_w_credit", 0),
            "annual_premium": p.get("scenario_healthy", 0),
            "clinical_year":  p.get("scenario_clinical", 0),
            "worst_case":     p.get("scenario_worst", 0),
            "deductible":     p.get("deductible", 0),
            "oop_max":        p.get("oop_max", 0),
            "hsa_eligible":   p.get("hsa_eligible", False),
            "est_drug_cost_annual": p.get("est_annual_drug_cost", 0),
            "pa_warning":     p.get("pa_warning", False),
            "drug_tiers":     [
                {"name": dd.get("name"), "tier": dd.get("tier"), "pa": dd.get("pa"),
                 "st": dd.get("st"), "copay_display": dd.get("copay_display")}
                for dd in p.get("drug_detail", [])
            ],
        }
        plan_rows.append(row)

    ranking_input = {
        "plans": plan_rows,
        "csr_variant": subsidy.get("csr_variant"),
        "monthly_aptc": subsidy.get("monthly_aptc", 0),
        "is_medicaid_eligible": subsidy.get("is_medicaid_eligible", False),
        "risk_flags": flags[:5],
    }

    import json
    ranking_prompt = (
        "Rank the following insurance plans across all three Simulated Year scenarios.\n\n"
        f"INPUT DATA:\n{json.dumps(ranking_input, indent=2)}\n\n"
        "Return ONLY the JSON specified in your system instructions. No extra text."
    )

    try:
        vertexai.init(project=PROJECT_ID, location=REGION)
        model = GenerativeModel(
            "gemini-2.0-flash-001",
            system_instruction=_RANKING_INSTRUCTION,
            generation_config=GenerationConfig(
                max_output_tokens=2048,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        response = await asyncio.to_thread(model.generate_content, ranking_prompt)
        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"[ranking_agent] LLM ranking failed: {e}")
        return {}


# ── PHASE 2: Gemini Direct Synthesis ─────────────────────────────────────────

async def _synthesize_with_gemini(synthesis_prompt: str, is_premium: bool = False) -> str:
    """
    Call Gemini 2.0 Flash directly with ORCHESTRATOR_INSTRUCTION as system prompt
    and the full structured data document as the user turn.
    Falls back to a data-only summary if Vertex AI is unavailable.
    """
    if not VERTEXAI_AVAILABLE:
        return (
            "**AI synthesis unavailable** (Vertex AI not configured locally). "
            "The full plan, drug, and doctor data is available in the structured sections above."
        )

    try:
        vertexai.init(project=PROJECT_ID, location=REGION)
        model = GenerativeModel(
            "gemini-2.0-flash-001",
            system_instruction=ORCHESTRATOR_INSTRUCTION,
            generation_config=GenerationConfig(
                max_output_tokens=8192,
                temperature=0.2,
            ),
        )
        response = await asyncio.to_thread(model.generate_content, synthesis_prompt)
        return response.text
    except Exception as e:
        traceback.print_exc()
        return f"**Synthesis error**: {e}\n\nRaw data has been returned in the structured fields."


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
        """
        Three-phase agentic pipeline:
          Phase 1   — parallel data collection (Python, no AI — 3 waves of API calls)
          Phase 1.5 — LLM Ranking Agent (Gemini JSON mode: ranks plans by scenario)
          Phase 2   — LLM Synthesis Agent (Gemini text: full 4-pillar recommendation)
        """
        user_id = profile.get("user_id", "anonymous")
        print(f"[orchestrator] analyze() start — user={user_id} ZIP={profile.get('zip_code')}")

        # ── Phase 1: data collection ─────────────────────────────────────────
        data = await _collect_analysis_data(profile)
        print(f"[orchestrator] Phase 1 complete — {len(data.get('plans',[]))} plans collected")

        # ── Phase 1.5: LLM Ranking Agent ─────────────────────────────────────
        ranking = await _rank_plans_with_llm(data)
        if ranking:
            data["llm_ranking"] = ranking
            print(f"[orchestrator] Phase 1.5 complete — LLM ranking: {ranking.get('top_recommendation',{}).get('plan_name','n/a')}")
        else:
            print("[orchestrator] Phase 1.5 skipped (Vertex AI unavailable or ranking failed)")

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

    async def chat(self, user_id: str, message: str, profile: Optional[dict] = None) -> dict:
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
            "─" * 60,
            f"USER FOLLOW-UP QUESTION: {message}",
            "",
            "Answer using ONLY the verified data above. Cite exact plan names, dollar amounts, tiers, "
            "and doctor names. Be concise and direct.",
        ])

        reply = await _synthesize_with_gemini(chat_prompt)
        return {"reply": reply, "memory_used": bool(prior_rec)}
