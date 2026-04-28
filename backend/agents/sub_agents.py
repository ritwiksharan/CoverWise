"""
Sub-Agents — each handles one domain of insurance analysis
All run in parallel via asyncio, orchestrator merges results
"""

import asyncio
from tools.gov_apis import (
    calculate_fpl_percentage, get_fpl_thresholds, search_plans,
    check_drug_formulary, verify_doctor_npi, get_medicaid_threshold, get_fips_from_zip
)

# ─── PROFILE AGENT ────────────────────────────────────────────────────────────

async def profile_agent(profile: dict) -> dict:
    """Gather and validate user profile. Entry point for handoff routing."""
    await asyncio.sleep(0)  # yield to event loop
    fpl_pct = calculate_fpl_percentage(profile["income"], profile["household_size"])
    fips = get_fips_from_zip(profile["zip_code"])
    medicaid_threshold = get_medicaid_threshold(fips or "")
    
    route = determine_route(fpl_pct, medicaid_threshold)
    
    return {
        "agent": "profile",
        "fpl_percentage": fpl_pct,
        "fips": fips,
        "route": route,
        "route_reason": get_route_reason(route, fpl_pct),
    }

def determine_route(fpl_pct: float, medicaid_threshold: float) -> str:
    """
    HANDOFF LOGIC — routes user to correct coverage path.
    This is the key agent handoff concept.
    """
    if fpl_pct < medicaid_threshold:
        return "medicaid"          # → Medicaid/CHIP agent, skip marketplace entirely
    elif fpl_pct < 138:
        return "chip"              # → CHIP for children
    elif fpl_pct <= 400:
        return "subsidized"        # → Subsidized marketplace plans (APTC eligible)
    elif fpl_pct <= 600:
        return "marketplace"       # → Unsubsidized marketplace
    else:
        return "full_price"        # → Full price, consider employer/short-term

def get_route_reason(route: str, fpl_pct: float) -> str:
    reasons = {
        "medicaid": f"At {fpl_pct}% FPL you likely qualify for free Medicaid coverage. No marketplace plan needed.",
        "chip": f"At {fpl_pct}% FPL children in your household may qualify for CHIP.",
        "subsidized": f"At {fpl_pct}% FPL you qualify for ACA premium tax credits (APTC).",
        "marketplace": f"At {fpl_pct}% FPL you don't qualify for subsidies but can use the marketplace.",
        "full_price": f"At {fpl_pct}% FPL consider employer coverage or full-price marketplace plans.",
    }
    return reasons.get(route, "")

# ─── SUBSIDY AGENT ────────────────────────────────────────────────────────────

async def subsidy_agent(profile: dict, fpl_pct: float) -> dict:
    """Calculate ACA premium tax credit (APTC) amount."""
    await asyncio.sleep(0)
    
    if fpl_pct > 400:
        return {"agent": "subsidy", "eligible": False, "monthly_credit": 0, "annual_credit": 0}
    
    # IRS applicable percentage table (2024)
    if fpl_pct <= 133: pct_of_income = 0.0
    elif fpl_pct <= 150: pct_of_income = 0.0
    elif fpl_pct <= 200: pct_of_income = 0.02
    elif fpl_pct <= 250: pct_of_income = 0.04
    elif fpl_pct <= 300: pct_of_income = 0.06
    elif fpl_pct <= 400: pct_of_income = 0.085
    else: pct_of_income = 0.085

    benchmark_premium = 450  # estimated Silver benchmark
    max_contribution = (profile["income"] * pct_of_income) / 12
    monthly_credit = max(0, benchmark_premium - max_contribution)
    
    csr_eligible = 138 <= fpl_pct <= 250
    
    return {
        "agent": "subsidy",
        "eligible": True,
        "monthly_credit": round(monthly_credit, 2),
        "annual_credit": round(monthly_credit * 12, 2),
        "csr_eligible": csr_eligible,
        "csr_note": "You qualify for Cost Sharing Reductions on Silver plans — lower deductibles and copays." if csr_eligible else None,
        "subsidy_cliff_warning": 390 <= fpl_pct <= 410,
    }

# ─── PLAN SEARCH AGENT ────────────────────────────────────────────────────────

async def plan_search_agent(profile: dict) -> dict:
    """Fetch available plans from CMS Marketplace."""
    await asyncio.sleep(0)
    plans = search_plans(
        profile["zip_code"], profile["age"],
        profile["income"], profile["household_size"]
    )
    return {"agent": "plan_search", "plans": plans, "plan_count": len(plans)}

# ─── DRUG CHECK AGENT ─────────────────────────────────────────────────────────

async def drug_check_agent(profile: dict, plans: list) -> dict:
    """Check drug formulary coverage for all user medications."""
    await asyncio.sleep(0)
    
    if not profile.get("drugs"):
        return {"agent": "drug_check", "results": [], "warnings": []}
    
    results = []
    warnings = []
    
    for drug in profile["drugs"]:
        # Check against first plan as representative (production: check all)
        plan_id = plans[0]["id"] if plans else "DEMO"
        info = check_drug_formulary(drug, plan_id)
        results.append(info)
        
        if info["tier"] >= 4:
            warnings.append(f"⚠️ {drug} is Tier {info['tier']} — estimated ${info['monthly_cost']}/month. Consider plans with better formulary coverage.")
        if info.get("prior_auth"):
            warnings.append(f"🔒 {drug} requires prior authorization on most plans.")
    
    return {"agent": "drug_check", "results": results, "warnings": warnings}

# ─── DOCTOR CHECK AGENT ───────────────────────────────────────────────────────

async def doctor_check_agent(profile: dict) -> dict:
    """Verify doctors exist in NPI registry and flag network concerns."""
    await asyncio.sleep(0)
    
    if not profile.get("doctors"):
        return {"agent": "doctor_check", "results": [], "warnings": []}
    
    results = []
    for doctor in profile["doctors"]:
        npi_info = verify_doctor_npi(doctor)
        results.append(npi_info)
    
    return {
        "agent": "doctor_check",
        "results": results,
        "warnings": ["Always verify your specific doctor is in-network before enrolling — network directories change frequently."] if results else []
    }

# ─── RISK & GAPS AGENT ────────────────────────────────────────────────────────

async def risk_gaps_agent(profile: dict, plans: list, fpl_pct: float) -> dict:
    """Proactively identify coverage gaps the user didn't think to ask about."""
    await asyncio.sleep(0)
    
    gaps = []
    flags = []
    
    # Mental health parity check
    gaps.append("Verify mental health and substance abuse coverage parity on shortlisted plans.")
    
    # OOP exposure
    if any(p.get("oop_max", 0) > 8000 for p in plans):
        flags.append("⚠️ Some plans have OOP max above $8,700 — consider your risk tolerance for a bad health year.")
    
    # Subsidy cliff warning
    if 390 <= fpl_pct <= 410:
        flags.append("🚨 Subsidy cliff alert: Your income is within 5% of the 400% FPL threshold. A small raise could cost you thousands in lost subsidies.")
    
    # HSA opportunity
    bronze_plans = [p for p in plans if p.get("metal_level") == "Bronze"]
    if bronze_plans:
        flags.append("💡 Bronze HDHP plans may qualify for HSA contributions — tax savings of $1,600–$3,200/year depending on household.")
    
    # Catastrophic plan check
    if profile["age"] < 30:
        flags.append("💡 As someone under 30, you may qualify for a Catastrophic plan — lower premiums, $9,450 deductible.")
    
    return {"agent": "risk_gaps", "gaps": gaps, "flags": flags}

# ─── METAL TIER AGENT ─────────────────────────────────────────────────────────

async def metal_tier_agent(profile: dict, plans: list, subsidy: dict) -> dict:
    """Analyze Bronze/Silver/Gold tradeoffs for this specific user."""
    await asyncio.sleep(0)
    
    monthly_credit = subsidy.get("monthly_credit", 0)
    csr_eligible = subsidy.get("csr_eligible", False)
    
    recommendation = ""
    if csr_eligible:
        recommendation = "Silver is your best metal tier. CSR makes Silver plans act like Gold with lower deductibles — this benefit only applies to Silver."
    elif profile["income"] > 80000:
        recommendation = "Gold makes sense if you expect frequent medical visits. The lower deductible pays off above ~$3,000/year in medical costs."
    else:
        recommendation = "Bronze + HSA is optimal if you're generally healthy. Pay low premiums, invest HSA contributions tax-free."
    
    return {
        "agent": "metal_tier",
        "recommendation": recommendation,
        "csr_eligible": csr_eligible,
        "monthly_credit": monthly_credit,
    }

# ─── MEDICAID AGENT ───────────────────────────────────────────────────────────

async def medicaid_agent(profile: dict, fpl_pct: float) -> dict:
    """Handle users who qualify for Medicaid — skip marketplace entirely."""
    await asyncio.sleep(0)
    return {
        "agent": "medicaid",
        "qualifies": fpl_pct < 138,
        "message": f"At {fpl_pct}% FPL, you likely qualify for Medicaid — free or near-free coverage. Visit your state Medicaid office or healthcare.gov to confirm and enroll.",
        "action": "Visit healthcare.gov → Apply for Coverage → Medicaid/CHIP",
    }
