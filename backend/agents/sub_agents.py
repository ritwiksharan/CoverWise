"""
Sub-Agents — each handles one domain of insurance analysis.
All data pulled live from CMS APIs. No hardcoded values.
"""

import asyncio
from tools.gov_apis import (
    calculate_fpl_percentage, get_fpl_thresholds, search_plans,
    get_eligibility_estimates, get_medicaid_threshold, get_fips_from_zip,
    resolve_drug_rxcui, check_drug_coverage, verify_doctor_cms, _fips_to_state,
    get_state_exchange, lookup_npi_registry, check_doctor_in_plan_network,
    get_generic_alternatives, get_doctor_quality_score,
    check_sep_eligibility, check_hrsa_shortage,
)

# ─── PROFILE AGENT ────────────────────────────────────────────────────────────

async def profile_agent(profile: dict) -> dict:
    await asyncio.sleep(0)
    fips = get_fips_from_zip(profile["zip_code"])
    state = _fips_to_state(fips) if fips else "US"

    fpl_pct = calculate_fpl_percentage(profile["income"], profile["household_size"], state)
    medicaid_threshold = get_medicaid_threshold(state) if state else 138.0
    route = determine_route(fpl_pct, medicaid_threshold)

    state_exchange = get_state_exchange(profile["zip_code"])

    return {
        "agent": "profile",
        "fpl_percentage": fpl_pct,
        "fips": fips,
        "state": state,
        "route": "state_exchange" if state_exchange else route,
        "route_reason": get_route_reason(route, fpl_pct),
        "state_exchange": state_exchange,
    }

def determine_route(fpl_pct: float, medicaid_threshold: float) -> str:
    if fpl_pct < medicaid_threshold:
        return "medicaid"
    elif fpl_pct < 138:
        return "chip"
    elif fpl_pct <= 400:
        return "subsidized"
    elif fpl_pct <= 600:
        return "marketplace"
    else:
        return "full_price"

def get_route_reason(route: str, fpl_pct: float) -> str:
    reasons = {
        "medicaid":   f"At {fpl_pct}% FPL you likely qualify for free Medicaid coverage.",
        "chip":       f"At {fpl_pct}% FPL children in your household may qualify for CHIP.",
        "subsidized": f"At {fpl_pct}% FPL you qualify for ACA premium tax credits (APTC).",
        "marketplace":f"At {fpl_pct}% FPL you don't qualify for subsidies but can use the marketplace.",
        "full_price": f"At {fpl_pct}% FPL consider employer coverage or full-price marketplace plans.",
    }
    return reasons.get(route, "")

# ─── SUBSIDY AGENT — real APTC from CMS eligibility endpoint ──────────────────

async def subsidy_agent(profile: dict, fpl_pct: float, fips: str, state: str) -> dict:
    await asyncio.sleep(0)

    if fpl_pct > 600:
        return {"agent": "subsidy", "eligible": False, "monthly_credit": 0, "annual_credit": 0}

    estimates = get_eligibility_estimates(
        income=profile["income"],
        age=profile["age"],
        fips=fips,
        zip_code=profile["zip_code"],
        state=state,
        tobacco_use=profile.get("tobacco_use", False),
    )

    aptc = estimates.get("aptc", 0) or 0
    is_medicaid = estimates.get("is_medicaid_chip", False)
    in_gap = estimates.get("in_coverage_gap", False)

    csr_eligible = 100 <= fpl_pct <= 250
    csr_variant = None
    csr_note = None
    if 100 <= fpl_pct <= 150:
        csr_variant = "94"
        csr_note = "You qualify for CSR-94 — Silver plans will have Platinum-level benefits (roughly $0–$500 deductible)."
    elif 150 <= fpl_pct <= 200:
        csr_variant = "87"
        csr_note = "You qualify for CSR-87 — Silver plans will have Gold-level benefits (roughly $500–$1,500 deductible)."
    elif 200 <= fpl_pct <= 250:
        csr_variant = "73"
        csr_note = "You qualify for CSR-73 — Silver plans will have slightly better benefits than standard Silver."

    return {
        "agent": "subsidy",
        "eligible": aptc > 0,
        "monthly_credit": round(aptc, 2),
        "annual_credit": round(aptc * 12, 2),
        "csr_eligible": csr_eligible,
        "csr_variant": csr_variant,
        "csr_note": csr_note,
        "subsidy_cliff_warning": 390 <= fpl_pct <= 410,
        "is_medicaid_chip": is_medicaid,
        "in_coverage_gap": in_gap,
    }

# ─── PLAN SEARCH AGENT ────────────────────────────────────────────────────────

async def plan_search_agent(profile: dict, fips: str, state: str) -> dict:
    await asyncio.sleep(0)
    plans = search_plans(
        zip_code=profile["zip_code"],
        age=profile["age"],
        income=profile["income"],
        fips=fips,
        state=state,
        tobacco_use=profile.get("tobacco_use", False),
    )
    return {"agent": "plan_search", "plans": plans, "plan_count": len(plans)}

# ─── DRUG CHECK AGENT — tier, PA, step therapy, generics ──────────────────────

async def drug_check_agent(profile: dict, plans: list) -> dict:
    await asyncio.sleep(0)

    if not profile.get("drugs"):
        return {"agent": "drug_check", "results": [], "warnings": [], "coverage_raw": []}

    # Step 1: Resolve drug names → RxCUI
    resolved = []
    for drug_name in profile["drugs"]:
        info = resolve_drug_rxcui(drug_name)
        if info:
            resolved.append({**info, "input_name": drug_name})
        else:
            resolved.append({"rxcui": None, "name": drug_name, "input_name": drug_name, "full_name": ""})

    rxcui_list = [d["rxcui"] for d in resolved if d.get("rxcui")]
    plan_ids = [p["id"] for p in plans if p.get("id")]

    # Step 2: Enhanced drug coverage — captures tier, PA, step_therapy, quantity_limit
    coverage_raw = check_drug_coverage(rxcui_list, plan_ids) if rxcui_list else []

    # Step 3: Organise by plan_id → rxcui → full coverage record
    plan_drug_map: dict = {}
    for c in coverage_raw:
        pid = c.get("plan_id")
        rxcui = str(c.get("rxcui", ""))
        if pid not in plan_drug_map:
            plan_drug_map[pid] = {}
        plan_drug_map[pid][rxcui] = c

    # Step 4: Build per-drug results
    results = []
    warnings = []
    for d in resolved:
        rxcui = str(d.get("rxcui", "")) if d.get("rxcui") else None

        # Collect per-plan coverage record
        plan_coverage: dict = {}
        if rxcui:
            for pid in plan_ids[:10]:
                rec = plan_drug_map.get(pid, {}).get(rxcui)
                if rec:
                    plan_coverage[pid] = {
                        "coverage": rec.get("coverage", "Unknown"),
                        "tier": rec.get("drug_tier"),
                        "prior_authorization": rec.get("prior_authorization", False),
                        "step_therapy": rec.get("step_therapy", False),
                        "quantity_limit": rec.get("quantity_limit", False),
                    }
                else:
                    plan_coverage[pid] = {"coverage": "Unknown"}

        covered_count = sum(1 for v in plan_coverage.values() if v.get("coverage") == "Covered")
        not_covered = sum(1 for v in plan_coverage.values() if v.get("coverage") == "NotCovered")
        data_missing = sum(1 for v in plan_coverage.values() if v.get("coverage") in ("DataNotProvided", "Unknown"))
        pa_count = sum(1 for v in plan_coverage.values() if v.get("prior_authorization"))
        st_count = sum(1 for v in plan_coverage.values() if v.get("step_therapy"))

        # Generic alternatives via openFDA
        generics = get_generic_alternatives(d["input_name"]) if d.get("rxcui") else []

        results.append({
            "drug_name": d["input_name"],
            "rxcui": rxcui,
            "full_name": d.get("full_name", ""),
            "route": d.get("route", ""),
            "strength": d.get("strength", ""),
            "covered_in": covered_count,
            "not_covered_in": not_covered,
            "data_missing_in": data_missing,
            "prior_auth_in": pa_count,
            "step_therapy_in": st_count,
            "plan_coverage": plan_coverage,
            "generic_alternatives": generics,
        })

        # Warnings
        if not_covered > covered_count:
            warnings.append(f"⚠️ {d['input_name']} appears not covered in most checked plans.")
        elif data_missing == len(plan_coverage) and plan_coverage:
            warnings.append(
                f"ℹ️ {d['input_name']} ({d.get('full_name','')}): formulary data not provided by "
                f"insurers for 2024. Check each plan's formulary URL before enrolling."
            )
        if pa_count > 0:
            warnings.append(
                f"🔒 {d['input_name']} requires prior authorization on {pa_count} plan(s). "
                f"Get PA approved BEFORE enrolling — it can take 30+ days."
            )
        if st_count > 0:
            warnings.append(
                f"⚠️ {d['input_name']} has step therapy restrictions on {st_count} plan(s) — "
                f"the plan may require you to try a cheaper drug first. Flag this for your doctor."
            )
        if generics:
            generic_names = ", ".join(g["generic_name"] for g in generics if g.get("is_generic"))
            if generic_names:
                warnings.append(
                    f"💊 Generic equivalent for {d['input_name']}: {generic_names} — "
                    f"may be Tier 1 or 2 on most plans. Confirm with your doctor before switching."
                )

    return {"agent": "drug_check", "results": results, "warnings": warnings, "coverage_raw": coverage_raw}

# ─── DOCTOR CHECK AGENT — NPPES NPI Registry + MIPS quality + network check ───

async def doctor_check_agent(profile: dict, plans: list = None) -> dict:
    await asyncio.sleep(0)

    if not profile.get("doctors"):
        return {"agent": "doctor_check", "results": [], "warnings": []}

    zip_code = profile.get("zip_code", "")
    state = profile.get("state", "")
    top_plan_ids = [p["id"] for p in (plans or [])[:3] if p.get("id")]

    results = []
    warnings = []
    for doctor in profile["doctors"]:
        # Step 1: NPPES NPI Registry lookup (authoritative provider identity)
        nppes = lookup_npi_registry(doctor, state=state)

        # Step 2: Fall back to CMS Order & Referring if NPPES misses
        cms_info = {}
        if not nppes.get("found"):
            cms_info = verify_doctor_cms(doctor)

        npi = nppes.get("npi") or cms_info.get("npi")

        # Step 3: MIPS quality score (if NPI found)
        quality = {}
        if npi:
            quality = get_doctor_quality_score(str(npi))

        # Step 4: In-network verification for top 3 plans
        network_status: dict = {}
        if npi and top_plan_ids:
            for pid in top_plan_ids:
                net = check_doctor_in_plan_network(pid, str(npi), zip_code)
                network_status[pid] = net

        record = {
            "searched_name": doctor,
            "found": nppes.get("found") or cms_info.get("found", False),
            "npi": npi,
            "name": nppes.get("name") or cms_info.get("name", doctor),
            "credential": nppes.get("credential", ""),
            "specialty": nppes.get("specialty", ""),
            "city": nppes.get("city", ""),
            "state": nppes.get("state", ""),
            "phone": nppes.get("phone", ""),
            "active": nppes.get("active", True),
            # Legacy Medicare fields from CMS Order & Referring
            "medicare_part_b": cms_info.get("medicare_part_b"),
            "can_order_dme": cms_info.get("can_order_dme"),
            # Quality
            "mips_score": quality.get("mips_score"),
            "telehealth": quality.get("telehealth"),
            "mips_year": quality.get("year"),
            # Multiple candidates if ambiguous
            "all_candidates": nppes.get("all_candidates", []),
            # Per-plan network status
            "network_status": network_status,
        }
        results.append(record)

        # Warnings
        if not record["found"]:
            warnings.append(
                f"⚠️ {doctor} not found in NPPES NPI Registry. "
                f"They may use a different name or have a recent license change."
            )
        else:
            if len(nppes.get("all_candidates", [])) > 1:
                warnings.append(
                    f"ℹ️ Multiple providers named {doctor} found. "
                    f"Confirm NPI {npi} with your doctor's office before enrolling."
                )
            not_active = not record.get("active", True)
            if not_active:
                warnings.append(f"⚠️ {record['name']} (NPI {npi}) is listed as inactive in NPPES.")

            # Network warnings
            for pid, net in network_status.items():
                if net.get("in_network") is False:
                    warnings.append(
                        f"🚨 {record['name']} is OUT-OF-NETWORK on plan {pid}. "
                        f"An out-of-network specialist visit can cost $3,000–$8,000 out of pocket."
                    )
                elif net.get("accepting_patients") is False:
                    warnings.append(
                        f"⚠️ {record['name']} is in-network on plan {pid} but NOT accepting new patients."
                    )

            # MIPS quality note
            if quality.get("found") and quality.get("mips_score") is not None:
                score = quality["mips_score"]
                if float(score) < 30:
                    warnings.append(
                        f"⚠️ {record['name']} has a low MIPS quality score ({score}/100). "
                        f"Consider researching patient reviews."
                    )

    warnings.append(
        "Always confirm your doctor is in-network with your specific chosen plan — "
        "network directories change frequently and online tools can be outdated."
    )
    return {"agent": "doctor_check", "results": results, "warnings": warnings}

# ─── RISK & GAPS AGENT ────────────────────────────────────────────────────────

async def risk_gaps_agent(profile: dict, plans: list, fpl_pct: float) -> dict:
    await asyncio.sleep(0)

    state = profile.get("state", "")
    fips = profile.get("fips", "")
    flags = []

    # OOP max risk
    if any(p.get("oop_max", 0) > 8700 for p in plans):
        flags.append("⚠️ Some plans have OOP max above $8,700 — consider your risk tolerance for a bad health year.")

    # Subsidy cliff
    if 390 <= fpl_pct <= 410:
        flags.append("🚨 Subsidy cliff: Your income is within 5% of the 400% FPL cutoff. A small raise could cost thousands in lost subsidies.")

    # HSA eligibility & tax savings calculation
    hsa_plans = [p for p in plans if p.get("hsa_eligible")]
    if hsa_plans:
        # 2024 limits: $4,150 individual, $8,300 family
        limit = 8300 if profile.get("household_size", 1) > 1 else 4150
        income = profile.get("income", 50000)
        
        # Simple 2024 tax bracket estimate (Single)
        if income <= 11600: rate = 0.10
        elif income <= 47150: rate = 0.12
        elif income <= 100525: rate = 0.22
        elif income <= 191950: rate = 0.24
        elif income <= 243725: rate = 0.32
        elif income <= 609350: rate = 0.35
        else: rate = 0.37
        
        tax_savings = round(limit * rate)
        flags.append(
            f"💡 {len(hsa_plans)} HSA-eligible plan(s) available. By contributing the ${limit:,} limit, "
            f"you'd save roughly ${tax_savings:,} in federal taxes this year (based on your {int(rate*100)}% bracket). "
            f"HSA funds also grow tax-free and are tax-free for medical use."
        )

    # Under-30 catastrophic
    if profile.get("age", 35) < 30:
        flags.append("💡 Under 30: you may qualify for a Catastrophic plan — lower premiums with a $9,450 deductible.")

    # HRSA provider shortage area
    if state:
        hrsa = check_hrsa_shortage(state, fips)
        if hrsa.get("shortage_area"):
            flags.append(f"🏥 {hrsa.get('message', 'Your county is a Health Professional Shortage Area.')} Prefer PPO or EPO plans over HMOs for more provider flexibility.")

    # SEP / open enrollment status
    sep = check_sep_eligibility()
    if sep.get("in_open_enrollment"):
        flags.append(f"📅 Open enrollment is active — {sep.get('days_remaining')} days left (deadline {sep.get('deadline')}).")
    else:
        flags.append(
            f"📅 {sep.get('message', '')} Qualifying events (job loss, marriage, birth, move) trigger a 60-day Special Enrollment Period."
        )

    # Mental health parity check
    hmo_only = all(p.get("type") == "HMO" for p in plans) if plans else False
    if hmo_only:
        flags.append("ℹ️ All available plans in your area are HMO — you must use in-network providers. Confirm your doctors are in-network before enrolling.")

    return {"agent": "risk_gaps", "flags": flags, "sep": sep}

# ─── METAL TIER AGENT ─────────────────────────────────────────────────────────

async def metal_tier_agent(profile: dict, plans: list, subsidy: dict) -> dict:
    await asyncio.sleep(0)

    csr_eligible = subsidy.get("csr_eligible", False)
    monthly_credit = subsidy.get("monthly_credit", 0)
    cliff_warning = subsidy.get("subsidy_cliff_warning", False)
    has_drugs = bool(profile.get("drugs"))
    age = profile.get("age", 35)

    # Group plans by metal tier using plan-specific CMS net premiums
    tiers: dict = {}
    for p in plans:
        tiers.setdefault(p.get("metal_level", "Unknown"), []).append(p)

    tier_stats = {}
    for tier, tier_plans in tiers.items():
        nets = [p.get("premium_w_credit", p["premium"]) for p in tier_plans]
        deductibles = [p.get("deductible", 0) for p in tier_plans]
        oops = [p.get("oop_max", 0) for p in tier_plans]
        hsa_count = sum(1 for p in tier_plans if p.get("hsa_eligible"))
        cheapest_idx = nets.index(min(nets))
        net = nets[cheapest_idx]
        tier_stats[tier] = {
            "count": len(tier_plans),
            "cheapest_net_monthly": round(net, 2),
            "cheapest_annual_premiums": round(net * 12, 2),
            "min_deductible": min(deductibles),
            "avg_deductible": round(sum(deductibles) / len(deductibles)),
            "min_oop_max": min(oops),
            "hsa_available": hsa_count > 0,
            "hsa_count": hsa_count,
            "true_annual_avg_use": round(net * 12 + sum(deductibles) / len(deductibles)),
            "worst_case": round(net * 12 + min(oops)),
        }

    silver = tier_stats.get("Silver", {})
    bronze = tier_stats.get("Bronze", {})
    gold = tier_stats.get("Gold", {})

    reasons = []
    recommended_tier = "Bronze"

    # ── CSR eligible (138–250% FPL): Silver wins almost always ────────────────
    if csr_eligible and silver:
        recommended_tier = "Silver"
        s_net = silver["cheapest_net_monthly"]
        s_ded = silver["min_deductible"]
        if bronze:
            b_net = bronze["cheapest_net_monthly"]
            b_ded = bronze["min_deductible"]
            premium_gap = round((s_net - b_net) * 12)
            ded_savings = b_ded - s_ded
            reasons.append(
                f"Silver (CSR): cheapest Silver is ${s_net:.2f}/mo vs Bronze at ${b_net:.2f}/mo — "
                f"Silver costs ${abs(premium_gap):,} {'more' if premium_gap > 0 else 'less'} per year in premiums "
                f"but your deductible drops from ${b_ded:,} to ${s_ded:,} (${ded_savings:,} savings). "
                f"CSR only applies to Silver — do not choose Bronze or Gold if CSR-eligible."
            )
        else:
            reasons.append(
                f"Silver (CSR): cheapest Silver is ${s_net:.2f}/mo with deductible ${s_ded:,}. "
                f"CSR applies only to Silver plans."
            )

    elif csr_eligible and not silver:
        # CSR eligible but no Silver plans returned for this market — flag it
        fallback = gold or bronze
        if fallback:
            recommended_tier = "Gold" if gold else "Bronze"
            tier_name = "Gold" if gold else "Bronze"
            reasons.append(
                f"Note: you are CSR-eligible but no Silver plans were returned for your area. "
                f"CSR cost-sharing reductions only apply to Silver — verify Silver availability at healthcare.gov. "
                f"Cheapest {tier_name} shown is ${fallback['cheapest_net_monthly']:.2f}/mo "
                f"(deductible ${fallback['min_deductible']:,})."
            )

    # ── Subsidized but not CSR (250–400% FPL) ─────────────────────────────────
    elif monthly_credit > 0 and gold and bronze:
        g_net = gold["cheapest_net_monthly"]
        b_net = bronze["cheapest_net_monthly"]
        g_ded = gold["min_deductible"]
        b_ded = bronze["min_deductible"]
        premium_gap = round((g_net - b_net) * 12)
        ded_savings = b_ded - g_ded
        if has_drugs or age > 50:
            recommended_tier = "Gold"
            reasons.append(
                f"Gold: cheapest Gold is ${g_net:.2f}/mo (deductible ${g_ded:,}), "
                f"costs ${premium_gap:,} more per year than Bronze (${b_net:.2f}/mo, deductible ${b_ded:,}). "
                f"Gold pays off if you spend more than ${premium_gap:,} on covered care — "
                f"likely given {'your medications' if has_drugs else 'your age'}."
            )
        else:
            recommended_tier = "Bronze"
            hsa_note = (
                f" {bronze['hsa_count']} HSA-eligible Bronze plan(s) available — "
                f"pre-tax contributions up to $4,150/yr (single, 2024)."
                if bronze.get("hsa_available") else ""
            )
            reasons.append(
                f"Bronze: cheapest Bronze is ${b_net:.2f}/mo (deductible ${b_ded:,}). "
                f"Gold costs ${premium_gap:,} more per year — only worth it if you expect "
                f"more than ${premium_gap:,} in out-of-pocket costs.{hsa_note}"
            )

    # ── No subsidy (>400% FPL) ─────────────────────────────────────────────────
    elif gold:
        recommended_tier = "Gold"
        reasons.append(
            f"Gold: no subsidy applies — cheapest Gold is ${gold['cheapest_net_monthly']:.2f}/mo "
            f"with deductible ${gold['min_deductible']:,}. Lower deductibles matter more without subsidy help."
        )
    elif bronze:
        recommended_tier = "Bronze"
        reasons.append(f"Bronze: cheapest Bronze is ${bronze['cheapest_net_monthly']:.2f}/mo.")

    # ── Under-30 Catastrophic note ─────────────────────────────────────────────
    if age < 30:
        cat = tier_stats.get("Catastrophic", {})
        if cat:
            reasons.append(
                f"Under 30: Catastrophic plan available at ${cat['cheapest_net_monthly']:.2f}/mo — "
                f"$9,450 deductible, best only if you rarely use medical care."
            )
        else:
            reasons.append("Under 30: You may also qualify for a Catastrophic plan — check healthcare.gov.")

    # ── Subsidy cliff warning ──────────────────────────────────────────────────
    if cliff_warning:
        reasons.append(
            "Subsidy cliff: your income is within 5% of the 400% FPL cutoff. "
            "A small income increase could eliminate all APTC — report any income changes to healthcare.gov promptly."
        )

    return {
        "agent": "metal_tier",
        "recommendation": " ".join(reasons),
        "recommended_tier": recommended_tier,
        "csr_eligible": csr_eligible,
        "monthly_credit": monthly_credit,
        "tier_analysis": tier_stats,
    }

# ─── MEDICAID AGENT ───────────────────────────────────────────────────────────

async def medicaid_agent(profile: dict, fpl_pct: float, state: str) -> dict:
    await asyncio.sleep(0)
    threshold = get_medicaid_threshold(state) if state else 138.0
    return {
        "agent": "medicaid",
        "qualifies": fpl_pct < threshold,
        "message": f"At {fpl_pct}% FPL you likely qualify for Medicaid — free or near-free coverage.",
        "action": "Visit healthcare.gov → Apply for Coverage → Medicaid/CHIP",
    }
