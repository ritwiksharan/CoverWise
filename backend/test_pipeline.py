"""
Pipeline test — runs every stage and dumps console output at each step.
Run from the backend/ directory:
    PYTHONPATH=. python3 test_pipeline.py

Modes:
  default     → quick coverage check across all ZIP codes, then full pipeline on primary
  --full      → full pipeline (LLM ranking + synthesis prompt) on every ZIP
  --zip 10001 → override primary ZIP for the full pipeline run
"""
import asyncio
import json
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
sys.path.insert(0, os.path.dirname(__file__))

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m";  BOLD = "\033[1m"
CYAN  = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
RED   = "\033[91m"; MAGENTA = "\033[95m"; DIM = "\033[2m"

def hdr(title: str):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

def sub(title: str):
    print(f"\n{BOLD}{YELLOW}▶ {title}{RESET}")
    print(f"{DIM}{'─'*50}{RESET}")

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def kv(k, v):  print(f"  {DIM}{k:<30}{RESET} {v}")

# ── test ZIP codes — federal marketplace states only (not state-based exchanges)
# States excluded (own exchange, not on CMS API): NY, CA, WA, MA, CO, CT, MD, MN, VT, DC, ID
TEST_ZIPS = [
    # (zip,   city/label,          income,  age, hh, utilization,  drugs,                       doctors)
    ("60601", "Chicago IL",          48000,  34,  1, "sometimes",  ["Ozempic", "Lisinopril"],   ["Dr. John Smith"]),
    ("30301", "Atlanta GA",          38000,  29,  1, "sometimes",  ["Adderall"],                []),
    ("33101", "Miami FL",            52000,  47,  3, "chronic",    ["Humira", "Metformin"],     []),
    ("85001", "Phoenix AZ",          44000,  31,  1, "rarely",     [],                          []),
    ("77001", "Houston TX",          61000,  45,  4, "frequently", ["Jardiance", "Lisinopril"], []),
    ("46201", "Indianapolis IN",      58000,  40,  2, "sometimes",  ["Eliquis"],                 []),
    ("27601", "Raleigh NC",          45000,  33,  1, "rarely",     ["Metformin"],               []),
    ("37201", "Nashville TN",        53000,  38,  2, "sometimes",  ["Atorvastatin", "Ozempic"], []),
    ("73102", "Oklahoma City OK",     67000,  50,  1, "frequently", ["Humira"],                  []),
    ("63101", "St. Louis MO",        42000,  28,  1, "rarely",     [],                          []),
]

# Primary profile used for the full pipeline run
def make_profile(zip_code, city, income, age, household, utilization, drugs, doctors):
    return {
        "zip_code":       zip_code,
        "income":         income,
        "age":            age,
        "household_size": household,
        "tobacco_use":    False,
        "utilization":    utilization,
        "drugs":          drugs,
        "doctors":        doctors,
        "is_premium":     True,
        "user_id":        f"test_{zip_code}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK COVERAGE CHECK — location + subsidy + plan count for every ZIP
# ─────────────────────────────────────────────────────────────────────────────
def quick_check(zip_code, city, income, age, household, utilization, drugs, doctors):
    from tools.gov_apis import (
        get_fips_from_zip, _fips_to_state,
        calculate_fpl_percentage, get_eligibility_estimates,
        get_medicaid_threshold, search_plans,
    )
    try:
        fips  = get_fips_from_zip(zip_code)
        state = _fips_to_state(fips) if fips else "?"
        fpl   = calculate_fpl_percentage(income, household, state)

        est   = get_eligibility_estimates(
            income=income, age=age, fips=fips,
            zip_code=zip_code, state=state, tobacco_use=False,
        )
        aptc  = est.get("aptc", 0) or 0

        csr = None
        if 100 <= fpl <= 150:   csr = "94"
        elif 150 <= fpl <= 200: csr = "87"
        elif 200 <= fpl <= 250: csr = "73"

        plans = search_plans(
            zip_code=zip_code, age=age, income=income,
            fips=fips, state=state, tobacco_use=False,
        )
        plan_count = len(plans)
        metals = {}
        for p in plans:
            m = p.get("metal_level","?")
            metals[m] = metals.get(m, 0) + 1
        metal_str = "  ".join(f"{m}×{c}" for m, c in sorted(metals.items()))

        status = GREEN + "✓" + RESET if plan_count > 0 else RED + "✗" + RESET
        print(
            f"  {status} {zip_code}  {city:<20}"
            f"  {state:<4} FPL={fpl:5.1f}%  APTC=${aptc:5.0f}/mo"
            f"  CSR={csr or '—':<4}  Plans={plan_count:3}  [{metal_str}]"
        )
        return fips, state, plans, aptc, fpl, csr
    except Exception as e:
        print(f"  {RED}✗{RESET} {zip_code}  {city:<20}  ERROR: {e}")
        return None, None, [], 0, 0, None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1a — Subsidy detail
# ─────────────────────────────────────────────────────────────────────────────
def test_subsidy(profile, fips, state):
    from tools.gov_apis import calculate_fpl_percentage, get_eligibility_estimates, get_medicaid_threshold
    hdr(f"PHASE 1a — Subsidy  (ZIP {profile['zip_code']}, {state})")

    fpl_pct = calculate_fpl_percentage(profile["income"], profile["household_size"], state)
    ok(f"FPL percentage: {fpl_pct:.1f}%")

    est = get_eligibility_estimates(
        income=profile["income"], age=profile["age"],
        fips=fips, zip_code=profile["zip_code"], state=state,
        tobacco_use=profile["tobacco_use"],
    )
    aptc = est.get("aptc", 0) or 0
    ok(f"Monthly APTC: ${aptc:.2f}  (${aptc*12:,.0f}/yr)")

    medicaid_threshold = get_medicaid_threshold(state)
    is_medicaid = fpl_pct < medicaid_threshold
    kv("Medicaid threshold:", f"{medicaid_threshold}% FPL → eligible: {is_medicaid}")

    csr = None
    if 100 <= fpl_pct <= 150:   csr = "94"
    elif 150 <= fpl_pct <= 200: csr = "87"
    elif 200 <= fpl_pct <= 250: csr = "73"
    ok(f"CSR variant: {csr or 'None'}")

    return {"fpl_percentage": fpl_pct, "monthly_aptc": aptc,
            "csr_variant": csr, "is_medicaid_eligible": is_medicaid}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1b — Plan detail
# ─────────────────────────────────────────────────────────────────────────────
def test_plans(profile, fips, state, monthly_aptc):
    from tools.gov_apis import search_plans
    hdr(f"PHASE 1b — Plans  (ZIP {profile['zip_code']}, {state})")

    plans = search_plans(
        zip_code=profile["zip_code"], age=profile["age"],
        income=profile["income"], fips=fips, state=state,
        tobacco_use=profile["tobacco_use"],
    )
    ok(f"Found {len(plans)} plans total — showing first 8")

    for i, p in enumerate(plans[:8], 1):
        net = max(0, p["premium"] - monthly_aptc)
        print(
            f"  {BOLD}#{i}{RESET}  {p.get('name','')[:44]:<45}"
            f"  {p.get('metal_level','?'):<8} {p.get('type','?'):<5}"
            f"  gross=${p['premium']:6.0f}  net=${net:6.0f}/mo"
            f"  ded=${p['deductible']:,}  oop=${p['oop_max']:,}"
            f"  HSA={'✓' if p.get('hsa_eligible') else '—'}"
        )
    return plans


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1c — Drug Coverage + Real Copays
# ─────────────────────────────────────────────────────────────────────────────
def test_drugs(profile, plans):
    from tools.gov_apis import (
        resolve_drug_rxcui, check_drug_coverage, get_plan_drug_copays,
        get_generic_alternatives,
    )
    from agents.adk_orchestrator import _calc_drug_monthly_cost

    if not profile["drugs"]:
        warn("No drugs in profile — skipping drug phase")
        return {}

    hdr(f"PHASE 1c — Drug Coverage  (ZIP {profile['zip_code']})")
    plan_ids = [p["id"] for p in plans[:5]]

    sub("Drug Name → RxCUI Resolution")
    resolved = []
    for name in profile["drugs"]:
        info = resolve_drug_rxcui(name)
        if info:
            ok(f"{name} → RxCUI {info['rxcui']}  ({info.get('full_name','') or info.get('name','')})")
            resolved.append(info)
        else:
            warn(f"{name} → could not resolve")

    if not resolved:
        warn("No drugs resolved — skipping")
        return {}

    rxcui_list = [d["rxcui"] for d in resolved]

    sub("CMS Drug Coverage  (+ RAG formulary fallback)")
    coverage = check_drug_coverage(rxcui_list, plan_ids, drug_names=profile["drugs"])
    print(f"  Coverage rows returned: {len(coverage)}")
    for row in coverage[:12]:
        status_color = GREEN if row.get("coverage") == "Covered" else (RED if row.get("coverage") == "NotCovered" else DIM)
        src = f"[{row.get('source','cms')}]"
        print(
            f"  {DIM}{row.get('plan_id','')[:20]:<22}{RESET}"
            f"  rxcui={row.get('rxcui',''):<12}"
            f"  {status_color}{row.get('coverage','?'):<18}{RESET}"
            f"  tier={row.get('drug_tier') or '—':<30}"
            f"  PA={row.get('prior_authorization',False)!s:<6}"
            f"  {DIM}{src}{RESET}"
        )

    sub("Real Drug Copays from CMS Plan Benefits Endpoint")
    for p in plans[:5]:
        pid = p["id"]
        copay_data = get_plan_drug_copays(pid)
        if copay_data:
            print(f"\n  {BOLD}{p['name'][:50]}{RESET}  ({p['metal_level']})")
            for tier_key, info in sorted(copay_data.items()):
                copay, display = _calc_drug_monthly_cost(tier_key, copay_data)
                print(
                    f"    {DIM}{tier_key:<28}{RESET}"
                    f"  copay_amount=${info.get('copay_amount',0):<8.0f}"
                    f"  coinsurance={info.get('coinsurance_rate',0)*100:.0f}%"
                    f"  → {GREEN}{display}{RESET}"
                    f"  after_deductible={info.get('after_deductible',False)}"
                )
        else:
            warn(f"{p['name'][:50]} — no copay data from CMS (fallback estimates will be used)")

    sub("Generic Alternatives (openFDA)")
    for name in profile["drugs"]:
        alts = get_generic_alternatives(name)
        generics = [g["generic_name"] for g in alts if g.get("is_generic")]
        if generics:
            ok(f"{name} → generics: {', '.join(generics)}")
        else:
            kv(f"{name}:", "no generics found")

    return coverage


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1d — Doctor Verification
# ─────────────────────────────────────────────────────────────────────────────
def test_doctors(profile, plans, state):
    if not profile["doctors"]:
        warn("No doctors in profile — skipping doctor phase")
        return
    from tools.gov_apis import lookup_npi_registry, get_doctor_quality_score, check_doctor_in_plan_network
    hdr(f"PHASE 1d — Doctor Verification  (ZIP {profile['zip_code']}, {state})")
    plan_ids = [p["id"] for p in plans[:3]]

    for doctor in profile["doctors"]:
        sub(f"Looking up: {doctor}")
        nppes = lookup_npi_registry(doctor, state=state)
        if nppes.get("found"):
            ok(f"Found: {nppes['name']}  NPI={nppes['npi']}")
            kv("Specialty:", nppes.get("specialty",""))
            kv("City/State:", f"{nppes.get('city','')}, {nppes.get('state','')}")

            npi = nppes["npi"]
            if npi:
                quality = get_doctor_quality_score(str(npi))
                if quality.get("found"):
                    ok(f"MIPS Score: {quality.get('mips_score')}/100  Telehealth: {quality.get('telehealth')}")
                else:
                    kv("MIPS:", "not in CMS quality dataset")

                print(f"\n  {DIM}Network status across top 3 plans:{RESET}")
                for pid in plan_ids:
                    net = check_doctor_in_plan_network(pid, str(npi), profile["zip_code"])
                    pname = next((p["name"][:35] for p in plans if p["id"] == pid), pid)
                    in_net = net.get("in_network")
                    if in_net is True:
                        print(f"    {GREEN}✓ In-Network{RESET}   {pname}")
                    elif in_net is False:
                        print(f"    {RED}✗ Out-of-Network{RESET}  {pname}")
                    else:
                        print(f"    {YELLOW}? Unknown{RESET}        {pname}  {DIM}{net.get('note','')[:60]}{RESET}")
        else:
            warn(f"Not found in NPPES: {doctor}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1e — Market Risks
# ─────────────────────────────────────────────────────────────────────────────
def test_market(profile, fips, state):
    from tools.gov_apis import check_sep_eligibility
    hdr(f"PHASE 1e — Market Risks  (ZIP {profile['zip_code']}, {state})")
    sep = check_sep_eligibility()
    kv("Open enrollment:", str(sep.get("in_open_enrollment", False)))
    kv("SEP message:", sep.get("message",""))
    if sep.get("in_open_enrollment"):
        ok(f"Deadline: {sep.get('deadline')} ({sep.get('days_remaining')} days remaining)")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1.5 — LLM Ranking Agent
# ─────────────────────────────────────────────────────────────────────────────
async def test_llm_ranking(data: dict, profile: dict):
    from agents.adk_orchestrator import _rank_plans_with_llm
    hdr(f"PHASE 1.5 — LLM Ranking Agent  (ZIP {profile['zip_code']}, Gemini JSON mode)")

    print(f"  Sending {len(data.get('plans',[]))} plans to Gemini for ranking...")
    ranking = await _rank_plans_with_llm(data, profile)

    if not ranking:
        warn("Ranking agent returned empty (Vertex AI unavailable or failed)")
        return {}

    # Utilization reasoning
    util_rsn = ranking.get("utilization_weight_reasoning", "")
    if util_rsn:
        sub("Utilization Weight Reasoning")
        print(f"  {CYAN}{util_rsn}{RESET}")

    # Top recommendation
    top = ranking.get("top_recommendation", {})
    if top:
        print(f"\n  {BOLD}{GREEN}TOP RECOMMENDATION: {top.get('plan_name','')}{RESET}")
        for line in top.get('rationale','').split('. '):
            if line.strip():
                print(f"  {line.strip()}.")

    # CSR override + explanation
    csr_ov  = ranking.get("csr_override")
    csr_exp = ranking.get("csr_explanation", "")
    if csr_ov:
        print(f"\n  {YELLOW}CSR Override → {csr_ov}{RESET}")
    if csr_exp:
        sub("CSR Reasoning")
        print(f"  {YELLOW}{csr_exp}{RESET}")

    # Scenario trade-off
    trade_off = ranking.get("scenario_trade_off", "")
    if trade_off:
        sub("Scenario Trade-Off (top 2 plans)")
        print(f"  {MAGENTA}{trade_off}{RESET}")

    # Expected Value ranking
    ev = ranking.get("expected_value_ranking", [])
    if ev:
        sub("Expected Value Ranking (LLM-computed)")
        print(f"  {'Rank':<6} {'Plan':<45} {'EV Score':>12}  Key Reason")
        print(f"  {DIM}{'─'*110}{RESET}")
        for r in ev:
            rank_sym = f"{GREEN}#{r.get('rank')}{RESET}" if r.get('rank') == 1 else f"#{r.get('rank')}"
            print(f"  {rank_sym:<6} {r.get('plan_name','')[:44]:<45} "
                  f"${r.get('ev_score',0):>10,.0f}  {DIM}{r.get('key_reason','')[:80]}{RESET}")

    # Per-scenario rankings
    for scenario_key, label in [
        ("healthy_year",  "Healthy Year  (premiums only)"),
        ("clinical_year", "Clinical Year (premiums + drugs)"),
        ("worst_case",    "Worst Case    (premiums + OOP max)"),
    ]:
        ranked = ranking.get("rankings", {}).get(scenario_key, [])
        if ranked:
            sub(f"Scenario: {label}")
            for r in ranked[:5]:
                marker = f"{GREEN}★{RESET}" if r.get("rank") == 1 else " "
                print(f"  {marker} #{r.get('rank')}  {r.get('plan_name','')[:40]:<42}"
                      f"  ${r.get('annual_cost',0):>9,.0f}/yr  {DIM}{r.get('reason','')[:70]}{RESET}")

    # Red flags
    red_flags = ranking.get("red_flags", [])
    if red_flags:
        sub("Red Flags from Ranking Agent")
        for flag in red_flags:
            warn(flag)

    # Raw JSON dump (first 80 lines)
    sub("Raw Ranking JSON (first 80 lines)")
    raw = json.dumps(ranking, indent=2)
    for line in raw.split("\n")[:80]:
        print(f"  {DIM}{line}{RESET}")
    if raw.count("\n") > 80:
        print(f"  {DIM}... ({raw.count(chr(10))-80} more lines){RESET}")

    return ranking


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE — _collect_analysis_data + synthesis prompt length
# ─────────────────────────────────────────────────────────────────────────────
async def test_full_pipeline(profile: dict):
    from agents.adk_orchestrator import _collect_analysis_data, _build_synthesis_prompt

    hdr(f"FULL PIPELINE — _collect_analysis_data()  (ZIP {profile['zip_code']})")
    print(f"  Profile: {profile['age']}yo  income=${profile['income']:,}  "
          f"utilization={profile['utilization']}  drugs={profile['drugs']}")

    data = await _collect_analysis_data(profile)

    sub("Processed Plans (with drug cost models)")
    plans = data.get("plans", [])
    print(f"  {'Plan':<40} {'Net/mo':>8} {'Healthy':>9} {'Clinical':>10} "
          f"{'Worst':>10} {'Drug/yr':>9} {'PA?'}")
    print(f"  {DIM}{'─'*105}{RESET}")
    for p in plans[:10]:
        pa = f"{RED}YES{RESET}" if p.get("pa_warning") else f"{GREEN}no{RESET}"
        print(
            f"  {p.get('name','')[:39]:<40}"
            f"  ${p.get('premium_w_credit',0):>7.0f}"
            f"  ${p.get('scenario_healthy',0):>8,.0f}"
            f"  ${p.get('scenario_clinical',0):>9,.0f}"
            f"  ${p.get('scenario_worst',0):>9,.0f}"
            f"  ${p.get('est_annual_drug_cost',0):>8,.0f}"
            f"  {pa}"
        )

    sub("Drug Detail — real copays per plan (top 3 plans)")
    for p in plans[:3]:
        print(f"\n  {BOLD}{p.get('name','')[:50]}{RESET}")
        for dd in p.get("drug_detail", []):
            cov_color = GREEN if dd.get("coverage") == "Covered" else RED
            print(
                f"    {dd.get('name',''):<18}"
                f"  {cov_color}{dd.get('coverage','?'):<18}{RESET}"
                f"  tier={dd.get('tier','—'):<28}"
                f"  {CYAN}{dd.get('copay_display','—')}{RESET}"
                f"  PA={dd.get('pa',False)!s:<6}"
                f"  src={dd.get('source','?')}"
            )

    sub("Risk Flags")
    for flag in data.get("risk_flags", []):
        print(f"  {flag}")

    prompt = _build_synthesis_prompt(profile, data)
    ok(f"Synthesis prompt built: {len(prompt):,} chars")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",  action="store_true", help="Run full pipeline on every ZIP")
    parser.add_argument("--zip",   default=None,        help="Override primary ZIP for full run")
    parser.add_argument("--quick", action="store_true", help="Coverage check only, no full pipeline")
    args = parser.parse_args()

    # ── Quick coverage check across all ZIPs ─────────────────────────────────
    hdr("COVERAGE CHECK — All ZIP Codes")
    print(f"  {'ZIP':<7} {'City':<20}  {'St':>4}  {'FPL':>8}  {'APTC':>10}  {'CSR':>4}  {'Plans':>7}  Metals")
    print(f"  {DIM}{'─'*90}{RESET}")

    results_by_zip = {}
    for row in TEST_ZIPS:
        zip_code, city, income, age, household, utilization, drugs, doctors = row
        fips, state, plans, aptc, fpl, csr = quick_check(*row)
        results_by_zip[zip_code] = {
            "fips": fips, "state": state, "plans": plans,
            "aptc": aptc, "fpl": fpl, "csr": csr,
            "city": city, "income": income, "age": age,
            "household": household, "utilization": utilization,
            "drugs": drugs, "doctors": doctors,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    hdr("COVERAGE SUMMARY")
    working  = [(z, r) for z, r in results_by_zip.items() if r["plans"]]
    no_plans = [(z, r) for z, r in results_by_zip.items() if not r["plans"]]
    ok(f"{len(working)}/{len(TEST_ZIPS)} ZIP codes returned plan data")
    if no_plans:
        for z, r in no_plans:
            warn(f"No plans: {z} ({r['city']}) — state={r['state']}")

    if args.quick:
        return

    # ── Full pipeline on primary ZIP (or --zip override) ──────────────────────
    primary_zips = [args.zip] if args.zip else [TEST_ZIPS[0][0]]
    if args.full:
        primary_zips = [z for z, r in results_by_zip.items() if r["plans"]]

    for zip_code in primary_zips:
        r = results_by_zip.get(zip_code)
        if not r or not r["plans"]:
            err(f"ZIP {zip_code} has no plan data — skipping full pipeline")
            continue

        profile = make_profile(
            zip_code, r["city"], r["income"], r["age"],
            r["household"], r["utilization"], r["drugs"], r["doctors"]
        )

        hdr(f"FULL PIPELINE — {zip_code} ({r['city']})")

        fips  = r["fips"]
        state = r["state"]

        subsidy = test_subsidy(profile, fips, state)
        plans   = test_plans(profile, fips, state, subsidy["monthly_aptc"])
        test_drugs(profile, plans)
        test_doctors(profile, plans, state)
        test_market(profile, fips, state)

        data    = await test_full_pipeline(profile)
        ranking = await test_llm_ranking(data, profile)

        hdr(f"DONE — {zip_code} ({r['city']})")
        ok(f"Plans collected: {len(data.get('plans',[]))}")
        ok(f"LLM Ranking: {'✓ computed — top pick: ' + ranking.get('top_recommendation',{}).get('plan_name','?') if ranking else '— skipped (no Vertex AI)'}")

        if len(primary_zips) > 1:
            print(f"\n{BOLD}{'─'*60} next ZIP ───{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
