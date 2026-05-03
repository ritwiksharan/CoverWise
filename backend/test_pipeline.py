"""
Pipeline test — runs every stage and dumps console output at each step.
Run from the backend/ directory:
    PYTHONPATH=. python3 test_pipeline.py
"""
import asyncio
import json
import os
import sys
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

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def kv(k, v):  print(f"  {DIM}{k:<30}{RESET} {v}")

# ── test profile ──────────────────────────────────────────────────────────────
PROFILE = {
    "zip_code":      "60601",   # Chicago, IL
    "state":         "IL",
    "income":        48000,
    "age":           34,
    "household_size": 1,
    "tobacco_use":   False,
    "utilization":   "sometimes",
    "drugs":         ["Ozempic", "Lisinopril"],
    "doctors":       ["Dr. John Smith"],
    "is_premium":    True,
    "user_id":       "test_user",
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Location
# ─────────────────────────────────────────────────────────────────────────────
def test_location():
    from tools.gov_apis import get_fips_from_zip, _fips_to_state
    hdr("PHASE 0 — Location (ZIP → FIPS → State)")
    fips  = get_fips_from_zip(PROFILE["zip_code"])
    state = _fips_to_state(fips) if fips else "?"
    ok(f"ZIP {PROFILE['zip_code']} → FIPS {fips} → State {state}")
    return fips, state

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1a — Subsidy
# ─────────────────────────────────────────────────────────────────────────────
def test_subsidy(fips, state):
    from tools.gov_apis import calculate_fpl_percentage, get_eligibility_estimates, get_medicaid_threshold
    hdr("PHASE 1a — Subsidy (FPL / APTC / CSR)")

    fpl_pct = calculate_fpl_percentage(PROFILE["income"], PROFILE["household_size"], state)
    ok(f"FPL percentage: {fpl_pct:.1f}%")

    est = get_eligibility_estimates(
        income=PROFILE["income"], age=PROFILE["age"],
        fips=fips, zip_code=PROFILE["zip_code"], state=state,
        tobacco_use=PROFILE["tobacco_use"],
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
# PHASE 1b — Plan Search
# ─────────────────────────────────────────────────────────────────────────────
def test_plans(fips, state, monthly_aptc):
    from tools.gov_apis import search_plans
    hdr("PHASE 1b — Plan Search (CMS Marketplace)")

    plans = search_plans(
        zip_code=PROFILE["zip_code"], age=PROFILE["age"],
        income=PROFILE["income"], fips=fips, state=state,
        tobacco_use=PROFILE["tobacco_use"],
    )
    ok(f"Found {len(plans)} plans total — showing first 5")

    for i, p in enumerate(plans[:5], 1):
        net = max(0, p["premium"] - monthly_aptc)
        sub(f"Plan {i}: {p['name']}")
        kv("Plan ID:",      p["id"])
        kv("Metal / Type:", f"{p['metal_level']} / {p['type']}")
        kv("Gross premium:", f"${p['premium']:.2f}/mo")
        kv("After APTC:",    f"${net:.2f}/mo  (${net*12:,.0f}/yr)")
        kv("Deductible:",    f"${p['deductible']:,}")
        kv("OOP Max:",       f"${p['oop_max']:,}")
        kv("HSA eligible:",  str(p.get("hsa_eligible", False)))
        kv("Issuer:",        p.get("issuer", ""))

    return plans

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1c — Drug Coverage + Real Copays
# ─────────────────────────────────────────────────────────────────────────────
def test_drugs(plans):
    from tools.gov_apis import (
        resolve_drug_rxcui, check_drug_coverage, get_plan_drug_copays,
        get_generic_alternatives,
    )
    from agents.adk_orchestrator import _calc_drug_monthly_cost

    hdr("PHASE 1c — Drug Coverage + Real Plan Copays")
    plan_ids = [p["id"] for p in plans[:5]]

    # Resolve RxCUIs
    sub("Drug Name → RxCUI Resolution")
    resolved = []
    for name in PROFILE["drugs"]:
        info = resolve_drug_rxcui(name)
        if info:
            ok(f"{name} → RxCUI {info['rxcui']}  ({info.get('full_name','') or info.get('name','')})")
            resolved.append(info)
        else:
            warn(f"{name} → could not resolve")

    if not resolved:
        warn("No drugs resolved — skipping coverage check")
        return {}

    rxcui_list = [d["rxcui"] for d in resolved]

    # Coverage per plan
    sub("CMS Drug Coverage API  (+ RAG formulary fallback)")
    coverage = check_drug_coverage(rxcui_list, plan_ids, drug_names=PROFILE["drugs"])
    print(f"  Raw coverage rows returned: {len(coverage)}")
    for row in coverage[:10]:
        status_color = GREEN if row.get("coverage") == "Covered" else (RED if row.get("coverage") == "NotCovered" else DIM)
        src = f"  [{row.get('source','cms')}]" if row.get("source") else ""
        print(
            f"  {DIM}{row.get('plan_id','')[:20]:<22}{RESET}"
            f"  rxcui={row.get('rxcui',''):<12}"
            f"  {status_color}{row.get('coverage','?'):<18}{RESET}"
            f"  tier={row.get('drug_tier') or '—':<30}"
            f"  PA={row.get('prior_authorization',False)!s:<6}"
            f"  {DIM}{src}{RESET}"
        )

    # Real copays from CMS plan benefits endpoint
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
            warn(f"  {p['name'][:50]} — no drug benefit data from CMS (will use fallback estimates)")

    # Generics
    sub("Generic Alternatives (openFDA)")
    for name in PROFILE["drugs"]:
        alts = get_generic_alternatives(name)
        generics = [g["generic_name"] for g in alts if g.get("is_generic")]
        if generics:
            ok(f"{name} → generics: {', '.join(generics)}")
        else:
            kv(f"{name}:", "no generic alternatives found")

    return coverage

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1d — Doctor Verification
# ─────────────────────────────────────────────────────────────────────────────
def test_doctors(plans, state):
    from tools.gov_apis import lookup_npi_registry, get_doctor_quality_score, check_doctor_in_plan_network
    hdr("PHASE 1d — Doctor Verification (NPPES + MIPS + Network)")
    plan_ids = [p["id"] for p in plans[:3]]

    for doctor in PROFILE["doctors"]:
        sub(f"Looking up: {doctor}")
        nppes = lookup_npi_registry(doctor, state=state)
        if nppes.get("found"):
            ok(f"Found: {nppes['name']}  NPI={nppes['npi']}")
            kv("Specialty:", nppes.get("specialty",""))
            kv("City/State:", f"{nppes.get('city','')}, {nppes.get('state','')}")
            kv("Phone:", nppes.get("phone",""))
            kv("Credential:", nppes.get("credential",""))
            kv("Active:", str(nppes.get("active",True)))

            npi = nppes["npi"]
            if npi:
                quality = get_doctor_quality_score(str(npi))
                if quality.get("found"):
                    ok(f"MIPS Score: {quality.get('mips_score')}/100  Telehealth: {quality.get('telehealth')}")
                else:
                    kv("MIPS:", "not in CMS quality dataset")

                print(f"\n  {DIM}Network status across top 3 plans:{RESET}")
                for pid in plan_ids:
                    net = check_doctor_in_plan_network(pid, str(npi), PROFILE["zip_code"])
                    pname = next((p["name"][:35] for p in plans if p["id"] == pid), pid)
                    in_net = net.get("in_network")
                    if in_net is True:
                        print(f"    {GREEN}✓ In-Network{RESET}   {pname}")
                    elif in_net is False:
                        print(f"    {RED}✗ Out-of-Network{RESET}  {pname}")
                    else:
                        note = net.get("note","")[:80]
                        print(f"    {YELLOW}? Unknown{RESET}        {pname}  {DIM}{note}{RESET}")
        else:
            warn(f"Not found in NPPES: {doctor}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1e — Market Risks
# ─────────────────────────────────────────────────────────────────────────────
def test_market(fips, state):
    from tools.gov_apis import check_hrsa_shortage, check_sep_eligibility
    hdr("PHASE 1e — Market Risks (HRSA + SEP)")
    hrsa = check_hrsa_shortage(state, fips)
    sep  = check_sep_eligibility()
    kv("HRSA shortage area:", str(hrsa.get("shortage_area", False)))
    if hrsa.get("message"):
        kv("HRSA message:", hrsa["message"])
    kv("Open enrollment:", str(sep.get("in_open_enrollment", False)))
    kv("SEP message:", sep.get("message",""))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1.5 — LLM Ranking Agent
# ─────────────────────────────────────────────────────────────────────────────
async def test_llm_ranking(data: dict):
    from agents.adk_orchestrator import _rank_plans_with_llm
    hdr("PHASE 1.5 — LLM Ranking Agent  (Gemini JSON mode)")

    print(f"  Sending {len(data.get('plans',[]))} plans to Gemini for ranking...")
    ranking = await _rank_plans_with_llm(data)

    if not ranking:
        warn("Ranking agent returned empty (Vertex AI unavailable or failed)")
        return {}

    # Top recommendation
    top = ranking.get("top_recommendation", {})
    if top:
        print(f"\n  {BOLD}{GREEN}TOP RECOMMENDATION: {top.get('plan_name','')}{RESET}")
        print(f"  {top.get('rationale','')}")

    # CSR override
    csr_ov = ranking.get("csr_override")
    if csr_ov:
        print(f"\n  {YELLOW}CSR Override → {csr_ov}{RESET}")

    # Expected Value ranking
    ev = ranking.get("expected_value_ranking", [])
    if ev:
        sub("Expected Value Ranking (LLM-computed)")
        print(f"  {'Rank':<6} {'Plan':<45} {'EV Score':>12}  Key Reason")
        print(f"  {DIM}{'─'*100}{RESET}")
        for r in ev:
            rank_sym = f"{GREEN}#{r.get('rank')}{RESET}" if r.get('rank') == 1 else f"#{r.get('rank')}"
            print(f"  {rank_sym:<6} {r.get('plan_name','')[:44]:<45} "
                  f"${r.get('ev_score',0):>10,.0f}  {DIM}{r.get('key_reason','')}{RESET}")

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
                      f"  ${r.get('annual_cost',0):>9,.0f}/yr  {DIM}{r.get('reason','')}{RESET}")

    # Red flags
    red_flags = ranking.get("red_flags", [])
    if red_flags:
        sub("Red Flags from Ranking Agent")
        for flag in red_flags:
            warn(flag)

    # Raw JSON dump
    sub("Raw Ranking JSON (truncated)")
    raw = json.dumps(ranking, indent=2)
    for line in raw.split("\n")[:60]:
        print(f"  {DIM}{line}{RESET}")
    if raw.count("\n") > 60:
        print(f"  {DIM}... ({raw.count(chr(10))-60} more lines){RESET}")

    return ranking

# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE — using _collect_analysis_data
# ─────────────────────────────────────────────────────────────────────────────
async def test_full_pipeline():
    from agents.adk_orchestrator import _collect_analysis_data, _build_synthesis_prompt

    hdr("FULL PIPELINE — _collect_analysis_data()")
    print(f"  Profile: {PROFILE['age']}yo, ZIP {PROFILE['zip_code']}, "
          f"income ${PROFILE['income']:,}, drugs: {PROFILE['drugs']}")

    data = await _collect_analysis_data(PROFILE)

    sub("Processed Plans (with drug cost models)")
    plans = data.get("plans", [])
    print(f"  {'Plan':<40} {'Net/mo':>8} {'Annual':>9} {'Clinical':>10} "
          f"{'Worst':>10} {'DrugCost':>10} {'PA?'}")
    print(f"  {DIM}{'─'*105}{RESET}")
    for p in plans[:8]:
        pa = f"{RED}YES{RESET}" if p.get("pa_warning") else f"{GREEN}no{RESET}"
        print(
            f"  {p.get('name','')[:39]:<40}"
            f"  ${p.get('premium_w_credit',0):>7.0f}"
            f"  ${p.get('scenario_healthy',0):>8,.0f}"
            f"  ${p.get('scenario_clinical',0):>9,.0f}"
            f"  ${p.get('scenario_worst',0):>9,.0f}"
            f"  ${p.get('est_annual_drug_cost',0):>9,.0f}"
            f"  {pa}"
        )

    sub("Drug Detail — per-plan real copays")
    for p in plans[:3]:
        print(f"\n  {BOLD}{p.get('name','')[:50]}{RESET}")
        for dd in p.get("drug_detail", []):
            cov_color = GREEN if dd.get("coverage") == "Covered" else RED
            print(
                f"    {dd.get('name',''):<15}"
                f"  {cov_color}{dd.get('coverage','?'):<18}{RESET}"
                f"  tier={dd.get('tier','—'):<30}"
                f"  {CYAN}{dd.get('copay_display','')}{RESET}"
                f"  PA={dd.get('pa',False)!s:<6}"
                f"  src={dd.get('source','?')}"
            )

    sub("Risk Flags")
    for flag in data.get("risk_flags", []):
        print(f"  {flag}")

    return data

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    print(f"\n{BOLD}CoverWise Pipeline Test — Profile: ZIP={PROFILE['zip_code']}, "
          f"Age={PROFILE['age']}, Income=${PROFILE['income']:,}{RESET}")
    print(f"Drugs: {PROFILE['drugs']}  |  Doctors: {PROFILE['doctors']}")

    # Step-by-step API tests
    fips, state = test_location()
    subsidy      = test_subsidy(fips, state)
    plans        = test_plans(fips, state, subsidy["monthly_aptc"])
    test_drugs(plans)
    test_doctors(plans, state)
    test_market(fips, state)

    # Full pipeline (builds enriched plan objects with drug cost models)
    data = await test_full_pipeline()

    # LLM Ranking Agent
    ranking = await test_llm_ranking(data)

    hdr("DONE")
    ok(f"Plans collected: {len(data.get('plans',[]))}")
    ok(f"LLM Ranking: {'✓ computed' if ranking else '— skipped (no Vertex AI)'}")
    ok(f"Synthesis prompt would be ~{len(_build_stub_prompt(data))} chars")

def _build_stub_prompt(data):
    from agents.adk_orchestrator import _build_synthesis_prompt
    return _build_synthesis_prompt(PROFILE, data)

if __name__ == "__main__":
    asyncio.run(main())
