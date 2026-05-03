"""
ADK Tools - wrapping gov_apis for Google ADK tool calling.
"""
import os
import asyncio
from typing import List, Dict, Any, Optional

from tools.gov_apis import (
    calculate_fpl_percentage, search_plans_by_zip,
    get_eligibility_estimates, get_medicaid_threshold, get_fips_from_zip,
    resolve_drug_rxcui, check_drug_coverage, _fips_to_state,
    get_state_exchange, lookup_npi_registry, check_doctor_in_plan_network,
    get_generic_alternatives, get_doctor_quality_score,
    check_sep_eligibility, check_hrsa_shortage
)


def get_location_info(zip_code: str) -> dict:
    """Get FIPS code and State from a ZIP code."""
    fips = get_fips_from_zip(zip_code)
    state = _fips_to_state(fips) if fips else "US"
    return {"fips": fips, "state": state}


def get_subsidy_estimate(income: float, age: int, household_size: int, zip_code: str, tobacco_use: bool = False) -> dict:
    """Calculate FPL percentage and get APTC/CSR eligibility estimates from CMS."""
    fips = get_fips_from_zip(zip_code)
    state = _fips_to_state(fips) if fips else "US"
    fpl_pct = calculate_fpl_percentage(income, household_size, state)

    estimates = get_eligibility_estimates(
        income=income, age=age, fips=fips, zip_code=zip_code, state=state, tobacco_use=tobacco_use
    )

    csr_eligible = 100 <= fpl_pct <= 250
    csr_variant = None
    if 100 <= fpl_pct <= 150: csr_variant = "94"
    elif 150 <= fpl_pct <= 200: csr_variant = "87"
    elif 200 <= fpl_pct <= 250: csr_variant = "73"

    monthly_aptc = estimates.get("aptc", 0)

    return {
        "fpl_percentage": fpl_pct,
        "monthly_aptc": monthly_aptc,
        "monthly_credit": monthly_aptc,
        "annual_credit": monthly_aptc * 12,
        "csr_eligible": csr_eligible,
        "csr_variant": csr_variant,
        "is_medicaid_eligible": fpl_pct < get_medicaid_threshold(state)
    }


def find_plans(zip_code: str, age: int, income: float, household_size: int = 1, tobacco_use: bool = False) -> list:
    """Find available real ACA health insurance plans using CMS Marketplace API."""
    plans = search_plans_by_zip(zip_code, age, income, household_size, tobacco_use)
    if not plans:
        print(f"find_plans: no plans found for {zip_code}")
    else:
        print(f"find_plans: {len(plans)} real plans for {zip_code}")
    return plans


def check_medication_coverage(drug_names: List[str], plan_ids: List[str]) -> dict:
    """Resolve drug names to RxCUIs and check their coverage across plans."""
    resolved = []
    rxcui_list = []
    for name in drug_names:
        info = resolve_drug_rxcui(name)
        if info:
            resolved.append(info)
            rxcui_list.append(info["rxcui"])

    if not rxcui_list:
        return {"results": [], "warnings": ["No medications to check."]}

    coverage = check_drug_coverage(rxcui_list, plan_ids)

    generics = {}
    for name in drug_names:
        g = get_generic_alternatives(name)
        if g:
            generics[name] = [x["generic_name"] for x in g if x.get("is_generic")]

    return {
        "resolved_drugs": resolved,
        "coverage_details": coverage,
        "generic_suggestions": generics
    }


def verify_doctors(doctor_names: List[str], state: str, zip_code: str, plan_ids: List[str]) -> dict:
    """Verify doctors in NPPES registry and check their quality scores and network status."""
    results = []
    for name in doctor_names:
        nppes = lookup_npi_registry(name, state=state)
        npi = nppes.get("npi")
        quality = get_doctor_quality_score(str(npi)) if npi else {}

        networks = {}
        if npi:
            for pid in plan_ids[:3]:
                networks[pid] = check_doctor_in_plan_network(pid, str(npi), zip_code)

        results.append({
            "name": name,
            "npi": npi,
            "nppes_info": nppes,
            "quality_score": quality.get("mips_score"),
            "network_status": networks
        })
    return {"results": results}


def get_market_risks(zip_code: str, state: str) -> dict:
    """Check for HRSA provider shortages and SEP status."""
    fips = get_fips_from_zip(zip_code)
    hrsa = check_hrsa_shortage(state, fips)
    sep = check_sep_eligibility()
    return {"hrsa": hrsa, "sep": sep}
