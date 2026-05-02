
"""
ADK Tools - wrapping gov_apis for Google ADK tool calling.
"""
import os
import asyncio
from typing import List, Dict, Any, Optional

from tools.gov_apis import (
    calculate_fpl_percentage, search_plans,
    get_eligibility_estimates, get_medicaid_threshold, get_fips_from_zip,
    resolve_drug_rxcui, check_drug_coverage, verify_doctor_cms, _fips_to_state,
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
    
    # CSR eligibility
    csr_eligible = 100 <= fpl_pct <= 250
    csr_variant = None
    if 100 <= fpl_pct <= 150: csr_variant = "94"
    elif 150 <= fpl_pct <= 200: csr_variant = "87"
    elif 200 <= fpl_pct <= 250: csr_variant = "73"

    return {
        "fpl_percentage": fpl_pct,
        "monthly_aptc": estimates.get("aptc", 0),
        "csr_eligible": csr_eligible,
        "csr_variant": csr_variant,
        "is_medicaid_eligible": fpl_pct < get_medicaid_threshold(state)
    }

def find_plans(zip_code: str, age: int, income: float, tobacco_use: bool = False) -> list:
    """Find available health insurance plans for a specific demographic."""
    fips = get_fips_from_zip(zip_code)
    state = _fips_to_state(fips) if fips else "US"
    return search_plans(zip_code, age, income, fips, state, tobacco_use)

def check_medication_coverage(drug_names: List[str], plan_ids: List[str]) -> dict:
    """Resolve drug names to RxCUIs and check their coverage and tiers across specific plans."""
    resolved = []
    rxcui_list = []
    for name in drug_names:
        info = resolve_drug_rxcui(name)
        if info:
            resolved.append(info)
            rxcui_list.append(info["rxcui"])
    
    if not rxcui_list:
        return {"results": [], "warnings": ["Could not resolve any drug names."]}
        
    coverage = check_drug_coverage(rxcui_list, plan_ids)
    
    # Suggest generics
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
    """Verify if doctors are in NPPES registry, check their MIPS quality scores, and in-network status."""
    results = []
    for name in doctor_names:
        nppes = lookup_npi_registry(name, state=state)
        npi = nppes.get("npi")
        quality = get_doctor_quality_score(str(npi)) if npi else {}
        
        networks = {}
        if npi:
            for pid in plan_ids:
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
    """Check for HRSA provider shortages and current enrollment period (SEP) status."""
    fips = get_fips_from_zip(zip_code)
    hrsa = check_hrsa_shortage(state, fips)
    sep = check_sep_eligibility()
    return {"hrsa": hrsa, "sep": sep}
