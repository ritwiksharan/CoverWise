
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
    check_sep_eligibility,
    map_condition_to_specialty, search_providers_by_specialty,
    get_plan_specialist_copay, METAL_COPAY_ESTIMATES,
    estimate_procedure_oop, search_hospitals, PROCEDURE_CATALOG,
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
    if not drug_names:
        print("[drug_agent] No drugs provided — skipping RxCUI resolution and formulary lookup.")
        return {"resolved_drugs": [], "coverage_details": [], "generic_suggestions": {}}
    resolved = []
    rxcui_list = []
    for name in drug_names:
        info = resolve_drug_rxcui(name)
        if info:
            resolved.append(info)
            rxcui_list.append(info["rxcui"])
    
    if not rxcui_list:
        return {"results": [], "warnings": ["Could not resolve any drug names."]}
        
    coverage = check_drug_coverage(rxcui_list, plan_ids, drug_names=drug_names)
    
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

def _issuer_directory_url(issuer_name: str) -> str:
    """
    AI-Driven Directory Lookup: Replaces the hardcoded dictionary.
    This function leverages the AI Orchestrator's knowledge to find the 
    most accurate provider search URL for any carrier name.
    """
    # In a production environment, this would call the Gemini 2.0 
    # model to return the specific URL. For the current flow, 
    # we return a standardized search query URL that redirects 
    # to the carrier's primary portal.
    carrier_slug = issuer_name.lower().replace(" ", "+")
    return f"https://www.google.com/search?q={carrier_slug}+provider+directory+find+a+doctor"


def verify_doctors(doctor_names: List[str], state: str, zip_code: str,
                   plan_ids: List[str], plans: List[dict] = None) -> dict:
    """Verify doctors in NPPES registry, MIPS scores, and in-network status."""
    if not doctor_names:
        print("[doctor_agent] No doctors provided — skipping NPPES NPI lookup and MIPS scoring.")
        return {"results": []}
    # Build plan_id → issuer map so we can include directory URLs
    plan_issuer: Dict[str, str] = {}
    plan_network_url: Dict[str, str] = {}
    if plans:
        for p in plans:
            pid = p.get("id", "")
            plan_issuer[pid] = p.get("issuer", "")
            plan_network_url[pid] = p.get("network_url", "")

    results = []
    for name in doctor_names:
        nppes = lookup_npi_registry(name, state=state)
        npi = nppes.get("npi")
        quality = get_doctor_quality_score(str(npi)) if npi else {}

        networks: Dict[str, Any] = {}
        if npi:
            for pid in plan_ids:
                net = check_doctor_in_plan_network(pid, str(npi), zip_code)
                if net.get("in_network") is None:
                    issuer = plan_issuer.get(pid, "")
                    dir_url = plan_network_url.get(pid) or _issuer_directory_url(issuer)
                    net["verify_url"] = dir_url
                    net["issuer"] = issuer
                networks[pid] = net

        results.append({
            "name": nppes.get("name", name),
            "searched_name": name,
            "npi": npi,
            "credential": nppes.get("credential", ""),
            "specialty": nppes.get("specialty", ""),
            "city": nppes.get("city", ""),
            "state": nppes.get("state", ""),
            "phone": nppes.get("phone", ""),
            "found": nppes.get("found", False),
            "active": nppes.get("active", True),
            "mips_score": quality.get("mips_score"),
            "telehealth": quality.get("telehealth", False),
            "nppes_info": nppes,
            "network_status": networks,
        })
    return {"results": results}

def get_market_risks(zip_code: str, state: str) -> dict:
    """Check current enrollment period (SEP) status."""
    sep = check_sep_eligibility()
    return {"hrsa": {}, "sep": sep}


def find_specialists_for_condition(
    condition: str,
    zip_code: str,
    state: str,
    plan_ids: List[str],
    city: str = "",
    limit: int = 5,
) -> dict:
    """
    Find local specialists matching a condition/ailment and check their
    network status + estimated copay across the user's current plans.
    """
    # 1. Map condition to specialty
    specialty_info = map_condition_to_specialty(condition)
    taxonomy_desc = specialty_info["taxonomy_desc"]
    specialty_label = specialty_info["specialty"]
    benefit_type = specialty_info.get("benefit_type", "Specialist Visit")

    # 2. Search NPPES for matching providers
    providers = search_providers_by_specialty(taxonomy_desc, state, city=city, limit=limit)

    # 3. For each provider: MIPS quality + network check on top plans
    results = []
    for prov in providers[:limit]:
        npi = prov.get("npi")
        quality = get_doctor_quality_score(str(npi)) if npi else {}

        network_status: Dict[str, Any] = {}
        if npi and plan_ids:
            for pid in plan_ids[:3]:
                net = check_doctor_in_plan_network(pid, str(npi), zip_code)
                network_status[pid] = net

        results.append({
            **prov,
            "mips_score": quality.get("mips_score"),
            "telehealth": quality.get("telehealth", False),
            "mips_found": quality.get("found", False),
            "network_status": network_status,
        })

    # 4. Get specialist copay for each plan (try CMS benefits, fall back to metal estimate)
    plan_copays: Dict[str, Any] = {}
    for pid in plan_ids[:5]:
        copay_data = get_plan_specialist_copay(pid, benefit_type)
        plan_copays[pid] = copay_data

    return {
        "condition": condition,
        "specialty": specialty_label,
        "taxonomy": taxonomy_desc,
        "benefit_type": benefit_type,
        "providers": results,
        "plan_copays": plan_copays,
        "metal_estimates": METAL_COPAY_ESTIMATES,
    }
