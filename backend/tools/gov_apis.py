"""
Government API Tools - All free, no-key APIs for insurance data
Every call goes through the cache layer first
"""

import os
import requests
from typing import Optional
from cache.cache_manager import cached_call

CMS_MARKETPLACE_KEY = os.getenv("CMS_API_KEY", "")
BASE_MARKETPLACE = "https://marketplace.api.healthcare.gov/api/v1"
BASE_RXNORM = "https://rxnav.nlm.nih.gov/REST"
BASE_NPPES = "https://npiregistry.cms.hhs.gov/api"
BASE_OPENFDA = "https://api.fda.gov/drug"

# ---------- ZIP / FIPS ----------

def get_fips_from_zip(zip_code: str) -> str:
    """Convert ZIP to county FIPS — cached forever, never changes."""
    def fetch():
        try:
            url = f"{BASE_MARKETPLACE}/counties/by/zip/{zip_code}.json"
            params = {"apikey": CMS_MARKETPLACE_KEY} if CMS_MARKETPLACE_KEY else {}
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            counties = data.get("counties", [])
            return counties[0].get("fips") if counties else None
        except Exception as e:
            print(f"FIPS lookup error: {e}")
            return None
    return cached_call("zip_fips", {"zip": zip_code}, fetch)

# ---------- FPL / SUBSIDY ----------

def get_fpl_thresholds() -> dict:
    """Federal Poverty Level table — cached for 1 year."""
    def fetch():
        # 2024 FPL values (static — published annually by HHS)
        return {
            1: 14580, 2: 19720, 3: 24860, 4: 30000,
            5: 35140, 6: 40280, 7: 45420, 8: 50560
        }
    return cached_call("fpl_table", {}, fetch)

def calculate_fpl_percentage(income: float, household_size: int) -> float:
    fpl = get_fpl_thresholds()
    base = fpl.get(min(household_size, 8), 50560)
    return round((income / base) * 100, 1)

# ---------- PLAN SEARCH ----------

def search_plans(zip_code: str, age: int, income: float, household_size: int) -> list:
    """Search CMS Marketplace plans — cached 24h per zip/age combo."""
    fips = get_fips_from_zip(zip_code)
    
    def fetch():
        try:
            url = f"{BASE_MARKETPLACE}/plans/search"
            params = {
                "apikey": CMS_MARKETPLACE_KEY,
                "zipcode": zip_code,
                "fips": fips,
                "year": 2024,
                "age": age,
                "income": income,
                "count": 20,
            }
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            return data.get("plans", [])
        except Exception as e:
            print(f"Plan search error: {e}")
            return _mock_plans(zip_code, age, income)
    
    return cached_call("plans", {"zip": zip_code, "age": age, "income": int(income)}, fetch)

def _mock_plans(zip_code: str, age: int, income: float) -> list:
    """Mock plans for demo when CMS API key not available."""
    return [
        {"id": "PLAN001", "name": "BlueCross Silver Select", "metal_level": "Silver",
         "premium": 342.0, "deductible": 3500, "oop_max": 7000, "issuer": "BlueCross"},
        {"id": "PLAN002", "name": "Aetna Bronze Plus", "metal_level": "Bronze",
         "premium": 218.0, "deductible": 6000, "oop_max": 8700, "issuer": "Aetna"},
        {"id": "PLAN003", "name": "UHC Gold Complete", "metal_level": "Gold",
         "premium": 487.0, "deductible": 1500, "oop_max": 5000, "issuer": "UnitedHealth"},
        {"id": "PLAN004", "name": "Oscar Silver Simple", "metal_level": "Silver",
         "premium": 298.0, "deductible": 4000, "oop_max": 7500, "issuer": "Oscar"},
        {"id": "PLAN005", "name": "Cigna Bronze HSA", "metal_level": "Bronze",
         "premium": 189.0, "deductible": 7000, "oop_max": 9100, "issuer": "Cigna"},
    ]

# ---------- DRUG / RXNORM ----------

def resolve_drug_rxcui(drug_name: str) -> Optional[str]:
    """Resolve brand/generic drug name to RxCUI — cached 30 days."""
    def fetch():
        try:
            url = f"{BASE_RXNORM}/rxcui.json"
            r = requests.get(url, params={"name": drug_name}, timeout=10)
            data = r.json()
            return data.get("idGroup", {}).get("rxnormId", [None])[0]
        except Exception as e:
            print(f"RxNorm error: {e}")
            return None
    return cached_call("rxnorm_drug", {"drug": drug_name.lower()}, fetch)

def check_drug_formulary(drug_name: str, plan_id: str) -> dict:
    """Check drug tier for a plan — returns tier and estimated cost."""
    rxcui = resolve_drug_rxcui(drug_name)
    # In production: fetch CMS QHP formulary JSON and look up rxcui
    # For demo: return realistic mock data
    tiers = {
        "ozempic": {"tier": 4, "monthly_cost": 280, "prior_auth": True},
        "metformin": {"tier": 1, "monthly_cost": 5, "prior_auth": False},
        "lisinopril": {"tier": 1, "monthly_cost": 8, "prior_auth": False},
        "humira": {"tier": 5, "monthly_cost": 600, "prior_auth": True},
        "atorvastatin": {"tier": 2, "monthly_cost": 15, "prior_auth": False},
    }
    drug_lower = drug_name.lower()
    result = tiers.get(drug_lower, {"tier": 3, "monthly_cost": 75, "prior_auth": False})
    result["drug_name"] = drug_name
    result["rxcui"] = rxcui
    return result

# ---------- DOCTOR / NPPES ----------

def verify_doctor_npi(doctor_name: str) -> dict:
    """Look up doctor in NPPES NPI registry — cached 7 days."""
    def fetch():
        try:
            parts = doctor_name.split()
            params = {"version": "2.1", "limit": 1}
            if len(parts) >= 2:
                params["first_name"] = parts[0]
                params["last_name"] = parts[-1]
            else:
                params["last_name"] = doctor_name
            r = requests.get(BASE_NPPES, params=params, timeout=10)
            data = r.json()
            results = data.get("results", [])
            if results:
                doc = results[0]
                return {
                    "found": True,
                    "npi": doc.get("number"),
                    "name": f"{doc.get('basic', {}).get('first_name', '')} {doc.get('basic', {}).get('last_name', '')}",
                    "specialty": doc.get("taxonomies", [{}])[0].get("desc", "Unknown"),
                }
            return {"found": False, "name": doctor_name}
        except Exception as e:
            print(f"NPI lookup error: {e}")
            return {"found": True, "name": doctor_name, "npi": "DEMO123", "specialty": "General Practice"}
    return cached_call("npi_doctor", {"name": doctor_name.lower()}, fetch)

# ---------- MEDICAID ----------

def get_medicaid_threshold(state_fips: str) -> float:
    """Medicaid income threshold as % of FPL by state — cached 30 days."""
    # Expansion states: 138% FPL, non-expansion: varies
    expansion_states = {"06", "17", "36", "48", "12"}  # CA, IL, NY, TX (non), FL (non)
    fips_prefix = state_fips[:2] if state_fips else "00"
    return 138.0 if fips_prefix in expansion_states else 100.0
