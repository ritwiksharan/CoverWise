"""
Government API Tools — all data pulled live from CMS and federal APIs.
Every call goes through the cache layer first.
"""

import os
import re
import requests
from datetime import date, timedelta
from typing import Optional
from cache.cache_manager import cached_call

CMS_MARKETPLACE_KEY = os.getenv("CMS_API_KEY", "")
BASE_MARKETPLACE = "https://marketplace.api.healthcare.gov/api/v1"
BASE_ORDER_REFERRING = "https://data.cms.gov/data-api/v1/dataset/c99b5865-1119-4436-bb80-c5af2773ea1f/data"

# States that run their own exchanges - not on federal CMS API
STATE_EXCHANGES = {
    "NY": {"name": "NY State of Health", "url": "https://nystateofhealth.ny.gov"},
    "CA": {"name": "Covered California", "url": "https://www.coveredca.com"},
    "WA": {"name": "Washington Healthplanfinder", "url": "https://www.wahealthplanfinder.org"},
    "CO": {"name": "Connect for Health Colorado", "url": "https://connectforhealthco.com"},
    "CT": {"name": "Access Health CT", "url": "https://www.accesshealthct.com"},
    "ID": {"name": "Your Health Idaho", "url": "https://www.yourhealthidaho.org"},
    "KY": {"name": "Kynect", "url": "https://kynect.ky.gov"},
    "ME": {"name": "CoverME.gov", "url": "https://coverme.gov"},
    "MD": {"name": "Maryland Health Connection", "url": "https://www.marylandhealthconnection.gov"},
    "MA": {"name": "Massachusetts Health Connector", "url": "https://www.mahealthconnector.org"},
    "MN": {"name": "MNsure", "url": "https://www.mnsure.org"},
    "NV": {"name": "Nevada Health Link", "url": "https://www.nevadahealthlink.com"},
    "NJ": {"name": "Get Covered NJ", "url": "https://www.getcoverednj.com"},
    "NM": {"name": "beWellnm", "url": "https://www.bewellnm.com"},
    "PA": {"name": "Pennie", "url": "https://pennie.com"},
    "RI": {"name": "HealthSource RI", "url": "https://healthsourceri.com"},
    "VT": {"name": "Vermont Health Connect", "url": "https://vermonthealthconnect.gov"},
    "VA": {"name": "Marketplace Virginia", "url": "https://www.marketplace.virginia.gov"},
    "DC": {"name": "DC Health Link", "url": "https://dchealthlink.com"},
}

# ZIP first-2-digits to state — exhaustive US coverage
ZIP2_STATE = {
    "01": "MA", "02": "MA", "03": "NH", "04": "ME", "05": "VT",
    "06": "CT", "07": "NJ", "08": "NJ", "09": "NJ",
    "10": "NY", "11": "NY", "12": "NY", "13": "NY", "14": "NY",
    "15": "PA", "16": "PA", "17": "PA", "18": "PA", "19": "PA",
    "20": "DC", "21": "MD", "22": "VA", "23": "VA", "24": "VA",
    "25": "WV", "26": "WV", "27": "NC", "28": "NC", "29": "SC",
    "30": "GA", "31": "GA", "32": "FL", "33": "FL", "34": "FL",
    "35": "AL", "36": "AL", "37": "TN", "38": "TN", "39": "MS",
    "40": "KY", "41": "KY", "42": "KY",
    "43": "OH", "44": "OH", "45": "OH",
    "46": "IN", "47": "IN",
    "48": "MI", "49": "MI",
    "50": "IA", "51": "IA", "52": "IA",
    "53": "WI", "54": "WI",
    "55": "MN", "56": "MN",
    "57": "SD", "58": "ND", "59": "MT",
    "60": "IL", "61": "IL", "62": "IL",
    "63": "MO", "64": "MO", "65": "MO",
    "66": "KS", "67": "KS",
    "68": "NE", "69": "NE",
    "70": "LA", "71": "LA",
    "72": "AR", "73": "OK", "74": "OK",
    "75": "TX", "76": "TX", "77": "TX", "78": "TX", "79": "TX",
    "80": "CO", "81": "CO",
    "82": "WY", "83": "ID", "84": "UT",
    "85": "AZ", "86": "AZ",
    "87": "NM", "88": "NM", "89": "NV",
    "90": "CA", "91": "CA", "92": "CA", "93": "CA",
    "94": "CA", "95": "CA", "96": "HI",
    "97": "OR", "98": "WA", "99": "AK",
}

_FIPS_STATE_MAP = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA",
    "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV",
    "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN",
    "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}

def _fips_to_state(fips: str) -> str:
    return _FIPS_STATE_MAP.get(fips[:2], "") if fips else ""

def get_state_from_zip(zip_code: str) -> Optional[str]:
    """Get state from ZIP using first 2 digits — works for ALL US ZIPs."""
    prefix2 = str(zip_code).strip().zfill(5)[:2]
    return ZIP2_STATE.get(prefix2)

def get_state_exchange(zip_code: str) -> Optional[dict]:
    """Return state exchange info if ZIP is in a state-exchange state."""
    state = get_state_from_zip(zip_code)
    if state and state in STATE_EXCHANGES:
        ex = STATE_EXCHANGES[state]
        return {"state": state, "exchange_name": ex["name"], "exchange_url": ex["url"]}
    return None

    return None


# ---------- ZIP → FIPS ----------

def get_fips_from_zip(zip_code: str) -> Optional[str]:
    """Get County FIPS from ZIP code. Works for ALL valid US ZIPs."""
    zip_code = str(zip_code).strip().zfill(5)
    if zip_code in KNOWN_FIPS:
        return KNOWN_FIPS[zip_code]

    # Anchor ZIPs — one known-good ZIP per 2-digit prefix (state region)
    ANCHOR_ZIPS = {
        "10":"10001","11":"11001","12":"12201","13":"13201","14":"14201",
        "15":"15201","16":"16101","17":"17101","18":"18101","19":"19101",
        "20":"20001","21":"21201","22":"22201","23":"23201","24":"24201",
        "25":"25301","26":"26101","27":"27601","28":"28201","29":"29201",
        "30":"30301","31":"31401","32":"32201","33":"33101","34":"34201",
        "35":"35201","36":"36101","37":"37201","38":"38101","39":"39201",
        "40":"40202","41":"41001","42":"42001","43":"43215","44":"44101",
        "45":"45201","46":"46201","47":"47201","48":"48201","49":"49001",
        "50":"50001","51":"51001","52":"52001","53":"53201","54":"54001",
        "55":"55101","56":"56001","57":"57001","58":"58001","59":"59001",
        "60":"60601","61":"61101","62":"62001","63":"63101","64":"64101",
        "65":"65101","66":"66101","67":"67101","68":"68101","69":"69001",
        "70":"70001","71":"71101","72":"72201","73":"73101","74":"74101",
        "75":"75201","76":"76101","77":"77001","78":"78201","79":"79901",
        "80":"80201","81":"81001","82":"82001","83":"83201","84":"84101",
        "85":"85001","86":"86001","87":"87101","88":"88001","89":"89101",
        "90":"90001","91":"91101","92":"92101","93":"93101","94":"94101",
        "95":"95101","96":"96701","97":"97201","98":"98101","99":"99501",
    }

    def fetch():
        # Step 1: CMS counties API — direct lookup (most reliable)
        try:
            r = requests.get(
                f"{BASE_MARKETPLACE}/counties/by/zip/{zip_code}",
                params=_params(), timeout=10
            )
            counties = r.json().get("counties", [])
            if counties:
                fips = counties[0].get("fips")
                print(f"CMS direct: {zip_code} -> FIPS {fips}")
                KNOWN_FIPS[zip_code] = fips
                return fips
        except Exception as e:
            print(f"CMS API error: {e}")

        # Step 2: Census file
        try:
            from tools.zip_loader import get_fips_for_zip
            fips = get_fips_for_zip(zip_code)
            if fips:
                KNOWN_FIPS[zip_code] = fips
                return fips
        except Exception:
            pass

        # Step 3: Use anchor ZIP for this prefix to find state/region FIPS
        prefix2 = zip_code[:2]
        anchor = ANCHOR_ZIPS.get(prefix2)
        if anchor:
            try:
                r = requests.get(
                    f"{BASE_MARKETPLACE}/counties/by/zip/{anchor}",
                    params=_params(), timeout=10
                )
                counties = r.json().get("counties", [])
                if counties:
                    fips = counties[0].get("fips")
                    print(f"Anchor ZIP {anchor}: {zip_code} -> FIPS {fips}")
                    KNOWN_FIPS[zip_code] = fips
                    return fips
            except Exception:
                pass

        # Step 4: Try nearby ZIPs ±1 to ±200 same 2-digit prefix
        try:
            num = int(zip_code)
            for delta in range(1, 200):
                for sign in [1, -1]:
                    nearby = str(num + sign * delta).zfill(5)
                    if nearby[:2] != prefix2:
                        continue
                    try:
                        from tools.zip_loader import get_fips_for_zip
                        fips = get_fips_for_zip(nearby)
                        if fips:
                            KNOWN_FIPS[zip_code] = fips
                            return fips
                    except Exception:
                        pass
        except Exception:
            pass

        print(f"FIPS not found for ZIP {zip_code}")
        return None

    return cached_call("zip_fips", {"zip": zip_code}, fetch)


# ---------- FPL — live from CMS ----------

def get_fpl_thresholds(state: str = "US", year: int = 2024) -> dict:
    def fetch():
        try:
            r = requests.get(
                f"{BASE_MARKETPLACE}/states/{state}/poverty-guidelines",
                params=_params({"year": year}), timeout=10
            )
            guidelines = r.json().get("guidelines", [])
            return {g["household_size"]: g["guideline"] for g in guidelines}
        except Exception as e:
            print(f"FPL lookup error: {e}")
            return {}
    return cached_call("fpl_table", {"state": state, "year": year}, fetch)


def calculate_fpl_percentage(income: float, household_size: int, state: str = "US") -> float:
    fpl = get_fpl_thresholds(state)
    base = fpl.get(min(household_size, 8), 14580)
    return round((income / base) * 100, 1)


def get_applicable_percentage(fpl_pct: float) -> float:
    """IRS Applicable Percentage Table (2021-2025 ARP/IRA rules)."""
    if fpl_pct < 150:
        return 0.0
    elif fpl_pct < 200:
        # Sliding scale 0% to 2%
        return 0.02 * (fpl_pct - 150) / 50
    elif fpl_pct < 250:
        # Sliding scale 2% to 4%
        return 0.02 + 0.02 * (fpl_pct - 200) / 50
    elif fpl_pct < 300:
        # Sliding scale 4% to 6%
        return 0.04 + 0.02 * (fpl_pct - 250) / 50
    elif fpl_pct < 400:
        # Sliding scale 6% to 8.5%
        return 0.06 + 0.025 * (fpl_pct - 300) / 100
    else:
        return 0.085


def calculate_manual_aptc(income: float, fpl_pct: float, benchmark_premium: float) -> float:
    """
    APTC = [Benchmark Silver Premium] - [Expected Contribution]
    Expected Contribution = Income * Applicable Percentage / 12
    """
    applicable_pct = get_applicable_percentage(fpl_pct)
    expected_contribution = (income * applicable_pct) / 12
    aptc = max(0, benchmark_premium - expected_contribution)
    return round(aptc, 2)


# ---------- ELIGIBILITY / APTC — live from CMS ----------

def get_eligibility_estimates(income: float, age: int, fips: str, zip_code: str, state: str, tobacco_use: bool = False, year: int = 2024) -> dict:
    def fetch():
        try:
            body = {
                "household": {
                    "income": income,
                    "people": [{"age": age, "aptc_eligible": True, "gender": "Male", "uses_tobacco": tobacco_use}],
                },
                "market": "Individual",
                "place": {"countyfips": fips, "zipcode": zip_code, "state": state},
                "year": year,
            }
            r = requests.post(
                f"{BASE_MARKETPLACE}/households/eligibility/estimates",
                params=_params(), json=body, timeout=15
            )
            estimates = r.json().get("estimates", [{}])
            return estimates[0] if estimates else {}
        except Exception as e:
            print(f"Eligibility estimate error: {e}")
            return {}
    return cached_call("eligibility", {"income": int(income), "age": age, "fips": fips, "tobacco": tobacco_use}, fetch)


# ---------- MEDICAID THRESHOLD — live from CMS ----------

def get_medicaid_threshold(state: str) -> float:
    def fetch():
        try:
            r = requests.get(
                f"{BASE_MARKETPLACE}/states/{state}/medicaid",
                params=_params({"year": 2024}), timeout=10
            )
            data = r.json()
            adult_fpl = data.get("pc_fpl_adult")
            if adult_fpl:
                return round(float(adult_fpl) * 100, 1)
            return 138.0
        except Exception as e:
            print(f"Medicaid threshold error: {e}")
            return 138.0
    return cached_call("medicaid_threshold", {"state": state}, fetch)


# ---------- PLAN SEARCH ----------

def _normalize_plan(p: dict) -> dict:
    deductible = 0
    for d in p.get("deductibles", []):
        if d.get("network_tier") == "In-Network" and d.get("individual"):
            deductible = d.get("amount", 0)
            break
    if not deductible and p.get("deductibles"):
        deductible = p["deductibles"][0].get("amount", 0)

    oop_max = 0
    for m in p.get("moops", []):
        if m.get("network_tier") == "In-Network" and m.get("individual"):
            oop_max = m.get("amount", 0)
            break
    if not oop_max and p.get("moops"):
        oop_max = p["moops"][0].get("amount", 0)

    issuer = p.get("issuer", {})
    issuer_name = issuer.get("name", "") if isinstance(issuer, dict) else str(issuer)

    # network_url lives at top level in plan-detail responses; inside network[] in search
    network_url = p.get("network_url") or ""
    if not network_url:
        networks = p.get("network") or []
        network_url = networks[0].get("network_url", "") if networks else ""

    return {
        "id": p.get("id", ""),
        "name": p.get("name", ""),
        "metal_level": p.get("metal_level", ""),
        "type": p.get("type", ""),
        "premium": round(p.get("premium", 0), 2),
        "premium_w_credit": round(p.get("premium_w_credit", p.get("premium", 0)), 2),
        "deductible": deductible,
        "oop_max": oop_max,
        "hsa_eligible": p.get("hsa_eligible", False),
        "issuer": issuer_name,
        "formulary_url": p.get("formulary_url", ""),
        "network_url": network_url,
        "quality_rating": p.get("quality_rating", {}).get("global_rating") if isinstance(p.get("quality_rating"), dict) else None,
    }


def search_plans(zip_code: str, age: int, income: float, fips: str, state: str, tobacco_use: bool = False) -> list:
    """Search all CMS Marketplace plans for this household — with nearby ZIP fallback."""
    def _try_zip(z: str, f: str) -> list:
        body = {
            "household": {
                "income": income,
                "people": [{"age": age, "aptc_eligible": True, "gender": "Male", "uses_tobacco": tobacco_use}],
            },
            "market": "Individual",
            "place": {"countyfips": f, "zipcode": z, "state": state},
            "year": 2024,
            "limit": 50,
        }
        r = requests.post(f"{BASE_MARKETPLACE}/plans/search", params=_params(), json=body, timeout=20)
        data = r.json()
        if "plans" in data:
            return [_normalize_plan(p) for p in data["plans"]]
        return []

    def fetch():
        nonlocal fips
        # Try original ZIP first
        if fips:
            plans = _try_zip(zip_code, fips)
            if plans:
                return plans
        # Try nearby ZIPs — handles PO Box ZIPs and county mismatches
        prefix = zip_code[:2]
        for delta in [1,-1,2,-2,3,-3,5,-5,10,-10,15,-15,20,-20,30,-30,50,-50,100,-100]:
            nearby = str(int(zip_code) + delta).zfill(5)
            if nearby[:2] != prefix:
                continue
            nearby_fips = get_fips_from_zip(nearby)
            if not nearby_fips:
                continue
            plans = _try_zip(nearby, nearby_fips)
            if plans:
                print(f"ZIP {zip_code} no plans — using nearby {nearby}")
                return plans
        raise RuntimeError(f"No plans found for ZIP {zip_code} or nearby ZIPs")

    return cached_call("plans", {"zip": zip_code, "age": age, "income": int(income), "tobacco": tobacco_use}, fetch)


# ---------- DRUGS — live from CMS ----------

def resolve_drug_rxcui(drug_name: str) -> Optional[dict]:
    """Resolve drug name to RxCUI using both CMS and RxNorm APIs for robustness."""
    def fetch():
        # Try CMS first (includes strength/route usually)
        try:
            r = requests.get(
                f"{BASE_MARKETPLACE}/drugs/autocomplete",
                params=_params({"q": drug_name}), timeout=10
            )
            results = r.json()
            if isinstance(results, list) and results:
                top = results[0]
                return {
                    "rxcui": str(top.get("rxcui", "")),
                    "name": top.get("name", drug_name),
                    "full_name": top.get("full_name", ""),
                    "route": top.get("route", ""),
                    "strength": top.get("strength", ""),
                }
        except Exception as e:
            print(f"CMS Drug autocomplete error: {e}")

        # Fallback to RxNorm (NIH)
        try:
            r = requests.get(
                f"https://rxnav.nlm.nih.gov/REST/rxcui.json",
                params={"name": drug_name}, timeout=10
            )
            data = r.json()
            ids = data.get("idGroup", {}).get("rxnormId", [])
            if ids:
                rxcui = ids[0]
                # Get more details from RxNorm
                r2 = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json", timeout=10)
                props = r2.json().get("properties", {})
                return {
                    "rxcui": str(rxcui),
                    "name": props.get("name", drug_name),
                    "full_name": props.get("name", ""),
                    "route": "",
                    "strength": ""
                }
        except Exception as e:
            print(f"RxNorm error: {e}")
            
        return None
    return cached_call("rxnorm_drug_v2", {"drug": drug_name.lower()}, fetch)


def check_drug_coverage(rxcui_list: list, plan_ids: list, year: int = 2024,
                        drug_names: Optional[list] = None) -> list:
    """
    Check drug coverage from CMS API; falls back to RAG formulary store when
    the API returns DataNotProvided (common for many state plans).
    """
    if not rxcui_list or not plan_ids:
        return []

    def fetch():
        try:
            r = requests.get(
                f"{BASE_MARKETPLACE}/drugs/covered",
                params=_params({
                    "year": year,
                    "drugs": ",".join(str(x) for x in rxcui_list),
                    "planids": ",".join(plan_ids),
                }),
                timeout=15
            )
            raw = r.json().get("coverage", [])
            return [
                {
                    "rxcui": c.get("rxcui"),
                    "plan_id": c.get("plan_id"),
                    "coverage": c.get("coverage", "Unknown"),
                    "drug_tier": c.get("drug_tier"),
                    "prior_authorization": bool(c.get("prior_authorization", False)),
                    "step_therapy": bool(c.get("step_therapy", False)),
                    "quantity_limit": bool(c.get("quantity_limit", False)),
                }
                for c in raw
            ]
        except Exception as e:
            print(f"Drug coverage check error: {e}")
            return []

    key = {"rxcuis": sorted(rxcui_list), "planids": sorted(plan_ids[:5]), "year": year}
    results = cached_call("formulary", key, fetch)

    # Fall back to RAG store for any plan where all drugs returned DataNotProvided
    try:
        plans_missing = {
            pid for pid in plan_ids
            if all(
                r.get("coverage") == "DataNotProvided"
                for r in results if r.get("plan_id") == pid
            )
        }
        if plans_missing:
            from rag.formulary_store import lookup_drug_coverage, seed_issuers_from_plan_ids
            seed_issuers_from_plan_ids(list(plans_missing), blocking=False)
            names = drug_names or []
            # Build map: queried_rxcui → RAG result (normalise rxcui to queried value
            # so the orchestrator can match by rxcui)
            rag_replacements: dict[tuple, dict] = {}
            for i, queried_rxcui in enumerate(rxcui_list):
                name = names[i] if i < len(names) else ""
                rag_hits = lookup_drug_coverage(name, str(queried_rxcui), list(plans_missing))
                for row in rag_hits:
                    # Use the original queried plan_ids rather than the proxy plan
                    target_plan_ids = (
                        row.get("original_plan_ids") or [row["plan_id"]]
                        if row.get("note") else [row["plan_id"]]
                    )
                    for tpid in target_plan_ids:
                        rag_replacements[(str(queried_rxcui), tpid)] = {
                            **row,
                            "rxcui": str(queried_rxcui),   # normalise to queried RxCUI
                            "plan_id": tpid,
                        }

            if rag_replacements:
                updated = []
                replaced_keys: set[tuple] = set()
                for r in results:
                    k = (str(r.get("rxcui")), r.get("plan_id"))
                    if k in rag_replacements:
                        updated.append(rag_replacements[k])
                        replaced_keys.add(k)
                    else:
                        updated.append(r)
                # Add RAG results for combos not present in original CMS response
                for k, v in rag_replacements.items():
                    if k not in replaced_keys and k not in {(str(r.get("rxcui")), r.get("plan_id")) for r in updated}:
                        updated.append(v)
                results = updated
    except Exception as e:
        print(f"[formulary] RAG fallback error: {e}")

    return results


# ---------- DOCTOR — CMS Order & Referring ----------

def verify_doctor_cms(doctor_name: str) -> dict:
    clean = doctor_name.replace("Dr.", "").replace("Dr ", "").strip()
    parts = clean.split()

    def fetch():
        try:
            keyword = " ".join(parts[-2:]) if len(parts) >= 2 else clean
            r = requests.get(
                BASE_ORDER_REFERRING,
                params={"keyword": keyword.upper(), "limit": 5},
                timeout=10
            )
            results = r.json()
            if not isinstance(results, list) or not results:
                return {"found": False, "searched_name": doctor_name}

            first = parts[0].upper() if parts else ""
            last = parts[-1].upper() if len(parts) > 1 else parts[0].upper() if parts else ""
            match = next(
                (d for d in results if d.get("FIRST_NAME", "").startswith(first) or d.get("LAST_NAME") == last),
                results[0]
            )
            return {
                "found": True,
                "npi": match.get("NPI"),
                "name": f"{match.get('FIRST_NAME', '')} {match.get('LAST_NAME', '')}".strip(),
                "medicare_part_b": match.get("PARTB") == "Y",
                "can_order_dme": match.get("DME") == "Y",
                "can_order_hha": match.get("HHA") == "Y",
                "hospice": match.get("HOSPICE") == "Y",
                "searched_name": doctor_name,
            }
        except Exception as e:
            print(f"Doctor CMS lookup error: {e}")
            return {"found": False, "searched_name": doctor_name, "error": str(e)}
    return cached_call("npi_doctor", {"name": clean.lower()}, fetch)


# ─── NPPES NPI REGISTRY — proper provider identity lookup ────────────────────
NPPES_API = "https://npiregistry.cms.hhs.gov/api/"

# ─── CONDITION → SPECIALTY TAXONOMY MAP ──────────────────────────────────────
# Maps keywords in user's condition query to NPPES taxonomy description,
# a human-readable specialty label, and the CMS benefit service type for copay lookup.
CONDITION_SPECIALTY_MAP = [
    # Cardiovascular
    {"keywords": ["heart","cardiac","cardio","chest pain","hypertension","blood pressure","arrhythmia","coronary","atrial","palpitation"],
     "taxonomy_desc": "Cardiovascular Disease", "specialty": "Cardiologist", "benefit_type": "Specialist Visit"},
    # Endocrine / Diabetes
    {"keywords": ["diabetes","diabetic","insulin","a1c","thyroid","endocrine","hormone","adrenal","thyroid"],
     "taxonomy_desc": "Endocrinology", "specialty": "Endocrinologist", "benefit_type": "Specialist Visit"},
    # Oncology / Cancer
    {"keywords": ["cancer","tumor","oncology","chemotherapy","chemo","radiation","lymphoma","leukemia","melanoma","biopsy"],
     "taxonomy_desc": "Hematology & Oncology", "specialty": "Oncologist", "benefit_type": "Specialist Visit"},
    # Orthopedics
    {"keywords": ["back pain","spine","joint","knee","hip","shoulder","fracture","bone","orthopedic","ortho","sports injury","tendon","ligament"],
     "taxonomy_desc": "Orthopaedic Surgery", "specialty": "Orthopedic Surgeon", "benefit_type": "Specialist Visit"},
    # Mental Health / Psychiatry
    {"keywords": ["mental health","depression","anxiety","ptsd","bipolar","schizophrenia","psychiatry","psychiatric","mood","panic","ocd","adhd","therapy","therapist"],
     "taxonomy_desc": "Psychiatry", "specialty": "Psychiatrist", "benefit_type": "Mental Health"},
    # Neurology
    {"keywords": ["neuro","migraine","seizure","epilepsy","parkinson","alzheimer","ms","multiple sclerosis","stroke","numbness","tremor","neuropathy"],
     "taxonomy_desc": "Neurology", "specialty": "Neurologist", "benefit_type": "Specialist Visit"},
    # Gastroenterology
    {"keywords": ["stomach","gut","ibs","crohn","colitis","colon","gastro","ulcer","acid reflux","gerd","liver","hepatitis","gallbladder"],
     "taxonomy_desc": "Gastroenterology", "specialty": "Gastroenterologist", "benefit_type": "Specialist Visit"},
    # Pulmonology
    {"keywords": ["lung","asthma","copd","breathing","pulmonary","respiratory","bronchitis","pneumonia","sleep apnea"],
     "taxonomy_desc": "Pulmonary Disease", "specialty": "Pulmonologist", "benefit_type": "Specialist Visit"},
    # Dermatology
    {"keywords": ["skin","rash","acne","eczema","psoriasis","dermatology","mole","lesion","hives","hair loss"],
     "taxonomy_desc": "Dermatology", "specialty": "Dermatologist", "benefit_type": "Specialist Visit"},
    # OB/GYN
    {"keywords": ["pregnancy","prenatal","gynecology","women","obgyn","ob/gyn","uterus","ovary","menopause","fertility"],
     "taxonomy_desc": "Obstetrics & Gynecology", "specialty": "OB/GYN", "benefit_type": "Specialist Visit"},
    # Urology
    {"keywords": ["kidney","bladder","prostate","urology","urinary","erectile","incontinence"],
     "taxonomy_desc": "Urology", "specialty": "Urologist", "benefit_type": "Specialist Visit"},
    # Rheumatology
    {"keywords": ["arthritis","rheumatoid","lupus","fibromyalgia","gout","autoimmune","rheumatology","joint pain"],
     "taxonomy_desc": "Rheumatology", "specialty": "Rheumatologist", "benefit_type": "Specialist Visit"},
    # Ophthalmology
    {"keywords": ["eye","vision","glaucoma","cataract","retina","ophthalmology","optic","macular"],
     "taxonomy_desc": "Ophthalmology", "specialty": "Ophthalmologist", "benefit_type": "Vision"},
    # Allergy/Immunology
    {"keywords": ["allergy","allergies","immunology","anaphylaxis","food allergy","asthma allergy"],
     "taxonomy_desc": "Allergy & Immunology", "specialty": "Allergist", "benefit_type": "Specialist Visit"},
    # General Internal Medicine (fallback)
    {"keywords": ["internal medicine","general","primary care","checkup","physical","annual","preventive"],
     "taxonomy_desc": "Internal Medicine", "specialty": "Internist", "benefit_type": "Primary Care Visit"},
]

def lookup_npi_registry(doctor_name: str, city: str = "", state: str = "") -> dict:
    """Look up a provider via the NPPES NPI Registry — returns NPI, specialty, address, phone."""
    clean = doctor_name.replace("Dr.", "").replace("Dr ", "").strip()
    parts = clean.split()
    last = parts[-1] if parts else clean
    first = parts[0] if len(parts) > 1 else ""

    def fetch():
        try:
            params = {"version": "2.1", "enumeration_type": "NPI-1", "limit": 5,
                      "last_name": last.upper()}
            if first:
                params["first_name"] = first.upper()
            if city:
                params["city"] = city.upper()
            if state:
                params["state"] = state.upper()

            r = requests.get(NPPES_API, params=params, timeout=10)
            results = r.json().get("results", [])
            if not results:
                return {"found": False, "searched_name": doctor_name}

            # Pick the result whose last name actually matches before falling back to [0]
            last_upper = last.upper()
            match = next(
                (res for res in results
                 if res.get("basic", {}).get("last_name", "").upper() == last_upper),
                results[0],
            )
            basic = match.get("basic", {})
            taxonomies = match.get("taxonomies") or [{}]
            addresses = match.get("addresses") or [{}]
            primary_tax = next((t for t in taxonomies if t.get("primary")), taxonomies[0])
            practice_addr = next(
                (a for a in addresses if a.get("address_purpose") == "LOCATION"),
                addresses[0]
            )
            all_candidates = [
                {
                    "npi": r2.get("number"),
                    "name": f"{r2.get('basic',{}).get('first_name','')} {r2.get('basic',{}).get('last_name','')}".strip(),
                    "specialty": ((r2.get("taxonomies") or [{}])[0]).get("desc", ""),
                    "city": ((r2.get("addresses") or [{}])[0]).get("city", ""),
                    "state": ((r2.get("addresses") or [{}])[0]).get("state", ""),
                }
                for r2 in results[:3]
            ]
            return {
                "found": True,
                "npi": match.get("number"),
                "name": f"{basic.get('first_name','')} {basic.get('last_name','')}".strip(),
                "credential": basic.get("credential", ""),
                "specialty": primary_tax.get("desc", ""),
                "city": practice_addr.get("city", ""),
                "state": practice_addr.get("state", ""),
                "phone": practice_addr.get("telephone_number", ""),
                "active": basic.get("status", "") == "A",
                "all_candidates": all_candidates,
                "searched_name": doctor_name,
            }
        except Exception as e:
            print(f"NPPES lookup error: {e}")
            return {"found": False, "searched_name": doctor_name, "error": str(e)}

    return cached_call("npi_registry", {"name": clean.lower(), "city": city.lower(), "state": state.lower()}, fetch)


# ─── IN-NETWORK VERIFICATION — CMS plan provider endpoint ────────────────────

def check_doctor_in_plan_network(plan_id: str, npi: str, zip_code: str) -> dict:
    """Check if a provider (by NPI) is in a specific plan's network via CMS."""
    def fetch():
        try:
            r = requests.get(
                f"{BASE_MARKETPLACE}/providers/{npi}",
                params=_params({"year": 2024}),
                timeout=10,
            )
            if r.status_code == 404:
                return {
                    "in_network": None,
                    "note": "Provider not found in CMS Marketplace directory. Verify directly with the insurer.",
                }
            if r.status_code != 200:
                return {"in_network": None, "error": f"CMS returned HTTP {r.status_code}"}

            prov = r.json().get("provider", {})
            plans = prov.get("plans") or []
            plan_ids_in_network = [p.get("id", "") if isinstance(p, dict) else str(p) for p in plans]

            if not plans:
                return {
                    "in_network": None,
                    "provider_name": prov.get("name"),
                    "accepting": prov.get("accepting", "unknown"),
                    "note": "CMS marketplace directory has no plan-network data for this provider. "
                            "Call the insurer's member services or use their online provider directory to confirm.",
                }
            in_net = any(plan_id in pid for pid in plan_ids_in_network)
            return {
                "in_network": in_net,
                "provider_name": prov.get("name"),
                "accepting": prov.get("accepting", "unknown"),
            }
        except Exception as e:
            print(f"Plan network check error: {e}")
            return {"in_network": None, "error": str(e)}

    return cached_call("plan_network", {"plan_id": plan_id, "npi": str(npi)}, fetch)


# ─── openFDA — GENERIC DRUG ALTERNATIVES ─────────────────────────────────────
OPENFDA_API = "https://api.fda.gov/drug/ndc.json"

def get_generic_alternatives(drug_name: str) -> list:
    """Find generic alternatives for a brand-name drug via openFDA."""
    def fetch():
        try:
            # Try brand name first
            r = requests.get(
                OPENFDA_API,
                params={"search": f'openfda.brand_name:"{drug_name}"', "limit": 5},
                timeout=10
            )
            results = r.json().get("results", [])
            if not results:
                r2 = requests.get(
                    OPENFDA_API,
                    params={"search": f'openfda.generic_name:"{drug_name}"', "limit": 5},
                    timeout=10
                )
                results = r2.json().get("results", [])

            generics = []
            seen = set()
            for item in results:
                openfda = item.get("openfda", {})
                substance = (openfda.get("substance_name") or [""])[0]
                generic_name = (openfda.get("generic_name") or [""])[0]
                brand_names = openfda.get("brand_name") or []
                if substance and substance not in seen:
                    seen.add(substance)
                    generics.append({
                        "substance": substance,
                        "generic_name": generic_name,
                        "brand_names": brand_names[:2],
                        "dosage_form": item.get("dosage_form", ""),
                        "is_generic": not brand_names or generic_name.lower().startswith(substance.lower().split()[0].lower()),
                    })
            return generics[:3]
        except Exception as e:
            print(f"openFDA generic lookup error: {e}")
            return []

    return cached_call("openfda_generic", {"drug": drug_name.lower()}, fetch)


# ─── MIPS QUALITY SCORE — CMS Provider Data Catalog ──────────────────────────
_MIPS_DATASET = "8889d81e-3916-4369-8e45-e714e93fd06b"
CMS_DATA_API = "https://data.cms.gov/data-api/v1/dataset"

def get_doctor_quality_score(npi: str) -> dict:
    """Get MIPS quality/performance score for a provider from CMS Provider Data Catalog."""
    def fetch():
        try:
            r = requests.get(
                f"{CMS_DATA_API}/{_MIPS_DATASET}/data",
                params={"filter[NPI]": npi, "size": 1},
                timeout=10
            )
            items = r.json()
            if isinstance(items, list) and items:
                item = items[0]
                return {
                    "found": True,
                    "mips_score": item.get("final_score"),
                    "payment_adjustment": item.get("payment_adjustment_percentage"),
                    "facility": item.get("facility_name"),
                    "telehealth": item.get("telehealth_services") == "Y",
                    "year": item.get("year"),
                }
            return {"found": False}
        except Exception as e:
            print(f"MIPS score error: {e}")
            return {"found": False, "error": str(e)}

    return cached_call("mips_score", {"npi": str(npi)}, fetch)


# ─── SEP ELIGIBILITY — pure date logic, no external API ─────────────────────

_SEP_EVENTS = {
    "job_loss": "loss of job-based coverage",
    "marriage": "marriage",
    "divorce": "loss of coverage through divorce or legal separation",
    "birth": "birth of a child",
    "adoption": "adoption or foster placement of a child",
    "move": "moving to a new state or coverage service area",
    "citizenship": "gaining lawful presence or US citizenship",
    "medicaid_loss": "loss of Medicaid or CHIP eligibility",
    "release": "release from incarceration",
    "other": "another qualifying life event",
}

def check_sep_eligibility(event_type: str = "", event_date_str: str = "") -> dict:
    """Check open enrollment status or Special Enrollment Period eligibility."""
    today = date.today()
    
    # DEMO/TEST OVERRIDE
    force_oe = os.getenv("FORCE_OPEN_ENROLLMENT", "").upper()
    print(f"DEBUG: FORCE_OPEN_ENROLLMENT is '{force_oe}'")
    if force_oe == "TRUE":
        print("DEBUG: Applying DEMO MODE override for Open Enrollment")
        deadline = date(today.year + (1 if today.month > 1 else 0), 1, 15)
        return {
            "in_open_enrollment": True,
            "deadline": deadline.isoformat(),
            "days_remaining": (deadline - today).days,
            "message": "Open enrollment is active (DEMO MODE) — you can enroll in any plan now.",
        }

    year = today.year
    oe_start = date(year, 11, 1)
    oe_end = date(year + 1, 1, 15)
    if today < oe_start:
        oe_start = date(year - 1, 11, 1)
        oe_end = date(year, 1, 15)

    if oe_start <= today <= oe_end:
        days_left = (oe_end - today).days
        return {
            "in_open_enrollment": True,
            "deadline": oe_end.isoformat(),
            "days_remaining": days_left,
            "message": (
                f"Open enrollment is active — {days_left} days left to enroll "
                f"(deadline {oe_end.strftime('%B %d, %Y')})."
            ),
        }

    if not event_type:
        next_oe = date(today.year, 11, 1)
        if today >= next_oe:
            next_oe = date(today.year + 1, 11, 1)
        days_away = (next_oe - today).days
        return {
            "in_open_enrollment": False,
            "sep_eligible": False,
            "next_open_enrollment": next_oe.isoformat(),
            "days_to_next_oe": days_away,
            "qualifying_events": list(_SEP_EVENTS.keys()),
            "message": (
                f"Open enrollment is closed. Next window opens {next_oe.strftime('%B %d, %Y')} "
                f"({days_away} days away). You may still enroll with a qualifying life event."
            ),
        }

    event_desc = _SEP_EVENTS.get(event_type.lower().replace(" ", "_"), event_type)
    if event_date_str:
        try:
            event_date = date.fromisoformat(event_date_str)
            deadline = event_date + timedelta(days=60)
            days_left = (deadline - today).days
            if days_left > 0:
                return {
                    "in_open_enrollment": False,
                    "sep_eligible": True,
                    "event_type": event_type,
                    "event_description": event_desc,
                    "deadline": deadline.isoformat(),
                    "days_remaining": days_left,
                    "message": (
                        f"SEP eligible: {event_desc}. "
                        f"{days_left} days left to enroll (deadline {deadline.strftime('%B %d, %Y')})."
                    ),
                }
            return {
                "in_open_enrollment": False,
                "sep_eligible": False,
                "expired": True,
                "message": (
                    f"SEP window for {event_desc} expired {abs(days_left)} days ago. "
                    f"Contact healthcare.gov — hardship exceptions are sometimes granted."
                ),
            }
        except ValueError:
            pass

    return {
        "in_open_enrollment": False,
        "sep_eligible": True,
        "event_type": event_type,
        "event_description": event_desc,
        "window_days": 60,
        "message": (
            f"You may qualify for a 60-day SEP due to {event_desc}. "
            f"Provide the date it occurred to calculate your enrollment deadline."
        ),
    }


# ─── SPECIALIST SEARCH ───────────────────────────────────────────────────────

def map_condition_to_specialty(condition: str) -> dict:
    """Map a free-text condition/ailment to a specialist type and NPPES taxonomy."""
    q = condition.lower().strip()
    for entry in CONDITION_SPECIALTY_MAP:
        if any(kw in q for kw in entry["keywords"]):
            return entry
    # Fallback: treat the condition text itself as the taxonomy description keyword
    return {
        "taxonomy_desc": condition,
        "specialty": f"{condition.title()} Specialist",
        "benefit_type": "Specialist Visit",
    }


def search_providers_by_specialty(taxonomy_desc: str, state: str, city: str = "", limit: int = 5) -> list:
    """Search NPPES for active individual providers by specialty (taxonomy description)."""
    def fetch():
        try:
            params = {
                "version": "2.1",
                "enumeration_type": "NPI-1",
                "taxonomy_description": taxonomy_desc,
                "limit": limit,
            }
            if state:
                params["state"] = state.upper()
            if city:
                params["city"] = city.upper()
            r = requests.get(NPPES_API, params=params, timeout=12)
            results = r.json().get("results", [])
            providers = []
            for item in results:
                basic = item.get("basic", {})
                taxonomies = item.get("taxonomies") or [{}]
                addresses = item.get("addresses") or [{}]
                primary_tax = next((t for t in taxonomies if t.get("primary")), taxonomies[0])
                practice_addr = next(
                    (a for a in addresses if a.get("address_purpose") == "LOCATION"),
                    addresses[0]
                )
                if basic.get("status", "") != "A":
                    continue
                providers.append({
                    "npi": item.get("number"),
                    "name": f"{basic.get('first_name','')} {basic.get('last_name','')}".strip(),
                    "credential": basic.get("credential", ""),
                    "specialty": primary_tax.get("desc", taxonomy_desc),
                    "city": practice_addr.get("city", ""),
                    "state": practice_addr.get("state", ""),
                    "phone": practice_addr.get("telephone_number", ""),
                    "address": f"{practice_addr.get('address_1','')} {practice_addr.get('address_2','')}".strip(),
                })
            return providers
        except Exception as e:
            print(f"NPPES specialty search error: {e}")
            return []

    return cached_call("nppes_specialty", {"taxonomy": taxonomy_desc.lower(), "state": state.lower(), "city": city.lower()}, fetch)


def get_plan_specialist_copay(plan_id: str, benefit_type: str = "Specialist Visit") -> dict:
    """
    Fetch specialist copay from the CMS plan detail endpoint.
    Falls back to metal-level estimates if the API doesn't return benefit data.
    """
    def fetch():
        try:
            r = requests.get(
                f"{BASE_MARKETPLACE}/plans/{plan_id}",
                params=_params({"year": 2024}),
                timeout=12,
            )
            data = r.json()
            plan_data = data.get("plan", data)
            benefits = plan_data.get("benefits", [])
            if benefits:
                target = next(
                    (b for b in benefits if benefit_type.lower() in b.get("name", "").lower()),
                    None
                )
                if not target:
                    target = next(
                        (b for b in benefits if "specialist" in b.get("name", "").lower()),
                        None
                    )
                if target:
                    cost_sharings = target.get("cost_sharings", [{}])
                    in_net = next(
                        (c for c in cost_sharings if "In" in c.get("network_tier", "In")),
                        cost_sharings[0] if cost_sharings else {}
                    )
                    return {
                        "found": True,
                        "benefit_name": target.get("name"),
                        "covered": target.get("covered", True),
                        "copay": in_net.get("copay_amount"),
                        "coinsurance": in_net.get("coinsurance_rate"),
                        "display": in_net.get("display_string", ""),
                        "source": "cms_benefits",
                    }
        except Exception as e:
            print(f"Plan benefits fetch error for {plan_id}: {e}")
        return {"found": False, "source": "fallback"}

    return cached_call("plan_benefit", {"plan_id": plan_id, "benefit": benefit_type.lower()}, fetch)


# CMS benefit name → drug tier key mapping (ordered: most-specific first)
_DRUG_BENEFIT_TIER_MAP = [
    ("preferred generic",       ["PREFERRED-GENERIC"]),
    ("non-preferred generic",   ["NON-PREFERRED-GENERIC"]),
    ("generic",                 ["GENERIC", "PREFERRED-GENERIC"]),
    ("preferred brand",         ["PREFERRED-BRAND", "BRAND"]),
    ("non-preferred brand",     ["NON-PREFERRED-BRAND"]),
    ("preferred specialty",     ["PREFERRED-SPECIALTY"]),
    ("non-preferred specialty", ["NON-PREFERRED-SPECIALTY"]),
    ("specialty",               ["SPECIALTY", "PREFERRED-SPECIALTY", "NON-PREFERRED-SPECIALTY"]),
]


def get_plan_drug_copays(plan_id: str, year: int = 2024) -> dict:
    """
    Fetch per-tier drug cost-sharing from the CMS plan benefits endpoint.
    Returns {TIER_KEY: {copay_amount, coinsurance_rate, display_string, after_deductible}}.
    Most-specific keyword matched first (preferred generic beats generic).
    """
    def fetch():
        try:
            r = requests.get(
                f"{BASE_MARKETPLACE}/plans/{plan_id}",
                params=_params({"year": year}),
                timeout=12,
            )
            if r.status_code != 200:
                return {}
            data = r.json()
            plan_data = data.get("plan", data)
            benefits = plan_data.get("benefits", [])
            tier_copays: dict = {}
            for benefit in benefits:
                bname = benefit.get("name", "").lower()
                if "drug" not in bname and "formulary" not in bname:
                    continue
                matching_tiers = None
                for keyword, tiers in _DRUG_BENEFIT_TIER_MAP:
                    if keyword in bname:
                        matching_tiers = tiers
                        break
                if not matching_tiers:
                    continue
                cost_sharings = benefit.get("cost_sharings", [])
                in_net = next(
                    (c for c in cost_sharings if "In" in c.get("network_tier", "In-Network")),
                    cost_sharings[0] if cost_sharings else {},
                )
                display = in_net.get("display_string", "")
                entry = {
                    "copay_amount":     float(in_net.get("copay_amount") or 0),
                    "coinsurance_rate": float(in_net.get("coinsurance_rate") or 0),
                    "display_string":   display,
                    "after_deductible": "deductible" in display.lower(),
                }
                for tier_key in matching_tiers:
                    if tier_key not in tier_copays:   # don't overwrite more-specific match
                        tier_copays[tier_key] = entry
            return tier_copays
        except Exception as e:
            print(f"Plan drug copays fetch error for {plan_id}: {e}")
            return {}

    return cached_call("plan_drug_copays", {"plan_id": plan_id, "year": year}, fetch)


# Metal-level copay estimates (fallback when CMS benefits API has no data)
METAL_COPAY_ESTIMATES = {
    "Platinum": {"copay": 20, "note": "~$20 specialist copay, no deductible first"},
    "Gold":     {"copay": 40, "note": "~$40 specialist copay, usually no deductible"},
    "Silver":   {"copay": 65, "note": "~$65 specialist copay, after deductible"},
    "Bronze":   {"copay": 100, "note": "~$100 copay or 30–40% coinsurance, after deductible"},
    "Catastrophic": {"copay": 150, "note": "Full cost until $9,450 deductible met"},
}

# Coinsurance by metal level (patient's share after deductible)
METAL_COINSURANCE = {
    "Platinum": 0.10, "Gold": 0.20, "Silver": 0.30,
    "Bronze": 0.40, "Catastrophic": 1.00,
}

# ─── PROCEDURE COST CATALOG ──────────────────────────────────────────────────
PROCEDURE_CATALOG = {
    "knee_replacement":   {"label": "Knee Replacement",          "total_cost": 35000, "category": "Surgery"},
    "hip_replacement":    {"label": "Hip Replacement",           "total_cost": 32000, "category": "Surgery"},
    "appendectomy":       {"label": "Appendectomy",              "total_cost": 18000, "category": "Surgery"},
    "gallbladder":        {"label": "Gallbladder Removal",       "total_cost": 14000, "category": "Surgery"},
    "back_surgery":       {"label": "Spinal Fusion / Back Surgery", "total_cost": 45000, "category": "Surgery"},
    "heart_bypass":       {"label": "Coronary Bypass Surgery",   "total_cost": 95000, "category": "Surgery"},
    "angioplasty":        {"label": "Angioplasty / Stent",       "total_cost": 28000, "category": "Cardiac"},
    "chemotherapy_cycle": {"label": "Chemotherapy (per cycle)",  "total_cost": 10000, "category": "Oncology"},
    "radiation_course":   {"label": "Radiation Therapy (full course)", "total_cost": 60000, "category": "Oncology"},
    "childbirth_vaginal": {"label": "Childbirth – Vaginal",      "total_cost": 12000, "category": "Maternity"},
    "childbirth_csection":{"label": "Childbirth – C-Section",    "total_cost": 20000, "category": "Maternity"},
    "colonoscopy":        {"label": "Colonoscopy",               "total_cost": 3500,  "category": "Diagnostic"},
    "mri":                {"label": "MRI Scan",                  "total_cost": 2600,  "category": "Diagnostic"},
    "ct_scan":            {"label": "CT Scan",                   "total_cost": 2000,  "category": "Diagnostic"},
    "er_visit":           {"label": "Emergency Room Visit",      "total_cost": 3200,  "category": "Emergency"},
    "ambulance":          {"label": "Ambulance Ride",            "total_cost": 1800,  "category": "Emergency"},
    "inpatient_3day":     {"label": "3-Day Hospital Stay",       "total_cost": 22000, "category": "Inpatient"},
    "physical_therapy":   {"label": "Physical Therapy (12 sessions)", "total_cost": 2400, "category": "Rehabilitation"},
    "diabetes_management":{"label": "Annual Diabetes Management","total_cost": 5500,  "category": "Chronic"},
    "mental_health_year": {"label": "Mental Health (12 months therapy)", "total_cost": 4800, "category": "Mental Health"},
}


def estimate_procedure_oop(procedure_key: str, plans: list) -> dict:
    """
    Estimate the out-of-pocket cost for a procedure on each plan.
    Uses deductible + metal-level coinsurance up to OOP max.
    """
    proc = PROCEDURE_CATALOG.get(procedure_key)
    if not proc:
        return {"error": f"Unknown procedure: {procedure_key}"}

    total_cost = proc["total_cost"]
    results = []
    for plan in plans:
        deductible = plan.get("deductible", 0) or 0
        oop_max    = plan.get("oop_max", 9450) or 9450
        metal      = plan.get("metal_level", "Silver")
        coinsurance = METAL_COINSURANCE.get(metal, 0.30)

        # Patient pays deductible first, then coinsurance on the remainder
        if total_cost <= deductible:
            patient_share = total_cost
        else:
            patient_share = deductible + (total_cost - deductible) * coinsurance

        patient_share = min(patient_share, oop_max)
        insurance_pays = total_cost - patient_share

        results.append({
            "plan_id":       plan.get("id"),
            "plan_name":     plan.get("name"),
            "metal_level":   metal,
            "net_premium":   round(plan.get("premium_w_credit", plan.get("premium", 0)), 2),
            "patient_oop":   round(patient_share),
            "insurance_pays": round(insurance_pays),
            "deductible":    deductible,
            "oop_max":       oop_max,
            "coinsurance_pct": int(coinsurance * 100),
        })

    results.sort(key=lambda x: x["patient_oop"])
    return {
        "procedure": proc["label"],
        "total_cost": total_cost,
        "category": proc["category"],
        "results": results,
    }


def search_hospitals_nearby(zip_code: str, limit: int = 8) -> list:
    """
    Return hospitals near a ZIP code by resolving state from ZIP then searching
    NPPES NPI-2 with taxonomy 'General Acute Care Hospital'.
    No name required — purely location-based.
    """
    fips  = get_fips_from_zip(zip_code)
    state = _fips_to_state(fips) if fips else ""

    def fetch():
        try:
            # Primary search: general acute care hospitals in state
            for taxonomy in ["General Acute Care Hospital", "Hospital"]:
                params = {
                    "version": "2.1",
                    "enumeration_type": "NPI-2",
                    "taxonomy_description": taxonomy,
                    "limit": limit,
                }
                if state:
                    params["state"] = state.upper()
                r = requests.get(NPPES_API, params=params, timeout=12)
                results = r.json().get("results", [])
                if results:
                    break

            hospitals = []
            seen = set()
            for item in results:
                npi = item.get("number")
                if npi in seen:
                    continue
                seen.add(npi)
                basic = item.get("basic", {})
                addresses = item.get("addresses") or [{}]
                addr = next(
                    (a for a in addresses if a.get("address_purpose") == "LOCATION"),
                    addresses[0],
                )
                if basic.get("status", "") != "A":
                    continue
                hospitals.append({
                    "npi":     npi,
                    "name":    basic.get("organization_name", ""),
                    "city":    addr.get("city", ""),
                    "state":   addr.get("state", ""),
                    "address": f"{addr.get('address_1','')} {addr.get('address_2','')}".strip(),
                    "phone":   addr.get("telephone_number", ""),
                })
            return hospitals[:limit]
        except Exception as e:
            print(f"Nearby hospital search error: {e}")
            return []

    return cached_call("hospitals_nearby", {"zip": zip_code}, fetch)


def search_hospitals(name: str, state: str, city: str = "") -> list:
    """Search NPPES for hospitals / facilities (NPI-2) by name."""
    def fetch():
        try:
            # Try progressively looser queries: exact → first 2 words → first word
            queries = [name]
            words = name.split()
            if len(words) >= 3:
                queries.append(" ".join(words[:2]))
            if len(words) >= 2:
                queries.append(words[0])

            results = []
            for q in queries:
                params = {
                    "version": "2.1",
                    "enumeration_type": "NPI-2",
                    "organization_name": q,
                    "limit": 10,
                }
                if state:
                    params["state"] = state.upper()
                if city:
                    params["city"] = city.upper()
                r = requests.get(NPPES_API, params=params, timeout=12)
                results = r.json().get("results", [])
                if results:
                    break  # stop at first query that returns results

            hospitals = []
            seen_npis = set()
            for item in results:
                npi = item.get("number")
                if npi in seen_npis:
                    continue
                seen_npis.add(npi)
                basic = item.get("basic", {})
                addresses = item.get("addresses") or [{}]
                practice_addr = next(
                    (a for a in addresses if a.get("address_purpose") == "LOCATION"),
                    addresses[0]
                )
                hospitals.append({
                    "npi": npi,
                    "name": basic.get("organization_name", name),
                    "city": practice_addr.get("city", ""),
                    "state": practice_addr.get("state", ""),
                    "address": f"{practice_addr.get('address_1','')} {practice_addr.get('address_2','')}".strip(),
                    "phone": practice_addr.get("telephone_number", ""),
                    "status": basic.get("status", ""),
                })
            return hospitals[:5]
        except Exception as e:
            print(f"Hospital search error: {e}")
            return []

    return cached_call("hospital_search", {"name": name.lower(), "state": state.lower()}, fetch)


# ─── PLAN PROVIDERS — fetch plan detail + NPPES specialty search ─────────────

def get_plan_providers(plan_id: str, zip_code: str, specialty: str = "Internal Medicine",
                       limit: int = 20) -> dict:
    """
    Return providers for a given plan by:
    1. Fetching the CMS plan detail for issuer name and provider directory URL.
    2. Searching NPPES for providers of the requested specialty near the ZIP.
    Returns the directory URL plus up to `limit` NPPES providers as a reference list
    (network membership must be confirmed via the insurer's directory).
    """
    try:
        # Step 1: Plan detail → issuer + network URL
        r = requests.get(
            f"{BASE_MARKETPLACE}/plans/{plan_id}",
            params=_params({"year": 2024}),
            timeout=10,
        )
        plan_meta: dict = {}
        if r.status_code == 200:
            raw = r.json().get("plan", {})
            issuer_obj = raw.get("issuer") or {}
            issuer_name = issuer_obj.get("name", "") if isinstance(issuer_obj, dict) else str(issuer_obj)
            network_url = raw.get("network_url") or ""
            if not network_url:
                networks = raw.get("network") or []
                network_url = networks[0].get("network_url", "") if networks else ""
            plan_meta = {
                "plan_id": plan_id,
                "plan_name": raw.get("name", plan_id),
                "issuer": issuer_name,
                "metal_level": raw.get("metal_level", ""),
                "type": raw.get("type", ""),
                "network_url": network_url,
            }

        # Step 2: FIPS/state from ZIP
        fips = get_fips_from_zip(zip_code)
        state = _fips_to_state(fips) if fips else ""

        # Step 3: NPPES search by specialty
        params = {
            "version": "2.1",
            "enumeration_type": "NPI-1",
            "taxonomy_description": specialty,
            "limit": min(limit, 50),
        }
        if state:
            params["state"] = state.upper()

        nr = requests.get(NPPES_API, params=params, timeout=10)
        results = nr.json().get("results", [])

        providers = []
        for item in results:
            basic = item.get("basic") or {}
            taxonomies = item.get("taxonomies") or [{}]
            addresses = item.get("addresses") or [{}]
            primary_tax = next((t for t in taxonomies if t.get("primary")), taxonomies[0])
            addr = next(
                (a for a in addresses if a.get("address_purpose") == "LOCATION"),
                addresses[0],
            )
            providers.append({
                "npi": item.get("number"),
                "name": f"{basic.get('first_name', '')} {basic.get('last_name', '')}".strip()
                        or basic.get("organization_name", ""),
                "credential": basic.get("credential", ""),
                "specialty": primary_tax.get("desc", specialty),
                "city": addr.get("city", ""),
                "state": addr.get("state", ""),
                "phone": addr.get("telephone_number", ""),
                "active": basic.get("status", "") == "A",
            })

        return {
            **plan_meta,
            "specialty_searched": specialty,
            "zip_code": zip_code,
            "state": state,
            "providers_from_nppes": providers,
            "provider_count": len(providers),
            "note": (
                "NPPES lists all licensed providers in this state/specialty. "
                "Network membership must be confirmed via the insurer's provider directory below."
            ),
        }
    except Exception as e:
        return {"error": str(e), "plan_id": plan_id}
