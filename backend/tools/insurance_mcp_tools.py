"""
Insurance MCP Tools — FEMA + SEC EDGAR data layer.

Provides four callable functions used by both the MCP server and the
/api/insurance-qa FastAPI endpoint.
"""

import json
import requests
from collections import defaultdict

HEADERS = {"User-Agent": "bhuvighosh3@gmail.com CoverWise/1.0"}
TIMEOUT = 10

# ── Known CIK map (zero-padded to 10 digits) ─────────────────────────────────
_KNOWN_CIKS: dict[str, str] = {
    "ALL":  "0000899051",
    "PGR":  "0000080661",
    "TRV":  "0000086312",
    "CB":   "0000896159",
    "MET":  "0001099219",
    "HIG":  "0000874766",
    "CI":   "0001739940",
    "UNH":  "0000072971",
    "CVS":  "0000064803",
    "ELV":  "0001156039",
    "HUM":  "0000049071",
    "AFL":  "0000004977",
    "PRU":  "0001137774",
    "LNC":  "0000059558",
    "UNM":  "0000100122",
    "GL":   "0000036987",
    "RLI":  "0000730469",
    "CINF": "0000020286",
}


def _lookup_cik(ticker: str) -> str | None:
    """Return 10-digit CIK for ticker. Checks hardcoded map first, then EDGAR."""
    t = ticker.upper().strip()
    if t in _KNOWN_CIKS:
        return _KNOWN_CIKS[t]
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        for entry in r.json().values():
            if entry.get("ticker", "").upper() == t:
                cik_raw = str(entry["cik_str"])
                return cik_raw.zfill(10)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. query_flood_claims
# ─────────────────────────────────────────────────────────────────────────────

def query_flood_claims(
    state: str,
    county: str = "",
    start_year: int = 2020,
    end_year: int = 2024,
    flood_zone: str = "",
    limit: int = 100,
) -> dict:
    """
    Search FEMA NFIP flood insurance claims by state, county, date range,
    and flood zone.

    Returns aggregated statistics plus a sample of raw claim rows.
    """
    try:
        # Build OData $filter
        filters = [f"state eq '{state.upper()}'"]
        filters.append(f"yearOfLoss ge {start_year}")
        filters.append(f"yearOfLoss le {end_year}")
        if county:
            filters.append(f"contains(countyCode, '{county.upper()}')")
        if flood_zone:
            filters.append(f"floodZoneCurrent eq '{flood_zone.upper()}'")

        params = {
            "$filter": " and ".join(filters),
            "$top": str(limit),
            "$orderby": "yearOfLoss desc",
            "$select": (
                "state,countyCode,yearOfLoss,"
                "amountPaidOnBuildingClaim,amountPaidOnContentsClaim,"
                "floodZoneCurrent,reportedCity,reportedZipCode"
            ),
            "$format": "json",
        }

        r = requests.get(
            "https://www.fema.gov/api/open/v2/FimaNfipClaims",
            params=params,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        claims = r.json().get("FimaNfipClaims", [])

        # Aggregations
        total_building = 0.0
        total_contents = 0.0
        by_year: dict[int, dict] = defaultdict(lambda: {"count": 0, "building": 0.0, "contents": 0.0})
        by_zone: dict[str, dict] = defaultdict(lambda: {"count": 0, "building": 0.0, "contents": 0.0})
        zip_counts: dict[str, int] = defaultdict(int)
        high_risk_zones = {"A", "V", "AE", "VE", "AH", "AO", "AR", "A99"}

        for c in claims:
            bldg = float(c.get("amountPaidOnBuildingClaim") or 0)
            cont = float(c.get("amountPaidOnContentsClaim") or 0)
            yr = c.get("yearOfLoss")
            zone = (c.get("floodZoneCurrent") or "Unknown").upper()
            zipcode = c.get("reportedZipCode") or "Unknown"

            total_building += bldg
            total_contents += cont

            if yr:
                by_year[int(yr)]["count"] += 1
                by_year[int(yr)]["building"] += bldg
                by_year[int(yr)]["contents"] += cont

            by_zone[zone]["count"] += 1
            by_zone[zone]["building"] += bldg
            by_zone[zone]["contents"] += cont

            zip_counts[zipcode] += 1

        n = len(claims)
        avg_building = (total_building / n) if n else 0.0

        # Top 5 ZIP codes by claim count
        top_zips = sorted(zip_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Check for high-risk flood zone exposure
        has_high_risk = any(
            z.split("-")[0].strip() in high_risk_zones for z in by_zone
        )

        return {
            "total_claims": n,
            "total_building_paid": round(total_building, 2),
            "total_contents_paid": round(total_contents, 2),
            "avg_building_claim": round(avg_building, 2),
            "top_zip_codes": [{"zip": z, "claims": cnt} for z, cnt in top_zips],
            "by_year": {
                yr: {
                    "count": v["count"],
                    "total_building_paid": round(v["building"], 2),
                    "total_contents_paid": round(v["contents"], 2),
                }
                for yr, v in sorted(by_year.items())
            },
            "by_flood_zone": {
                zone: {
                    "count": v["count"],
                    "total_building_paid": round(v["building"], 2),
                }
                for zone, v in sorted(by_zone.items(), key=lambda x: x[1]["count"], reverse=True)
            },
            "has_high_risk_zones": has_high_risk,
            "sample_claims": claims[:5],
            "filters_applied": {
                "state": state.upper(),
                "county": county or None,
                "start_year": start_year,
                "end_year": end_year,
                "flood_zone": flood_zone or None,
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. query_insurer_financials
# ─────────────────────────────────────────────────────────────────────────────

def query_insurer_financials(
    ticker: str,
    metrics: list | None = None,
    years: int = 3,
) -> dict:
    """
    Fetch SEC EDGAR XBRL financial facts for an insurance company.

    Extracts Revenues, NetIncomeLoss, Assets, Liabilities from 10-K filings
    for the last `years` annual periods.
    """
    if metrics is None:
        metrics = ["Revenues", "NetIncomeLoss", "Assets", "Liabilities"]

    try:
        cik = _lookup_cik(ticker)
        if not cik:
            return {"error": f"Could not find CIK for ticker '{ticker}'"}

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        facts_data = r.json()

        entity_name = facts_data.get("entityName", ticker.upper())
        us_gaap = facts_data.get("facts", {}).get("us-gaap", {})

        # For each metric, collect annual 10-K entries
        raw: dict[str, dict[str, float]] = {}  # {year_str: {metric: value}}
        for metric in metrics:
            metric_data = us_gaap.get(metric, {})
            usd_entries = metric_data.get("units", {}).get("USD", [])
            for entry in usd_entries:
                form = entry.get("form", "")
                end_date = entry.get("end", "")
                val = entry.get("val")
                if form == "10-K" and end_date and val is not None:
                    year = end_date[:4]
                    if year not in raw:
                        raw[year] = {}
                    # Keep latest entry per year (last seen wins)
                    raw[year][metric] = float(val)

        # Sort years descending and take `years` most recent
        sorted_years = sorted(raw.keys(), reverse=True)[:years]

        financials = []
        for yr in sorted(sorted_years):  # ascending for display
            row = raw[yr]
            revenues = row.get("Revenues", 0.0)
            net_income = row.get("NetIncomeLoss", 0.0)
            assets = row.get("Assets", 0.0)
            liabilities = row.get("Liabilities", 0.0)
            # Estimate combined ratio: 1 - (net_income / revenues) if revenues > 0
            combined_ratio_est = None
            if revenues and revenues > 0:
                combined_ratio_est = round(1 - (net_income / revenues), 4)
            financials.append({
                "year": int(yr),
                "revenues": revenues,
                "net_income": net_income,
                "assets": assets,
                "liabilities": liabilities,
                "combined_ratio_est": combined_ratio_est,
            })

        return {
            "ticker": ticker.upper(),
            "company_name": entity_name,
            "cik": cik,
            "financials": financials,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. risk_score
# ─────────────────────────────────────────────────────────────────────────────

def risk_score(zip_code: str, state: str = "") -> dict:
    """
    Calculate a flood risk score (0–100) for a ZIP code using FEMA NFIP data
    from the last 10 years.
    """
    try:
        current_year = 2024
        start_year = current_year - 10

        filters = [f"reportedZipCode eq '{zip_code}'"]
        filters.append(f"yearOfLoss ge {start_year}")

        params = {
            "$filter": " and ".join(filters),
            "$top": "500",
            "$select": (
                "state,yearOfLoss,amountPaidOnBuildingClaim,"
                "amountPaidOnContentsClaim,floodZoneCurrent,reportedZipCode"
            ),
            "$format": "json",
        }

        r = requests.get(
            "https://www.fema.gov/api/open/v2/FimaNfipClaims",
            params=params,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        claims = r.json().get("FimaNfipClaims", [])

        n = len(claims)
        total_paid = 0.0
        flood_zones: set[str] = set()
        detected_state = state

        for c in claims:
            bldg = float(c.get("amountPaidOnBuildingClaim") or 0)
            cont = float(c.get("amountPaidOnContentsClaim") or 0)
            total_paid += bldg + cont
            zone = (c.get("floodZoneCurrent") or "").upper()
            if zone:
                flood_zones.add(zone)
            if not detected_state:
                detected_state = c.get("state", "")

        avg_claim = (total_paid / n) if n else 0.0

        # Base score by claim count
        if n == 0:
            base_score = 0
        elif n <= 5:
            base_score = 20
        elif n <= 20:
            base_score = 40
        elif n <= 50:
            base_score = 60
        elif n <= 100:
            base_score = 80
        else:
            base_score = 100

        # Adjust +10 for high-risk zones (A or V prefix)
        high_risk_zones = {"A", "V", "AE", "VE", "AH", "AO", "AR", "A99"}
        zone_bonus = 10 if any(
            z.split("-")[0].strip() in high_risk_zones for z in flood_zones
        ) else 0
        score = min(100, base_score + zone_bonus)

        # Risk level label
        if score == 0:
            risk_level = "Low"
        elif score <= 40:
            risk_level = "Medium"
        elif score <= 70:
            risk_level = "High"
        else:
            risk_level = "Extreme"

        explanation = (
            f"ZIP {zip_code} had {n} flood insurance claims in the last 10 years "
            f"totaling ${total_paid:,.0f} in paid claims. "
        )
        if flood_zones:
            explanation += f"Flood zones present: {', '.join(sorted(flood_zones))}. "
        if zone_bonus:
            explanation += "High-risk A/V flood zones detected — +10 risk adjustment applied. "
        if n == 0:
            explanation += "No NFIP claims found — low historical flood risk."

        return {
            "zip_code": zip_code,
            "state": detected_state.upper() if detected_state else "",
            "risk_score": score,
            "total_claims": n,
            "total_paid": round(total_paid, 2),
            "avg_claim": round(avg_claim, 2),
            "flood_zones_affected": sorted(flood_zones),
            "risk_level": risk_level,
            "explanation": explanation,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. compare_insurers
# ─────────────────────────────────────────────────────────────────────────────

def compare_insurers(tickers: list) -> dict:
    """
    Side-by-side financial comparison of up to 4 insurance companies using
    SEC EDGAR data. Produces comparison notes highlighting key differences.
    """
    tickers = [t.upper().strip() for t in tickers[:4]]
    companies = []

    for ticker in tickers:
        result = query_insurer_financials(ticker, years=3)
        if "error" in result:
            companies.append({
                "ticker": ticker,
                "name": ticker,
                "error": result["error"],
            })
            continue

        financials = result.get("financials", [])
        if not financials:
            companies.append({
                "ticker": ticker,
                "name": result.get("company_name", ticker),
                "error": "No financial data found",
            })
            continue

        # Use most recent year
        latest = max(financials, key=lambda x: x["year"])
        revenues_b = round(latest["revenues"] / 1e9, 2) if latest["revenues"] else 0.0
        net_income_b = round(latest["net_income"] / 1e9, 2) if latest["net_income"] else 0.0
        assets_b = round(latest["assets"] / 1e9, 2) if latest["assets"] else 0.0
        profit_margin_pct = (
            round(latest["net_income"] / latest["revenues"] * 100, 1)
            if latest["revenues"] and latest["revenues"] > 0
            else None
        )

        companies.append({
            "ticker": ticker,
            "name": result.get("company_name", ticker),
            "latest_year": latest["year"],
            "revenues_b": revenues_b,
            "net_income_b": net_income_b,
            "profit_margin_pct": profit_margin_pct,
            "assets_b": assets_b,
            "combined_ratio_est": latest.get("combined_ratio_est"),
        })

    # Generate comparison notes
    notes: list[str] = []
    valid = [c for c in companies if "error" not in c]

    if len(valid) >= 2:
        # Profit margin comparison
        margin_sorted = sorted(
            [c for c in valid if c.get("profit_margin_pct") is not None],
            key=lambda x: x["profit_margin_pct"],
            reverse=True,
        )
        if len(margin_sorted) >= 2:
            top = margin_sorted[0]
            second = margin_sorted[1]
            diff = round(top["profit_margin_pct"] - second["profit_margin_pct"], 1)
            if diff > 0:
                notes.append(
                    f"{top['name']} ({top['ticker']}) has a {diff}% higher profit margin "
                    f"than {second['name']} ({second['ticker']}) "
                    f"({top['profit_margin_pct']}% vs {second['profit_margin_pct']}%)."
                )

        # Revenue comparison
        rev_sorted = sorted(
            [c for c in valid if c.get("revenues_b")],
            key=lambda x: x["revenues_b"],
            reverse=True,
        )
        if rev_sorted:
            notes.append(
                f"Largest by revenue: {rev_sorted[0]['name']} "
                f"(${rev_sorted[0]['revenues_b']}B). "
                + (
                    f"Smallest: {rev_sorted[-1]['name']} (${rev_sorted[-1]['revenues_b']}B)."
                    if len(rev_sorted) > 1
                    else ""
                )
            )

        # Assets comparison
        assets_sorted = sorted(
            [c for c in valid if c.get("assets_b")],
            key=lambda x: x["assets_b"],
            reverse=True,
        )
        if assets_sorted:
            notes.append(
                f"Largest balance sheet: {assets_sorted[0]['name']} "
                f"with ${assets_sorted[0]['assets_b']}B in total assets."
            )

        # Combined ratio note
        cr_available = [c for c in valid if c.get("combined_ratio_est") is not None]
        if cr_available:
            best_cr = min(cr_available, key=lambda x: x["combined_ratio_est"])
            notes.append(
                f"Best estimated combined ratio: {best_cr['name']} "
                f"at {round(best_cr['combined_ratio_est'] * 100, 1)}% "
                f"(lower is better for underwriting profitability)."
            )

    return {
        "companies": companies,
        "comparison_notes": notes,
    }
