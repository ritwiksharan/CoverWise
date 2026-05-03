"""
CoverWise Insurance MCP Server

Exposes four tools to Claude / MCP clients via stdio transport:
  - query_flood_claims   — FEMA NFIP flood claims aggregation
  - query_insurer_financials — SEC EDGAR XBRL financials
  - risk_score           — ZIP-level flood risk scoring
  - compare_insurers     — Side-by-side insurer comparison
"""

import sys
import os

# Ensure the backend package root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("insurance-mcp-server")


@mcp.tool()
def query_flood_claims(
    state: str,
    county: str = "",
    start_year: int = 2020,
    end_year: int = 2024,
    flood_zone: str = "",
    limit: int = 100,
) -> dict:
    """
    Search and filter FEMA NFIP flood insurance claims by state, county,
    date range, and flood zone.

    Returns aggregate statistics: total claims, total paid amounts, average
    building claim, breakdown by year and flood zone, top ZIP codes, and
    a small sample of raw claim records.

    Args:
        state: Two-letter US state code (e.g. "TX", "FL").
        county: Optional county code substring to filter by.
        start_year: Earliest year of loss to include (default 2020).
        end_year: Latest year of loss to include (default 2024).
        flood_zone: Optional flood zone code (e.g. "AE", "X").
        limit: Maximum number of claim rows to retrieve (default 100).
    """
    from tools.insurance_mcp_tools import query_flood_claims as _fn
    return _fn(
        state=state,
        county=county,
        start_year=start_year,
        end_year=end_year,
        flood_zone=flood_zone,
        limit=limit,
    )


@mcp.tool()
def query_insurer_financials(
    ticker: str,
    metrics: list | None = None,
    years: int = 3,
) -> dict:
    """
    Retrieve annual financial data for a publicly traded insurance company
    from SEC EDGAR XBRL filings (10-K).

    Returns revenues, net income, total assets, total liabilities, and an
    estimated combined ratio for each of the most recent annual filings.

    Supported tickers include: ALL, PGR, TRV, CB, MET, HIG, CI, UNH, CVS,
    ELV, HUM, AFL, PRU, LNC, UNM, GL, RLI, CINF.

    Args:
        ticker: Stock ticker symbol (e.g. "PGR" for Progressive).
        metrics: GAAP metric names to fetch (default: Revenues, NetIncomeLoss,
                 Assets, Liabilities).
        years: Number of most-recent annual periods to return (default 3).
    """
    from tools.insurance_mcp_tools import query_insurer_financials as _fn
    return _fn(ticker=ticker, metrics=metrics, years=years)


@mcp.tool()
def risk_score(zip_code: str, state: str = "") -> dict:
    """
    Calculate a flood risk score (0–100) for a specific ZIP code based on
    10 years of FEMA NFIP claim history.

    Risk levels:
      0       → Low
      1–40    → Medium
      41–70   → High
      71–100  → Extreme

    Score is boosted by +10 if high-risk flood zones (A or V series) are
    present in the claims data.

    Args:
        zip_code: 5-digit US ZIP code (e.g. "77002").
        state: Optional two-letter state code to provide context.
    """
    from tools.insurance_mcp_tools import risk_score as _fn
    return _fn(zip_code=zip_code, state=state)


@mcp.tool()
def compare_insurers(tickers: list) -> dict:
    """
    Compare up to 4 insurance companies side by side using their most recent
    SEC EDGAR annual financials.

    Returns a structured comparison with revenues (in $B), net income (in $B),
    profit margin %, total assets (in $B), and auto-generated comparison notes
    highlighting key differences.

    Args:
        tickers: List of up to 4 stock ticker symbols
                 (e.g. ["ALL", "PGR", "TRV"]).
    """
    from tools.insurance_mcp_tools import compare_insurers as _fn
    return _fn(tickers=tickers)


if __name__ == "__main__":
    mcp.run()
