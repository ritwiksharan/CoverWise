"""
Insurance Q&A Agent — Google ADK agent with MCP tool functions registered.

Gemini 2.0 Flash reads the natural-language question, decides which tool(s)
to call (and with what arguments), calls them, then synthesises a response.
No keyword routing — the model drives everything.
"""

import sys
import os
import asyncio
import traceback
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

from tools.insurance_mcp_tools import (
    query_flood_claims,
    query_insurer_financials,
    risk_score,
    compare_insurers,
)

APP_NAME = "InsuranceQA"

SYSTEM_PROMPT = """\
You are an expert insurance data analyst with access to live public data from:
- FEMA's National Flood Insurance Program (NFIP) — real flood claims, disaster declarations
- SEC EDGAR — annual financial filings (10-K/10-Q) for publicly traded insurers

You have four tools. Call whichever are needed to answer the user's question accurately.

TOOLS:
1. query_flood_claims(state, county, start_year, end_year, flood_zone, limit)
   → FEMA NFIP claims aggregated by year and flood zone, with totals and sample rows.
   Use for: flood history, claims volume, payout data, disaster queries, county/state trends.

2. query_insurer_financials(ticker, years)
   → SEC EDGAR XBRL data: revenues, net income, assets, liabilities, combined ratio estimate.
   Supported tickers: ALL, PGR, TRV, CB, MET, HIG, CI, UNH, CVS, ELV, HUM, AFL, PRU, LNC, UNM, GL, RLI, CINF.
   Use for: earnings, revenue, profitability, balance sheet, annual report questions.

3. risk_score(zip_code, state)
   → Flood risk score (0–100) with level (Low/Medium/High/Extreme) from 10 years of FEMA data.
   Use for: "is this ZIP risky?", flood risk assessment, insurance premium context.

4. compare_insurers(tickers)
   → Side-by-side comparison of up to 4 insurers with auto-generated analysis notes.
   Use for: "compare X vs Y", competitive analysis, ranking insurers.

RULES:
- ALWAYS call at least one tool before answering — never answer from prior knowledge alone.
- If the question mentions a specific ZIP code → call risk_score.
- If the question compares companies → call compare_insurers.
- If the question names a single insurer ticker or company → call query_insurer_financials.
- If the question is about floods, FEMA, claims, disasters, or a state/county → call query_flood_claims.
- You may call multiple tools if the question requires it.
- In your final answer: use markdown, cite specific numbers from the tool data, and be concise.
"""


# ── ADK tool wrappers (sync functions — ADK calls them in a thread) ───────────

def tool_query_flood_claims(
    state: str,
    county: str = "",
    start_year: int = 2020,
    end_year: int = 2024,
    flood_zone: str = "",
    limit: int = 100,
) -> dict:
    """Search FEMA NFIP flood claims. state: 2-letter code (e.g. 'TX'). Returns aggregated stats."""
    return query_flood_claims(
        state=state, county=county,
        start_year=start_year, end_year=end_year,
        flood_zone=flood_zone, limit=limit,
    )


def tool_query_insurer_financials(ticker: str, years: int = 3) -> dict:
    """Get SEC EDGAR annual financials for an insurer. ticker: e.g. 'PGR', 'ALL', 'TRV'."""
    return query_insurer_financials(ticker=ticker, years=years)


def tool_risk_score(zip_code: str, state: str = "") -> dict:
    """Flood risk score (0-100) for a ZIP code from 10 years of FEMA data."""
    return risk_score(zip_code=zip_code, state=state)


def tool_compare_insurers(tickers: list) -> dict:
    """Side-by-side SEC EDGAR financial comparison of up to 4 insurers. tickers: e.g. ['ALL','PGR']."""
    return compare_insurers(tickers=tickers)


# ── Agent ─────────────────────────────────────────────────────────────────────

class InsuranceQAAgent:
    def __init__(self):
        self._runner: Optional[Runner] = None
        self._session_service = InMemorySessionService() if ADK_AVAILABLE else None

    def _build_runner(self):
        if self._runner:
            return
        agent = Agent(
            name="insurance_analyst",
            model="gemini-2.0-flash",
            instruction=SYSTEM_PROMPT,
            tools=[
                tool_query_flood_claims,
                tool_query_insurer_financials,
                tool_risk_score,
                tool_compare_insurers,
            ],
        )
        self._runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=self._session_service,
        )

    async def ask(self, user_id: str, question: str) -> dict:
        """
        Process a natural-language insurance question.
        Returns {answer, tool_calls, error}.
        """
        if not ADK_AVAILABLE:
            return {"answer": "ADK not available.", "tool_calls": [], "error": "adk_unavailable"}

        self._build_runner()
        session_id = f"qa_{user_id}"

        try:
            await self._session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state={},
            )
        except Exception:
            pass  # session already exists

        msg = Content(role="user", parts=[Part(text=question)])
        answer = ""
        tool_calls: list[str] = []

        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=msg,
        ):
            # Capture tool call names for transparency
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        tool_calls.append(part.function_call.name)

            if hasattr(event, "is_final_response") and event.is_final_response():
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            answer += part.text

        return {
            "answer": answer,
            "tool_calls": list(dict.fromkeys(tool_calls)),  # dedupe, preserve order
            "error": None,
        }


# Module-level singleton
_agent: Optional[InsuranceQAAgent] = None


def get_agent() -> InsuranceQAAgent:
    global _agent
    if _agent is None:
        _agent = InsuranceQAAgent()
    return _agent
