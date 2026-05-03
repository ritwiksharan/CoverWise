"""
Health Insurance Q&A Agent — Google ADK agent that answers natural language
questions about health insurance using live CMS, NPPES, RxNorm and HRSA data.

Scope: health insurance only. Off-topic questions (flood, auto, property,
stock analysis, etc.) receive a polite decline.
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

APP_NAME = "HealthInsuranceQA"

# ── Topic guard — classify before spending tool calls ────────────────────────

_HEALTH_INSURANCE_KEYWORDS = {
    # Plan types & structure
    "health insurance", "health plan", "marketplace", "aca", "obamacare",
    "hmo", "ppo", "epo", "hdhp", "catastrophic", "metal", "bronze", "silver",
    "gold", "platinum", "open enrollment", "sep", "special enrollment",
    # Benefits & costs
    "deductible", "copay", "coinsurance", "out-of-pocket", "oop", "premium",
    "aptc", "subsidy", "tax credit", "csr", "cost sharing", "actuarial",
    # Programs
    "medicaid", "medicare", "chip", "cobra", "tricare", "va coverage",
    "marketplace", "healthcare.gov", "exchange", "state exchange",
    # Drugs / providers
    "drug", "prescription", "formulary", "prior auth", "step therapy",
    "generic", "brand", "rxcui", "tier", "medication", "pharmacy",
    "doctor", "provider", "network", "in-network", "out-of-network",
    "specialist", "referral", "npi", "nppes", "primary care",
    # Finance / savings
    "hsa", "fsa", "hra", "flexible spending", "health savings",
    "subsidy cliff", "fpl", "federal poverty", "income",
    # Events / topics
    "life event", "marriage", "baby", "job loss", "enroll", "coverage",
    "claim", "explanation of benefits", "eob", "balance billing",
    "preexisting", "pre-existing", "aca", "affordable care",
    "employer plan", "group plan", "individual plan",
    # General
    "insurance", "insurer", "plan", "benefit", "coverage", "health",
}

_OFF_TOPIC_KEYWORDS = {
    "flood", "fema", "nfip", "hurricane", "earthquake", "wildfire",
    "auto insurance", "car insurance", "home insurance", "homeowners",
    "renters insurance", "property insurance", "liability insurance",
    "life insurance", "stock", "sec edgar", "ticker", "revenue",
    "earnings", "wall street", "invest",
}


def _is_health_insurance_question(question: str) -> bool:
    q = question.lower()
    # Hard reject clear off-topic signals
    for kw in _OFF_TOPIC_KEYWORDS:
        if kw in q:
            return False
    # Accept if any health-insurance keyword present
    for kw in _HEALTH_INSURANCE_KEYWORDS:
        if kw in q:
            return True
    # Ambiguous — let the agent decide (return True, agent will decline if needed)
    return True


# ── Health-insurance tool wrappers ───────────────────────────────────────────

def tool_search_health_plans(
    zip_code: str,
    age: int,
    income: float,
    household_size: int = 1,
    tobacco_use: bool = False,
) -> dict:
    """
    Search ACA Marketplace health plans for a given ZIP code, age, and income.
    Returns the top plans with premiums, deductibles, OOP max, metal level, and
    the estimated monthly APTC subsidy.

    Args:
        zip_code: 5-digit ZIP code.
        age: Applicant age in years.
        income: Annual household income in USD.
        household_size: Number of people in the household (default 1).
        tobacco_use: Whether the applicant uses tobacco (default False).
    """
    try:
        from tools.gov_apis import (
            get_fips_from_zip, _fips_to_state,
            get_subsidy_estimate, search_plans,
        )
        fips = get_fips_from_zip(zip_code)
        state = _fips_to_state(fips) if fips else "US"
        subsidy = get_subsidy_estimate(income, age, household_size, zip_code, tobacco_use)
        plans = search_plans(zip_code, age, income, fips, state, tobacco_use)
        monthly_credit = subsidy.get("monthly_aptc", 0)
        simplified = []
        for p in plans[:6]:
            net = max(0, p.get("premium", 0) - monthly_credit)
            simplified.append({
                "name": p.get("name"),
                "metal_level": p.get("metal_level"),
                "type": p.get("type"),
                "monthly_premium_after_subsidy": round(net, 2),
                "full_premium": p.get("premium"),
                "deductible": p.get("deductible"),
                "oop_max": p.get("oop_max"),
                "hsa_eligible": p.get("hsa_eligible"),
                "issuer": p.get("issuer"),
            })
        return {
            "zip_code": zip_code,
            "state": state,
            "monthly_aptc_subsidy": monthly_credit,
            "fpl_percentage": subsidy.get("fpl_percentage"),
            "csr_eligible": subsidy.get("csr_eligible"),
            "csr_variant": subsidy.get("csr_variant"),
            "is_medicaid_eligible": subsidy.get("is_medicaid_eligible"),
            "plans": simplified,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_get_subsidy_info(
    income: float,
    household_size: int,
    age: int,
    zip_code: str,
    tobacco_use: bool = False,
) -> dict:
    """
    Calculate ACA subsidy eligibility: APTC monthly credit, FPL percentage,
    CSR variant, and Medicaid eligibility for a given income and household.

    Args:
        income: Annual household income in USD.
        household_size: Number of people in the household.
        age: Primary applicant age.
        zip_code: 5-digit ZIP code (used to determine state).
        tobacco_use: Whether the applicant uses tobacco.
    """
    try:
        from tools.gov_apis import get_subsidy_estimate
        result = get_subsidy_estimate(income, age, household_size, zip_code, tobacco_use)
        # Add plain-English explanation
        fpl = result.get("fpl_percentage", 0)
        aptc = result.get("monthly_aptc", 0)
        csr = result.get("csr_variant")
        explanation_parts = []
        if result.get("is_medicaid_eligible"):
            explanation_parts.append(
                f"At {fpl:.0f}% FPL, this household likely qualifies for Medicaid (free or near-free coverage)."
            )
        elif fpl <= 400:
            explanation_parts.append(
                f"At {fpl:.0f}% FPL, this household qualifies for a ${aptc:.0f}/month APTC subsidy on marketplace plans."
            )
            if csr:
                csr_labels = {"94": "94% actuarial value (near-Platinum)", "87": "87% actuarial value (near-Gold)", "73": "73% actuarial value (enhanced Silver)"}
                explanation_parts.append(
                    f"CSR-{csr} eligible: choosing a Silver plan gives {csr_labels.get(csr, 'enhanced benefits')} at no extra premium cost."
                )
        else:
            explanation_parts.append(
                f"At {fpl:.0f}% FPL (above 400%), this household does not qualify for an APTC subsidy."
            )
        result["plain_english"] = " ".join(explanation_parts)
        return result
    except Exception as e:
        return {"error": str(e)}


def tool_lookup_drug_coverage(
    drug_name: str,
    zip_code: str,
    age: int = 35,
    income: float = 50000,
) -> dict:
    """
    Look up a drug's RxCUI, tier status, prior authorization requirements, and
    estimated monthly cost on top ACA marketplace plans for a given ZIP code.

    Args:
        drug_name: Brand or generic drug name (e.g. "Ozempic", "metformin").
        zip_code: 5-digit ZIP code.
        age: Applicant age (used to find relevant plans, default 35).
        income: Annual income in USD (used to find relevant plans, default 50000).
    """
    try:
        from tools.gov_apis import (
            resolve_drug_rxcui, search_plans, check_drug_coverage,
            get_generic_alternatives, get_fips_from_zip, _fips_to_state,
        )
        rxcui_info = resolve_drug_rxcui(drug_name)
        if not rxcui_info:
            return {"error": f"Could not resolve '{drug_name}' to an RxCUI. Check the spelling."}

        fips = get_fips_from_zip(zip_code)
        state = _fips_to_state(fips) if fips else "US"
        plans = search_plans(zip_code, age, income, fips, state)
        plan_ids = [p["id"] for p in plans[:5]]

        coverage = check_drug_coverage([rxcui_info["rxcui"]], plan_ids)
        generics = get_generic_alternatives(drug_name)

        tier_costs = {"Tier 1": "$0–$15/mo", "Tier 2": "$20–$50/mo", "Tier 3": "$50–$100/mo",
                      "Tier 4": "$100–$200/mo", "Tier 5": "$200+/mo (specialty)"}

        coverage_by_plan = []
        for c in coverage:
            plan_name = next((p["name"] for p in plans if p["id"] == c.get("plan_id")), c.get("plan_id"))
            tier = c.get("drug_tier", "Unknown")
            coverage_by_plan.append({
                "plan": plan_name,
                "coverage": c.get("coverage"),
                "tier": tier,
                "est_cost": tier_costs.get(tier, "varies"),
                "prior_auth_required": c.get("prior_authorization", False),
                "step_therapy_required": c.get("step_therapy", False),
            })

        return {
            "drug_name": rxcui_info.get("name", drug_name),
            "rxcui": rxcui_info.get("rxcui"),
            "strength": rxcui_info.get("strength"),
            "coverage_by_plan": coverage_by_plan,
            "generic_alternatives": [g["generic_name"] for g in generics if g.get("is_generic")][:3],
            "note": "Tier costs are estimates. Check your plan's Evidence of Coverage for exact amounts.",
        }
    except Exception as e:
        return {"error": str(e)}


def tool_find_local_specialists(
    condition: str,
    zip_code: str,
    state: str = "",
) -> dict:
    """
    Find local in-network specialists for a health condition near a ZIP code.
    Maps the condition to an NPPES taxonomy and returns provider names, NPIs,
    addresses, phone numbers, and MIPS quality scores.

    Args:
        condition: Health condition or specialty (e.g. "diabetes", "back pain", "cardiology").
        zip_code: 5-digit ZIP code.
        state: Optional 2-letter state code (derived from ZIP if omitted).
    """
    try:
        from tools.gov_apis import (
            map_condition_to_specialty, search_providers_by_specialty,
            get_doctor_quality_score, get_fips_from_zip, _fips_to_state,
        )
        if not state:
            fips = get_fips_from_zip(zip_code)
            state = _fips_to_state(fips) if fips else "US"

        specialty_info = map_condition_to_specialty(condition)
        providers = search_providers_by_specialty(
            specialty_info["taxonomy_desc"], state, limit=5
        )
        enriched = []
        for p in providers[:5]:
            quality = get_doctor_quality_score(str(p.get("npi", ""))) if p.get("npi") else {}
            enriched.append({
                **p,
                "mips_score": quality.get("mips_score"),
                "telehealth_available": quality.get("telehealth", False),
            })
        return {
            "condition": condition,
            "specialty": specialty_info["specialty"],
            "state": state,
            "providers": enriched,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_check_enrollment_period() -> dict:
    """
    Check whether ACA open enrollment is currently active, how many days remain,
    and what qualifies as a Special Enrollment Period (SEP) triggering event.
    Returns the current enrollment window status and SEP guidance.
    """
    try:
        from tools.gov_apis import check_sep_eligibility, check_hrsa_shortage
        sep = check_sep_eligibility()
        return {
            "open_enrollment_active": sep.get("in_open_enrollment", False),
            "days_remaining": sep.get("days_remaining"),
            "deadline": sep.get("deadline"),
            "message": sep.get("message"),
            "qualifying_life_events": [
                "Loss of job-based coverage",
                "Marriage or domestic partnership",
                "Birth or adoption of a child",
                "Move to a new ZIP code or state",
                "Gain of citizenship or lawful presence",
                "Release from incarceration",
                "Income change affecting Medicaid/CHIP eligibility",
            ],
            "sep_window": "60 days from the qualifying life event",
            "how_to_enroll": "Visit healthcare.gov or your state's exchange website.",
        }
    except Exception as e:
        return {"error": str(e)}


def tool_explain_health_insurance_concept(concept: str) -> dict:
    """
    Explain a health insurance term or concept clearly.
    Use this for questions like 'what is a deductible?', 'how does CSR work?',
    'what is the difference between HMO and PPO?', 'what is an HSA?', etc.

    Args:
        concept: The insurance term or concept to explain (e.g. "deductible",
                 "APTC", "HMO vs PPO", "HSA", "CSR", "step therapy").
    """
    glossary = {
        "deductible": "The amount you pay for covered health services before your insurance plan starts to pay. Example: with a $2,000 deductible, you pay the first $2,000 of covered services each year. After that, you usually pay only a copay or coinsurance.",
        "copay": "A fixed amount you pay for a covered health-care service after you've paid your deductible. Example: $30 for a doctor visit or $10 for a generic drug.",
        "coinsurance": "Your share of the costs of a covered health-care service, calculated as a percentage of the allowed amount. Example: 20% coinsurance means you pay 20% and insurance pays 80% after your deductible.",
        "out-of-pocket maximum": "The most you have to pay for covered services in a plan year. After you spend this amount on deductibles, copays, and coinsurance, your health plan pays 100% of covered services. In 2024 the federal cap is $9,450/individual.",
        "premium": "The amount you pay each month for your health insurance plan, regardless of whether you use medical services. Subsidies (APTC) reduce this amount for eligible marketplace enrollees.",
        "aptc": "Advance Premium Tax Credit — a federal subsidy that lowers your monthly marketplace health insurance premium. Eligibility is based on household income between 100%–400% of the Federal Poverty Level (FPL). The credit is paid directly to your insurer.",
        "csr": "Cost-Sharing Reduction — a discount that lowers the amount you pay out-of-pocket (deductible, copay, coinsurance) on Silver marketplace plans. Available to people with income 100%–250% FPL who enroll in Silver plans.",
        "hmo": "Health Maintenance Organization — a plan type that requires you to use a network of doctors and get referrals from your primary care doctor to see specialists. Generally lower premiums and simpler billing.",
        "ppo": "Preferred Provider Organization — a plan type that lets you see any doctor (in-network or out-of-network) without a referral, though in-network care costs less. More flexibility than HMO, typically higher premiums.",
        "epo": "Exclusive Provider Organization — like an HMO in that you must use the plan's network, but like a PPO in that you don't need referrals. No out-of-network coverage except emergencies.",
        "hdhp": "High-Deductible Health Plan — a plan with a higher deductible ($1,600+ for individuals in 2024) but lower premiums. Must be paired with an HSA for tax-advantaged savings.",
        "hsa": "Health Savings Account — a tax-advantaged savings account paired with an HDHP. Contributions are pre-tax, growth is tax-free, and withdrawals for medical expenses are tax-free. 2024 limit: $4,150/individual, $8,300/family. Funds roll over indefinitely.",
        "fsa": "Flexible Spending Account — a pre-tax account for medical expenses. Employer-sponsored; funds typically expire at year-end ('use it or lose it'). 2024 limit: $3,200.",
        "cobra": "Continuation coverage that lets you keep your employer's group health plan for up to 18–36 months after losing eligibility (job loss, hours reduction, divorce, etc.). You pay the full premium plus up to 2% admin fee.",
        "medicaid": "A joint federal/state program providing free or low-cost health coverage to people with low incomes. In states that expanded Medicaid under the ACA, individuals earning up to 138% FPL qualify.",
        "medicare": "Federal health insurance for people 65+, or younger people with certain disabilities. Part A covers hospital stays, Part B covers outpatient care. Parts C and D are optional private plans.",
        "fpl": "Federal Poverty Level — the income threshold set by the federal government used to determine eligibility for subsidies and programs. In 2024, the FPL is $15,060 for a single person. Marketplace subsidies are available from 100% to 400% FPL.",
        "formulary": "A list of prescription drugs covered by your health plan, organized into tiers. Tier 1 = cheapest generics; Tier 5 = expensive specialty drugs. Your plan's formulary determines your drug costs.",
        "prior authorization": "Approval from your health insurer that's required before you can receive certain medications, procedures, or services. If skipped, the insurer may not pay. Takes 1–30+ days depending on urgency.",
        "step therapy": "A cost-control rule requiring you to try less expensive medications first before your plan will cover a more expensive one. Example: must try generic metformin before Ozempic is covered.",
        "balance billing": "When an out-of-network provider bills you for the difference between their charge and what your insurer pays. Banned for emergency care under the No Surprises Act (2022).",
        "network": "The facilities, providers, and suppliers your health insurer has contracted with to provide health care services at negotiated rates. Staying in-network usually costs significantly less.",
        "hmo vs ppo": "HMO: lower cost, requires referrals, no out-of-network coverage, simpler. PPO: higher cost, no referrals needed, out-of-network allowed (at higher cost), more flexibility. Choose HMO if your doctors are in-network and you want lower premiums; choose PPO if you want flexibility.",
    }

    # Normalize lookup
    key = concept.lower().strip()
    # Try exact, then partial match
    answer = glossary.get(key)
    if not answer:
        for k, v in glossary.items():
            if key in k or k in key:
                answer = v
                key = k
                break

    if answer:
        return {"concept": key, "explanation": answer}
    # Return a prompt for the model to answer from its own knowledge
    return {
        "concept": concept,
        "explanation": None,
        "note": f"No glossary entry found for '{concept}'. Answer from your health insurance knowledge.",
    }


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a knowledgeable and friendly health insurance advisor for CoverWise.
You ONLY answer questions about health insurance and directly related topics.

TOPICS YOU COVER:
- ACA Marketplace plans (Bronze/Silver/Gold/Platinum/Catastrophic)
- Plan types: HMO, PPO, EPO, HDHP
- Costs: premiums, deductibles, copays, coinsurance, out-of-pocket maximums
- Subsidies: APTC, CSR, Medicaid, CHIP eligibility
- Drugs & formularies: coverage tiers, prior auth, step therapy, generics
- Providers: finding specialists, in-network vs out-of-network, NPI lookup
- Enrollment: open enrollment, Special Enrollment Periods, life events
- Savings accounts: HSA, FSA, HRA
- Programs: Medicare, Medicaid, COBRA, TRICARE
- General health insurance education and terminology

TOPICS YOU DO NOT COVER — politely decline these:
- Flood insurance, FEMA, natural disaster claims
- Auto, home, renters, life, or property insurance
- Insurer stock prices, SEC filings, earnings reports, investment analysis
- Anything unrelated to health insurance

TOOLS — use these only when you need LIVE data (plans, drugs, providers, enrollment):

1. tool_search_health_plans(zip_code, age, income, household_size)
   → Real-time ACA marketplace plans with net premiums and APTC subsidy.
   Use when: asked about specific plans or what's available in a location.

2. tool_get_subsidy_info(income, household_size, age, zip_code)
   → APTC dollar amount, FPL%, CSR variant, Medicaid eligibility.
   Use when: asked "how much subsidy do I get?" or "do I qualify for Medicaid?".

3. tool_lookup_drug_coverage(drug_name, zip_code, age, income)
   → Drug tier, prior auth requirements, step therapy, estimated monthly cost.
   Use when: asked about a specific medication's coverage or cost.

4. tool_find_local_specialists(condition, zip_code, state)
   → NPPES provider list with MIPS quality scores.
   Use when: asked to find doctors or specialists for a health condition.

5. tool_check_enrollment_period()
   → Whether open enrollment is active, deadline date, SEP trigger events.
   Use when: asked about enrollment windows, deadlines, or qualifying events.

WHEN TO USE TOOLS vs KNOWLEDGE:
- Conceptual questions ("what is a deductible?", "HMO vs PPO", "how does CSR work?",
  "what is step therapy?") → answer directly from your knowledge. Do NOT call a tool.
- Questions requiring live data ("what plans are available?", "is Ozempic covered?",
  "how much subsidy for $60k income?") → call the appropriate tool first.
- If a tool call needs ZIP/age/income that wasn't provided, use reasonable defaults:
  zip_code="33139", age=35, income=50000 — and note the assumption in your answer.

FOR OFF-TOPIC QUESTIONS: respond with exactly:
"I'm CoverWise's health insurance advisor. I can only help with health insurance questions — things like plan costs, drug coverage, subsidies, enrolling in coverage, and finding doctors. Could you ask me a health insurance question instead?"

STYLE: Be concise, warm, and specific. Use markdown. Cite numbers from tool data.
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class InsuranceQAAgent:
    def __init__(self):
        self._runner: Optional[Runner] = None
        self._session_service = InMemorySessionService() if ADK_AVAILABLE else None

    def _build_runner(self):
        if self._runner:
            return
        agent = Agent(
            name="health_insurance_advisor",
            model="gemini-2.0-flash",
            instruction=SYSTEM_PROMPT,
            tools=[
                tool_search_health_plans,
                tool_get_subsidy_info,
                tool_lookup_drug_coverage,
                tool_find_local_specialists,
                tool_check_enrollment_period,
            ],
        )
        self._runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=self._session_service,
        )

    async def ask(self, user_id: str, question: str) -> dict:
        if not ADK_AVAILABLE:
            return {"answer": "ADK not available.", "tool_calls": [], "error": "adk_unavailable"}

        # Fast keyword guard — return immediately for clearly off-topic questions
        if not _is_health_insurance_question(question):
            return {
                "answer": (
                    "I'm CoverWise's health insurance advisor. I can only help with health insurance "
                    "questions — things like ACA plan costs, drug coverage, subsidies, enrollment "
                    "windows, and finding doctors.\n\n"
                    "Try asking something like:\n"
                    "- *\"What plans are available in ZIP 33139 for a 35-year-old earning $45,000?\"*\n"
                    "- *\"How does the APTC subsidy work?\"*\n"
                    "- *\"Is Ozempic covered on Silver plans?\"*"
                ),
                "tool_calls": [],
                "error": None,
                "off_topic": True,
            }

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
            pass  # session already exists — that's fine

        msg = Content(role="user", parts=[Part(text=question)])
        answer = ""
        tool_calls: list[str] = []

        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=msg,
        ):
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
            "tool_calls": list(dict.fromkeys(tool_calls)),
            "error": None,
        }


_agent: Optional[InsuranceQAAgent] = None


def get_agent() -> InsuranceQAAgent:
    global _agent
    if _agent is None:
        _agent = InsuranceQAAgent()
    return _agent
