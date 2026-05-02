# CoverWise — AI Health Insurance Advisor

An agentic AI system that helps Americans find their optimal ACA health insurance plan by analyzing income, medications, and doctors against live government data.

**Live URL:** [your-cloud-run-url] ← update after deploy

---

## Run Locally

```bash
cp .env.example .env
# Fill in GOOGLE_CLOUD_PROJECT and CMS_API_KEY

# Authenticate with Google
gcloud auth application-default login
gcloud config set project agenticai-ritwik

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
# Visit http://localhost:8080
```

## Deploy to Cloud Run

```bash
gcloud run deploy coverwise --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=agenticai-ritwik,GOOGLE_CLOUD_REGION=us-central1,CMS_API_KEY=your_key
```

---

## Class Concepts Used

### 1. Multi-Agent Orchestration (`backend/agents/orchestrator.py`, `backend/agents/sub_agents.py`)
The `OrchestratorAgent` dispatches six specialized sub-agents simultaneously using `asyncio.gather()`: Profile, Plan Search, Subsidy, Drug Check, Doctor Check, and Risk & Gaps agents. Each agent owns a single domain and returns a typed result dict. The orchestrator merges all outputs before passing to the Gemini recommendation engine on Vertex AI.

### 2. Agent Handoff / Conditional Routing (`backend/agents/sub_agents.py` → `determine_route()`)
The Profile Agent makes an explicit routing decision based on the user's income as a % of Federal Poverty Level (FPL):
- **< 138% FPL** → hands off to Medicaid Agent (skips all marketplace agents entirely)
- **State-based exchange detected** → hands off to State Exchange redirect (e.g. NY, CA, MA)
- **138–400% FPL** → routes to Subsidy-eligible marketplace path (APTC eligible)
- **> 400% FPL** → routes to full-price marketplace path

This prevents running expensive API calls for users who qualify for free Medicaid or live in non-federal exchange states.

### 3. mem0 Persistent Memory (`backend/memory/mem0_client.py`)
After each session, user profile facts (ZIP, age, income, medications, doctors) are stored in mem0 backed by ChromaDB. On return visits, the orchestrator injects this memory context into the Gemini prompt so returning users skip the intake process entirely.

### 4. TTL-Based Caching (`backend/cache/cache_manager.py`)
All government API responses are cached with type-appropriate TTLs to reduce redundant calls and cost-to-serve:
- FPL tables: 1 year (published annually by HHS)
- ZIP→FIPS crosswalk: forever (never changes)
- RxNorm drug IDs: 30 days
- CMS plan data: 24 hours
- Doctor NPI lookups: 7 days

Reduces redundant API calls by ~70-80% for overlapping ZIP codes.

### 5. Human-in-the-Loop (`frontend/index.html`)
The user reviews and confirms their household profile before the recommendation agent synthesizes the final output. This prevents running the expensive parallel API fan-out on incorrect inputs.

### 6. Parallel Execution (`backend/agents/orchestrator.py`)
Two waves of parallel agent execution using `asyncio.gather()`:
- Wave 1: Subsidy, Plan Search, Doctor Check agents fire simultaneously
- Wave 2: Drug Check, Risk & Gaps, Metal Tier agents fire simultaneously (depend on Wave 1 output)

---

## Tech Stack
- **Backend:** FastAPI + Python
- **LLM:** Gemini 2.0 Flash via Vertex AI (Google Cloud)
- **Memory:** mem0 + ChromaDB
- **Deployment:** Google Cloud Run
- **APIs:** CMS Marketplace, RxNorm (NIH), NPPES NPI Registry, HHS FPL, openFDA

---

## Data Sources (All Free Government APIs)

| API | Used For | Key Required |
|---|---|---|
| CMS Marketplace API | Live plan search by ZIP/income | Free signup |
| CMS QHP Formulary files | Drug tiers per plan | No |
| HHS FPL table | Subsidy eligibility | No (static) |
| RxNorm (NIH NLM) | Drug name → RxCUI resolution | No |
| openFDA | Drug safety info | No |
| NPPES NPI Registry | Doctor verification | No |
| IRS applicable % table | APTC calculation | No (static) |
| CMS Medicaid/CHIP API | Free coverage eligibility | Free signup |

---

## Architecture

```
User Input (zip, age, income, household, drugs, doctors)
    ↓
Profile Agent → FPL calculation → HANDOFF ROUTING
    ├── State exchange detected → Redirect (NY, CA, MA, etc.)
    ├── < 138% FPL → Medicaid Agent (terminates here)
    └── ≥ 138% FPL → Marketplace path
            ↓ asyncio.gather() — parallel wave 1
    ┌─────────────────────────────────────────┐
    │ Plan Search │ Subsidy │ Doctor Check    │
    └─────────────────────────────────────────┘
            ↓ asyncio.gather() — parallel wave 2
    ┌─────────────────────────────────────────┐
    │ Drug Check │ Risk & Gaps │ Metal Tier  │
    └─────────────────────────────────────────┘
            ↓
    Recommendation Agent (Gemini 2.0 Flash via Vertex AI)
            ↓
    Human-in-the-loop confirmation
            ↓
    mem0 → store profile for next session
            ↓
    Final ranked recommendations + warnings
```
