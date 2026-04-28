# CoverWise — AI Health Insurance Advisor

An agentic AI system that helps Americans find their optimal ACA health insurance plan by analyzing income, medications, and doctors against live government data.

**Live URL:** [your-cloud-run-url]

---

## Run Locally

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY and CMS_API_KEY

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
# Visit http://localhost:8080
```

## Deploy to Cloud Run

```bash
gcloud run deploy coverwise --source . \
  --set-env-vars ANTHROPIC_API_KEY=xxx,CMS_API_KEY=xxx \
  --allow-unauthenticated --region us-central1
```

---

## Class Concepts Used

### 1. Multi-Agent Orchestration (`backend/agents/orchestrator.py`, `backend/agents/sub_agents.py`)
The `OrchestratorAgent` dispatches six specialized sub-agents simultaneously using `asyncio.gather()`: Profile, Plan Search, Subsidy, Drug Check, Doctor Check, and Risk & Gaps agents. Each agent owns a single domain and returns a typed result dict. The orchestrator merges all outputs before passing to the LLM recommendation engine.

### 2. Agent Handoff / Conditional Routing (`backend/agents/sub_agents.py` → `determine_route()`)
The Profile Agent makes an explicit routing decision based on the user's income as a percentage of the Federal Poverty Level:
- **< 138% FPL** → hands off to Medicaid Agent (skips all marketplace agents entirely)
- **138–400% FPL** → routes to Subsidy-eligible marketplace path
- **> 400% FPL** → routes to full-price marketplace path

This prevents running expensive API calls for users who qualify for free Medicaid. The handoff terminates early and returns immediately with Medicaid enrollment instructions.

### 3. mem0 Persistent Memory (`backend/memory/mem0_client.py`)
After each analysis session, the user's profile facts (ZIP, age, income, medications, doctors) are stored in mem0 backed by ChromaDB. On return visits, the orchestrator injects this memory context into the LLM system prompt so returning users skip the intake process. Uses semantic search to retrieve relevant memories per query.

### 4. TTL-Based Caching (`backend/cache/cache_manager.py`)
All government API responses are cached with type-appropriate TTLs:
- FPL tables: 1 year (published annually by HHS)
- ZIP→FIPS crosswalk: forever (never changes)
- RxNorm drug IDs: 30 days (very stable)
- CMS plan data: 24 hours
- Doctor NPI lookups: 7 days

This reduces redundant API calls by ~70–80% for overlapping ZIP codes and directly reduces cost-to-serve per user.

### 5. Human-in-the-Loop (`frontend/index.html` → confirmation flow)
Before the recommendation agent synthesizes the final output, the user reviews and confirms their household profile. This prevents running the expensive parallel API fan-out on incorrect inputs — a key cost control and accuracy mechanism.

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
    ├── < 138% FPL → Medicaid Agent (terminates here)
    └── ≥ 138% FPL → Marketplace path
            ↓ asyncio.gather() — parallel
    ┌─────────────────────────────────────────┐
    │ Plan Search │ Subsidy │ Doctor Check    │
    └─────────────────────────────────────────┘
            ↓ second parallel wave
    ┌─────────────────────────────────────────┐
    │ Drug Check │ Risk & Gaps │ Metal Tier  │
    └─────────────────────────────────────────┘
            ↓
    Recommendation Agent (Claude Sonnet)
            ↓
    Human-in-the-loop confirmation
            ↓
    mem0 → store profile for next session
            ↓
    Final ranked recommendations + warnings
```
