# CoverWise — AI Health Insurance Advisor

Conversational AI agent that finds your optimal ACA health insurance plan.

**Live URL:** https://coverwise-272387131334.us-central1.run.app

## Run Locally

```bash
cp .env.example .env
# Add GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION, CMS_API_KEY

gcloud auth application-default login
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

## Deploy

```bash
gcloud run deploy coverwise --source . \
  --region us-central1 --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=agenticai-ritwik,GOOGLE_CLOUD_REGION=us-central1,CMS_API_KEY=your_key
```

## Class Concepts

### 1. Google ADK Agent Framework (`backend/agents/adk_orchestrator.py`, `intake_agent.py`)
Uses Google Agent Development Kit for structured multi-turn intake and deep insurance analysis. ADK provides proper session management, tool calling, and agent orchestration.

### 2. Multi-Agent Orchestration (`backend/agents/orchestrator.py`, `sub_agents.py`)
Orchestrator dispatches 6 specialized sub-agents in parallel using asyncio.gather(): Profile, Plan Search, Subsidy, Drug Check, Doctor Check, Risk & Gaps. Each owns a single domain.

### 3. Agent Handoff / Conditional Routing (`backend/agents/sub_agents.py`)
Profile Agent routes by FPL%: Medicaid (<138%), state exchange redirect, subsidized marketplace (138-400%), full price (>400%). Prevents wasted API calls for Medicaid-eligible users.

### 4. mem0 Persistent Memory (`backend/memory/mem0_client.py`)
User profile stored after each session via mem0+ChromaDB. Returning users get pre-filled intake — 8 questions reduced to 1-2. Memory injected into ADK agent context.

### 5. TTL-Based Caching (`backend/cache/cache_manager.py`)
All government API responses cached with type-appropriate TTLs: FPL 1yr, ZIP-FIPS forever, plans 24h, NPI 7d. Reduces redundant CMS API calls ~70-80%.

### 6. Human-in-the-Loop
Profile confirmation gate before analysis fires. User reviews collected ZIP, age, income, household, drugs, doctors and explicitly confirms before any CMS API calls run.

### 7. Parallel Execution (`backend/agents/orchestrator.py`)
Two waves of asyncio.gather(): Wave 1 (Subsidy + Plan Search + Doctor Check), Wave 2 (Drug Check + Risk & Gaps + Metal Tier). Total time = slowest agent, not sum.

## Data Sources

| API | Purpose | Key |
|---|---|---|
| CMS Marketplace API | Live plan search | Free signup |
| CMS Eligibility API | APTC subsidy calculation | Same key |
| RxNorm (NIH) | Drug name resolution | None |
| openFDA | Generic alternatives | None |
| NPPES NPI Registry | Doctor verification | None |
| CMS MIPS Catalog | Doctor quality scores | None |
| HRSA | Provider shortage areas | None |
| HHS FPL / IRS APT | Subsidy math | None (static) |
| Census ZCTA file | ZIP→FIPS (33k ZIPs) | None (bundled) |
