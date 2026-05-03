# CoverWise — AI Health Insurance Advisor

An agentic AI system that helps Americans find their optimal ACA health insurance plan by analyzing income, medications, and doctors against live government data sources — all in one place, for free.

**Live URL:** https://coverwise-272387131334.us-central1.run.app

---

## What It Does

You fill out a short form (ZIP, age, income, household size, medications, doctors). CoverWise runs a parallel multi-agent analysis against six government APIs, then Gemini 2.0 Flash synthesises a plain-English recommendation covering:

- Which ACA plans are available and what they actually cost after your subsidy
- Whether your medications are covered, what tier, and if prior authorization is required
- Whether your doctors are in the NPPES registry and their specialty
- Whether you qualify for Medicaid, APTC subsidy, or Cost-Sharing Reduction (CSR)
- Risk flags: high OOP exposure, shortage areas, enrollment deadline

---

## System Architecture

```mermaid
graph TD
    U[User fills form\nZIP · age · income · household\nmedications · doctors] --> HiL

    HiL[Human-in-the-Loop\nConfirmation Gate] --> PA

    PA[Profile Agent\nFPL calculation & routing] --> R{Route Decision}

    R -->|FPL < 138%| MA[Medicaid Agent\nFree coverage path]
    R -->|State exchange NY/CA/MA| SE[State Exchange Redirect]
    R -->|138–400% FPL| W1
    R -->|> 400% FPL| W1

    subgraph W1[Wave 1 — asyncio.gather]
        PS[Plan Search Agent\nCMS Marketplace API]
        SA[Subsidy Agent\nAPTC · CSR · FPL %]
        DC[Doctor Check Agent\nNPPES NPI Registry]
    end

    W1 --> W2

    subgraph W2[Wave 2 — asyncio.gather]
        DK[Drug Check Agent\nRxNorm + CMS Formulary]
        RG[Risk & Gaps Agent\nHRSA · SEP · OOP flags]
        MT[Metal Tier Agent\nTrue annual cost model]
    end

    W2 --> GEM[Gemini 2.0 Flash\nVertex AI — Recommendation synthesis]
    GEM --> MEM[mem0 + ChromaDB\nStore profile for return visits]
    MEM --> OUT[Ranked plan recommendations\nwith drug · doctor · subsidy context]
```

---

## Agent Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant ORCH as ADK Orchestrator
    participant CMS as CMS Marketplace API
    participant NPPES as NPPES NPI Registry
    participant RX as RxNorm (NIH)
    participant GEM as Gemini 2.0 Flash

    U->>FE: Fill form (ZIP, age, income, drugs, doctors)
    FE->>FE: showConfirmation() — Human-in-the-loop gate
    U->>FE: Confirm profile
    FE->>API: POST /api/analyze
    API->>ORCH: analyze(profile)

    par Wave 1 — parallel
        ORCH->>CMS: search_plans(zip, age, income)
        ORCH->>CMS: get_eligibility_estimates() → APTC
        ORCH->>NPPES: lookup_npi_registry(doctor_name)
    end

    par Wave 2 — parallel
        ORCH->>RX: resolve_drug_rxcui(drug_name)
        ORCH->>CMS: check_drug_coverage(rxcui, plan_ids)
        ORCH->>CMS: check_hrsa_shortage(state, fips)
    end

    ORCH->>GEM: Synthesise recommendation with all context
    GEM-->>ORCH: Markdown recommendation
    ORCH-->>API: {plans, subsidy, drugs, doctors, risks, recommendation}
    API-->>FE: JSON response
    FE->>FE: renderResults() — tables, cards, charts
    FE-->>U: Full ranked analysis
```

---

## API Request / Response Flow

```mermaid
flowchart LR
    subgraph Frontend
        F1[Form input]
        F2[Confirmation gate]
        F3[Results render]
    end

    subgraph Backend[FastAPI Backend]
        E1[POST /api/analyze]
        E2[POST /api/chat]
        E3[POST /api/doctor-search]
        E4[POST /api/specialty-search]
        E5[POST /api/procedure-cost]
        E6[POST /api/hospital-search]
        E7[POST /api/insurance-qa]
    end

    subgraph GovAPIs[Government APIs]
        G1[CMS Marketplace\nhealthcare.gov]
        G2[NPPES NPI Registry\nnpiregistry.cms.hhs.gov]
        G3[RxNorm NIH\nrxnav.nlm.nih.gov]
        G4[openFDA\napi.fda.gov]
        G5[HRSA Shortage Areas\ndata.hrsa.gov]
    end

    F1 --> F2 --> E1
    E1 --> G1
    E1 --> G2
    E1 --> G3
    E3 --> G2
    E4 --> G2
    E6 --> G2
    E7 --> G1
    E7 --> G3
    G1 & G2 & G3 & G4 & G5 --> E1
    E1 --> F3
```

---

## Features

### Core Analysis (`POST /api/analyze`)
Runs the full multi-agent pipeline. Returns ranked plans, subsidy figures, drug coverage across plans, doctor identity verification, and risk flags — all from live government APIs.

**Input fields:**
| Field | Type | Description |
|---|---|---|
| `zip_code` | string | 5-digit ZIP |
| `age` | int | Primary applicant age |
| `income` | float | Annual household income (USD) |
| `household_size` | int | Number of people in household |
| `drugs` | string[] | Medication names (e.g. `["Ozempic", "Metformin"]`) |
| `doctors` | string[] | Doctor names to keep (e.g. `["Dr. Sarah Patel"]`) |
| `utilization` | string | `rarely` / `sometimes` / `frequently` / `chronic` |
| `is_premium` | bool | Free (3 plans, 1 drug) vs Premium (10 plans, all drugs) |

### Year-Round Chat Advisor (`POST /api/chat`)
Gemini-powered chat with full plan/drug/doctor context injected. Ask follow-up questions like "why is the Bronze plan cheaper long-term?" or "what does prior authorization mean for Ozempic?"

### Doctor Lookup (`POST /api/doctor-search`)
Standalone NPPES NPI Registry lookup. Returns NPI, specialty, city/state, phone, credential, active status, and up to 3 name-matched candidates.

```bash
curl -X POST /api/doctor-search \
  -d '{"name": "Dr. Sarah Patel", "state": "IL"}'
# → { "npi": "1487077079", "name": "SARAH PATEL",
#     "specialty": "Physician Assistant", "city": "CHICAGO", ... }
```

### Specialist Finder (`POST /api/specialty-search`)
Maps a condition (e.g. "diabetes", "back pain") to an NPPES taxonomy code and searches for local providers with MIPS quality scores.

### Procedure Cost Estimator (`POST /api/procedure-cost`)
Estimates patient out-of-pocket cost for 20 common procedures (MRI, colonoscopy, delivery, etc.) across the user's plans using deductible + coinsurance modelling.

### Hospital Network Check (`POST /api/hospital-search`)
Finds hospitals by name via NPPES and checks CMS Marketplace network status for up to 3 plans.

### Health Insurance Q&A (`POST /api/insurance-qa`)
ADK agent (Gemini 2.0 Flash decides which tools to call). Restricted to health insurance topics. Off-topic questions (flood, auto, stock) are politely declined.

**Available tools the agent can call:**
- `tool_search_health_plans` — live ACA plan lookup
- `tool_get_subsidy_info` — APTC/CSR/Medicaid eligibility
- `tool_lookup_drug_coverage` — drug tier + prior auth status
- `tool_find_local_specialists` — NPPES specialty search
- `tool_check_enrollment_period` — open enrollment / SEP status

---

## Subsidy & Routing Logic

```mermaid
flowchart TD
    I[Income + Household Size] --> FPL[Calculate FPL %\nvia HHS 2024 table]
    FPL --> R1{FPL < 138%?}
    R1 -->|Yes| M[Medicaid eligible\nFree or near-free coverage]
    R1 -->|No| R2{State exchange\nNY / CA / MA / etc?}
    R2 -->|Yes| SE[Redirect to state exchange\nhealthcare.gov won't show plans]
    R2 -->|No| R3{FPL 138–250%?}
    R3 -->|Yes| CSR[APTC + CSR eligible\nSilver plan enhanced benefits]
    R3 -->|No| R4{FPL 250–400%?}
    R4 -->|Yes| APTC[APTC eligible only\nNo CSR discount]
    R4 -->|No| FULL[No subsidy\nFull-price marketplace plans]
```

---

## Caching Strategy

All government API responses are cached with TTLs appropriate to how frequently the data changes:

```mermaid
graph LR
    subgraph Cache[TTL-Based Cache — cache_manager.py]
        C1[FPL tables\n1 year]
        C2[ZIP → FIPS crosswalk\nForever]
        C3[RxNorm drug IDs\n30 days]
        C4[CMS plan data\n24 hours]
        C5[Doctor NPI lookups\n7 days]
        C6[Drug coverage tiers\n24 hours]
        C7[HRSA shortage areas\n7 days]
    end
```

Reduces redundant API calls by ~70–80% for overlapping ZIP codes.

---

## Freemium Model

| Feature | Free | Premium |
|---|---|---|
| Plans shown | 3 cheapest | 10 plans |
| Drug checks | 1 medication | All medications |
| Doctor checks | 1 doctor | All doctors |
| Chat advisor | ✓ | ✓ |
| Specialist finder | ✓ | ✓ |
| Procedure estimator | ✓ | ✓ |
| Hospital search | ✓ | ✓ |
| Insurance Q&A | ✓ | ✓ |

---

## Data Sources

| API | Used For | Auth |
|---|---|---|
| CMS Marketplace API (`healthcare.gov`) | Live plan search, drug formulary, provider network | Free API key |
| NPPES NPI Registry | Doctor/hospital identity, specialty, NPI | None |
| RxNorm (NIH NLM) | Drug name → RxCUI resolution | None |
| openFDA | Generic drug alternatives | None |
| HHS FPL tables | Subsidy eligibility thresholds | None (static) |
| IRS applicable % table | APTC calculation | None (static) |
| HRSA Data Warehouse | Primary care shortage area scoring | None |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.11+ |
| AI / LLM | Gemini 2.0 Flash via Vertex AI (Google ADK) |
| Agent framework | Google ADK (`google-adk`) |
| Memory | mem0 + ChromaDB |
| Caching | In-process TTL dict cache |
| Frontend | Vanilla JS + HTML/CSS (no framework) |
| Deployment | Google Cloud Run |
| Protocol | REST (FastAPI) |

---

## Run Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/ritwiksharan/CoverWise
cd CoverWise
cp .env.example .env
# Fill in: GOOGLE_CLOUD_PROJECT, CMS_API_KEY

# 2. Authenticate with Google (for Vertex AI / Gemini)
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# 3. Install dependencies and run
cd backend
pip install -r requirements.txt
python3 main.py
# → http://localhost:8080
```

**Required environment variables:**

| Variable | Description |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | GCP project with Vertex AI enabled |
| `GOOGLE_CLOUD_REGION` | e.g. `us-central1` |
| `GOOGLE_GENAI_USE_VERTEXAI` | Set to `TRUE` for Vertex AI |
| `CMS_API_KEY` | Free key from `developer.cms.gov` |
| `MEM0_API_KEY` | Optional — for persistent cross-session memory |

---

## Sample Test Case

**Form input:**
```
ZIP Code:       60601  (Chicago, IL)
Age:            34
Annual Income:  $52,000
Household Size: 1
Healthcare Use: Sometimes (2–4 visits/year)
Medications:    Ozempic, Metformin
Doctors:        Dr. Sarah Patel
```

**Results:**
```
FPL: 345%  |  APTC subsidy: $58/month  |  Medicaid: No

Plans found (Bronze HMO):
  • Blue FocusCare Bronze℠ 209       $276/mo → $218/mo after subsidy  deductible $7,400
  • Aetna Bronze S (+$0 telehealth)  $317/mo → $259/mo after subsidy  deductible $7,500
  • Aetna Bronze 1 (Rx copay)        $321/mo → $263/mo after subsidy  deductible $8,995

Ozempic:    RxCUI resolved (1991311) — coverage data not provided by IL plans
Metformin:  RxCUI resolved — check plan formulary URLs for tier
Dr. Sarah Patel → SARAH PATEL, Physician Assistant, Chicago IL  NPI 1487077079  ☎ 312-340-5948

Risk flags:
  ⚠  OOP max above $8,700 on some plans
  📅 Open enrollment active — 257 days left
  ℹ  All IL plans in this area are HMO — confirm doctors are in-network
```

---

## Project Structure

```
CoverWise/
├── backend/
│   ├── main.py                    # FastAPI app + all endpoints
│   ├── agents/
│   │   ├── adk_orchestrator.py    # Main analysis pipeline
│   │   ├── insurance_qa_agent.py  # Health insurance Q&A ADK agent
│   │   ├── intake_agent.py        # Conversational intake (ADK)
│   │   └── tools.py               # Shared agent tool wrappers
│   ├── tools/
│   │   ├── gov_apis.py            # All government API calls
│   ├── cache/
│   │   └── cache_manager.py       # TTL-based caching
│   └── memory/
│       └── mem0_client.py         # Persistent user memory
├── frontend/
│   └── index.html                 # Single-page app (vanilla JS)
├── .env.example
└── README.md
```
