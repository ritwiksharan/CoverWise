# MLflow Integration — CoverWise

## Why MLflow

CoverWise runs a three-phase LLM pipeline for every user analysis:

- **Phase 1** — parallel CMS government API calls (location, subsidy, plans, drugs, doctors)
- **Phase 1.5** — Gemini in JSON mode ranks plans by expected value across three healthcare scenarios
- **Phase 2** — Gemini synthesizes a full four-pillar recommendation from structured data

Without observability, this pipeline is a black box. When a recommendation looks wrong, there is no way to answer:

- What inputs did that user provide?
- What exact prompt did Gemini receive?
- Which phase was slow — the CMS API calls or Gemini?
- Did changing the system prompt actually improve recommendations?

MLflow gives every analysis run a permanent, queryable record so these questions can be answered without re-running anything.

---

## What is tracked

Every call to `/api/analyze` is logged as one MLflow **run** with three phases.

### Input params (logged once per run)
Captured from the user profile before the pipeline starts:

| Param | Description |
|---|---|
| `zip_code` | User's ZIP code |
| `age` | User's age |
| `income` | Annual household income |
| `household_size` | Number of people in the household |
| `utilization` | Healthcare usage level (rarely / sometimes / frequently / chronic) |
| `tobacco_use` | Tobacco use flag |
| `is_premium` | Whether the user is on the premium tier |
| `num_drugs` | Number of medications entered |
| `num_doctors` | Number of doctors entered |

### Phase 1 metrics
Logged after the parallel CMS API wave completes:

| Metric | Description |
|---|---|
| `phase1_latency_s` | Time taken for all CMS API calls (seconds) |
| `fpl_percentage` | Federal poverty level as a percentage of the user's income |
| `monthly_aptc` | Monthly Advance Premium Tax Credit from CMS |
| `annual_aptc` | Annual APTC (monthly x 12) |
| `num_plans` | Number of plans returned by CMS Marketplace API |
| `num_hsa_plans` | Number of HSA-eligible plans in the results |
| `num_risk_flags` | Number of risk flags raised (subsidy cliff, OOP exposure, etc.) |
| `csr_eligible` | 1 if the user qualifies for Cost Sharing Reduction, 0 otherwise |

**Tags:** `state`, `csr_variant` (94 / 87 / 73 / none), `is_medicaid`

**Artifact:** `plans_ranked.json` — full ranked plan list with all three scenario costs (healthy year, clinical year, worst case).

### Phase 1.5 metrics
Logged after the LLM ranking agent completes:

| Metric | Description |
|---|---|
| `phase15_latency_s` | Time taken for Gemini JSON-mode ranking call (seconds) |
| `phase15_plans_ranked` | Number of plans ranked by the LLM |
| `top_plan_ev_score` | Expected value score of the winning plan (lower is better) |
| `ranking_red_flags` | Number of red flags raised by the ranking agent |

**Tags:** `top_plan_name`, `csr_override`, `phase15_status`

**Artifact:** `llm_ranking.json` — full ranking output including EV scores, scenario breakdowns, CSR analysis, and red flags.

### Phase 2 metrics
Logged after the Gemini synthesis call completes:

| Metric | Description |
|---|---|
| `phase2_latency_s` | Time taken for the final Gemini synthesis call (seconds) |
| `recommendation_chars` | Character count of the generated recommendation |
| `synthesis_prompt_chars` | Character count of the prompt sent to Gemini |

**Artifacts:** `recommendation.md` (full recommendation text), `synthesis_prompt.txt` (exact prompt sent to Gemini).

### Cache metrics
`cache_hits`, `cache_misses`, and any other values returned by `cache_manager.get_cache_stats()` are logged as `cache_*` metrics.

---

## Key concepts tracked

### APTC vs CSR

These are two separate federal benefits that often appear together in runs:

**APTC (Advance Premium Tax Credit)** reduces the monthly premium. It is paid directly to the insurer each month before the user uses any healthcare. The amount depends on income relative to the benchmark Silver plan cost in the user's area.

**CSR (Cost Sharing Reduction)** reduces out-of-pocket costs — deductibles, copays, and the out-of-pocket maximum. It only applies to Silver plans and only for users between 100–250% FPL:

| CSR Variant | FPL Range | Effective Deductible |
|---|---|---|
| CSR-94 | 100–150% FPL | ~$0–500 (Platinum-level) |
| CSR-87 | 150–200% FPL | ~$500–1,500 (Gold-level) |
| CSR-73 | 200–250% FPL | ~$1,500–3,000 (enhanced Silver) |

Users with both high APTC and CSR eligibility are getting the maximum benefit from the ACA subsidy system — two simultaneous reductions that Bronze or Gold plans cannot provide.

---

## Architecture

### `backend/mlflow_tracker.py`

The tracking module. Uses `MlflowClient` directly rather than MLflow's fluent API (`mlflow.log_metric`, `mlflow.start_run`). This is intentional: the fluent API stores the active run in thread-local state, which is shared across all concurrent async requests in FastAPI. Direct client calls with explicit run IDs keep each request's tracking fully isolated.

```python
# Each request creates its own run with an explicit run_id
run = client.create_run(experiment_id=_experiment_id)
client.log_metric(run.info.run_id, "fpl_percentage", 278.9)
client.set_terminated(run.info.run_id, "FINISHED")
```

### `backend/agents/adk_orchestrator.py`

The three-phase pipeline is wrapped in a `mlflow_tracker.analysis_run` context manager inside `ADKOrchestrator.analyze()`. Each phase is timed and logged immediately after it completes.

### `backend/mlflow_dashboard.html`

A standalone monitoring dashboard served at `/mlflow-dashboard`. Separate from the main frontend (`/`). Reads data from the FastAPI proxy endpoints rather than connecting to MLflow directly, so no separate MLflow UI server is required.

### `backend/main.py`

Three endpoints added:

| Endpoint | Description |
|---|---|
| `GET /mlflow-dashboard` | Serves the standalone dashboard HTML |
| `GET /api/mlflow/runs` | Returns all runs with params, metrics, and tags |
| `GET /api/mlflow/runs/{run_id}` | Returns full detail for one run including artifact list |
| `GET /api/mlflow/runs/{run_id}/artifacts/{filename}` | Returns artifact file content |

---

## Setup

### Install

```bash
pip install "mlflow>=2.14.0"
```

Or install all project dependencies:

```bash
pip install -e ".[dev]"
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Where MLflow stores run data |
| `MLFLOW_EXPERIMENT_NAME` | `coverwise-recommendations` | Experiment name in MLflow |

Add these to your `.env` file if you want to override the defaults.

### Running

Start the backend as normal. MLflow tracking is automatic — every `/api/analyze` request is logged.

```bash
cd backend
uvicorn main:app --reload --port 8080
```

Then open the dashboard:

```
http://localhost:8080/mlflow-dashboard
```

No separate MLflow server needed. The dashboard reads directly from `mlflow.db` via the FastAPI backend.

### Data location

```
backend/
├── mlflow.db                      # SQLite database — params, metrics, tags, run metadata
└── mlruns/
    └── 1/
        └── <run_id>/
            └── artifacts/
                ├── recommendation.md       # Full Gemini recommendation
                ├── synthesis_prompt.txt    # Exact prompt sent to Gemini
                ├── plans_ranked.json       # All plans with scenario costs
                └── llm_ranking.json        # EV scores and red flags
```

---

## Dashboard

The dashboard at `/mlflow-dashboard` has three views:

**Overview** — summary stat cards (total runs, avg APTC, avg FPL%, avg total latency, avg plans found) and four charts (stacked latency per run, APTC trend, FPL% distribution, plans found).

**All Runs** — full table with every logged value: status, run ID, time, ZIP, age, income, utilization, FPL%, APTC, plans found, HSA plans, CSR eligibility, top plan name, EV score, per-phase latency breakdown, prompt size, recommendation size.

**Charts** — stacked latency breakdown, income vs APTC scatter, CSR eligibility donut, utilization mix donut.

**Run detail** — click any run to see full params, all metrics with color coding, all tags, a horizontal latency bar chart per phase, and an artifact viewer for all four artifact files.

---

## What you can do with this data

| Question | How to answer it |
|---|---|
| Why did a user get a bad recommendation? | Open the run, read `synthesis_prompt.txt` to see exactly what Gemini received |
| Which ZIP codes have the slowest CMS API response? | Filter All Runs by `state` tag, sort by `phase1_latency_s` |
| Did editing the system prompt improve recommendations? | Compare `recommendation.md` artifacts across runs before and after the change |
| What is the average subsidy for users we serve? | Aggregate `monthly_aptc` across all runs in the Overview chart |
| Where is pipeline latency coming from? | The latency bar in All Runs shows the Phase 1 / Ranking / Gemini split per run |
| Are CSR-eligible users getting Silver recommendations? | Filter by `csr_eligible=1`, check `top_plan_name` tag for Silver plans |
