# MLflow Tracking — CoverWise

## What this branch adds

MLflow experiment tracking across the three-phase analysis pipeline in `ADKOrchestrator.analyze()`. Every call to `/api/analyze` is logged as one run. A standalone dashboard is served at `/mlflow-dashboard` with no separate MLflow server required.

---

## Why MLflow specifically

CoverWise already uses Google ADK, which has built-in cloud tracing via Cloud Trace and Cloud Logging. MLflow was added on top of that for three reasons ADK does not cover:

1. **Works locally without GCP.** Cloud Trace requires a live GCP project. MLflow writes to a local SQLite file (`mlflow.db`) — useful during development and for anyone running the project without cloud credentials.

2. **Cross-run comparison.** ADK tracing tells you what happened inside a single agent turn. MLflow lets you compare `phase1_latency_s`, `monthly_aptc`, and `top_plan_ev_score` across dozens of runs to spot regressions after a prompt change or a CMS API change.

3. **Artifact storage per run.** The exact prompt sent to Gemini (`synthesis_prompt.txt`) and the full recommendation text (`recommendation.md`) are saved as files attached to each run. If a recommendation looks wrong, you can open that specific run and read exactly what Gemini received — without re-running the request.

---

## What is logged

Every `/api/analyze` request creates one MLflow run. The run captures inputs, outputs, and timing across all three phases.

### Params — user profile inputs

Logged once at the start of the run, before any API calls:

```
zip_code, age, income, household_size, utilization,
tobacco_use, is_premium, num_drugs, num_doctors
```

These are the raw inputs the user submitted. They let you filter runs by profile type (e.g. all users with chronic utilization and 3+ drugs).

### Phase 1 — CMS API data collection

`_collect_analysis_data()` runs three waves of parallel CMS API calls: location lookup, subsidy estimate + plan search, then drug coverage + doctor verification + risk flags. After all three waves finish:

| Metric | What it tells you |
|---|---|
| `phase1_latency_s` | How long the CMS government APIs took. Spikes here mean CMS is slow, not Gemini. |
| `fpl_percentage` | The user's income as a % of federal poverty level. Drives subsidy eligibility. |
| `monthly_aptc` | The monthly premium subsidy CMS calculated for this user. |
| `annual_aptc` | `monthly_aptc * 12`. |
| `num_plans` | How many marketplace plans CMS returned for this ZIP/age/income. |
| `num_hsa_plans` | Plans that are HSA-eligible — relevant to the tax savings flag in risk analysis. |
| `num_risk_flags` | Flags raised by `risk_gaps_agent`: subsidy cliff, OOP exposure, HSA opportunity, etc. |
| `csr_eligible` | 1 if FPL is between 100–250%, else 0. Determines whether Silver plan CSR applies. |

Tags logged: `state`, `csr_variant` (94 / 87 / 73 / none), `is_medicaid`.

Artifact saved: `plans_ranked.json` — every plan returned by CMS with its three scenario costs (healthy year = premiums only, clinical year = premiums + drug costs, worst case = premiums + full OOP max).

### Phase 1.5 — LLM ranking agent

`_rank_plans_with_llm()` calls Gemini in JSON mode with utilization-adjusted EV weights to rank all plans. After it returns:

| Metric | What it tells you |
|---|---|
| `phase15_latency_s` | How long Gemini took to rank plans in JSON mode. |
| `phase15_plans_ranked` | How many plans the LLM ranked. Should match `num_plans`. |
| `top_plan_ev_score` | The expected value score of the winning plan in dollars (lower = cheaper). |
| `ranking_red_flags` | Specific warnings raised by the ranking agent (e.g. prior auth required on top plan). |

Tags logged: `top_plan_name`, `csr_override` (Silver plan ID if CSR makes it clearly superior), `phase15_status`.

Artifact saved: `llm_ranking.json` — full JSON output from the ranking agent including EV scores for every plan across all three scenarios, CSR analysis, and red flags with dollar amounts.

### Phase 2 — Gemini synthesis

`_synthesize_with_gemini()` calls Gemini with the full structured data document and `ORCHESTRATOR_INSTRUCTION` as the system prompt. After it returns:

| Metric | What it tells you |
|---|---|
| `phase2_latency_s` | How long the final Gemini synthesis call took. |
| `synthesis_prompt_chars` | Size of the prompt sent to Gemini. Grows with plan count and drug/doctor detail. |
| `recommendation_chars` | Size of the generated recommendation. |

Artifacts saved: `recommendation.md` (the full text shown to the user), `synthesis_prompt.txt` (the exact prompt Gemini received, including all CMS data tables).

### Cache metrics

Whatever `get_cache_stats()` returns is logged as `cache_*` metrics (e.g. `cache_hits`, `cache_misses`). Lets you track CMS API cache effectiveness over time.

---

## Implementation detail — why MlflowClient and not the fluent API

The standard MLflow fluent API (`mlflow.start_run()`, `mlflow.log_metric()`) stores the active run in thread-local state. In a FastAPI async server, all concurrent requests run in the same thread, so concurrent `/api/analyze` calls would share the same "active run" and overwrite each other's metrics.

`mlflow_tracker.py` uses `MlflowClient` directly with explicit run IDs instead:

```python
# In analysis_run.__enter__():
run = client.create_run(experiment_id=_experiment_id)
self._run_id = run.info.run_id  # stored on the instance, not globally

# In log_phase1():
client.log_metric(run.run_id, "phase1_latency_s", 5.2)
```

Each request's `analysis_run` instance holds its own `run_id`. No shared state. Safe for concurrent async use.

---

## Files changed

| File | Change |
|---|---|
| `backend/mlflow_tracker.py` | New — tracking module with `analysis_run` context manager and per-phase log functions |
| `backend/mlflow_dashboard.html` | New — standalone dashboard with overview stats, charts, runs table, and artifact viewer |
| `backend/agents/adk_orchestrator.py` | Modified — `ADKOrchestrator.analyze()` wrapped with `mlflow_tracker.analysis_run` |
| `backend/main.py` | Modified — added `/mlflow-dashboard`, `/api/mlflow/runs`, `/api/mlflow/runs/{id}`, `/api/mlflow/runs/{id}/artifacts/{file}` |
| `pyproject.toml` | Modified — added `mlflow>=2.14.0` dependency |

---

## Setup

```bash
pip install "mlflow>=2.14.0"
```

Optional `.env` overrides:

```
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=coverwise-recommendations
```

Run the backend normally. Tracking is automatic on every `/api/analyze` call:

```bash
cd backend
uvicorn main:app --reload --port 8080
```

Dashboard: `http://localhost:8080/mlflow-dashboard`

Data is stored in `backend/mlflow.db` (metrics/params/tags) and `backend/mlruns/` (artifact files).

---

## Using the data

| If you want to... | Look at... |
|---|---|
| Debug a bad recommendation | Open the run, read `synthesis_prompt.txt` — that is exactly what Gemini saw |
| Find which ZIP codes hit slow CMS APIs | Sort All Runs by `phase1_latency_s` descending |
| Verify a prompt change improved output quality | Compare `recommendation.md` artifacts across runs before and after the edit |
| See where total latency is coming from | The latency bar in All Runs splits Phase 1 / Ranking / Gemini per run |
| Check whether CSR-eligible users got Silver plans | Filter `csr_eligible=1`, read `top_plan_name` tag |
| Track average subsidy across all users | `monthly_aptc` in the Overview chart |
