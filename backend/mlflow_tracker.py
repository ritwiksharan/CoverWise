"""
MLflow experiment tracking for the CoverWise 3-phase analysis pipeline.

Uses MlflowClient directly (not the fluent API) so that concurrent async
requests each get their own isolated run — no shared thread-local state.
All calls are best-effort; exceptions are swallowed so tracking failures
never interrupt the recommendation flow.
"""

import json
import os

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT_NAME", "coverwise-recommendations")

_client: "MlflowClient | None" = None
_experiment_id: "str | None"   = None


def _get_client() -> "MlflowClient | None":
    global _client, _experiment_id
    if not _MLFLOW:
        return None
    try:
        if _client is None:
            _client = MlflowClient(tracking_uri=TRACKING_URI)
        if _experiment_id is None:
            exp = _client.get_experiment_by_name(EXPERIMENT)
            if exp is None:
                _experiment_id = _client.create_experiment(EXPERIMENT)
            else:
                _experiment_id = exp.experiment_id
        return _client
    except Exception:
        return None


def _safe(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

class analysis_run:
    """
    Context manager that creates one MLflow run per request using the
    client API with an explicit run_id — safe for concurrent async use.
    """

    def __init__(self, profile: dict):
        self._profile = profile
        self._run_id: "str | None" = None

    def __enter__(self):
        c = _get_client()
        if c is None:
            return self
        try:
            run = c.create_run(experiment_id=_experiment_id)
            self._run_id = run.info.run_id
            p = self._profile
            params = {
                "zip_code":       str(p.get("zip_code", "")),
                "age":            str(p.get("age", "")),
                "income":         str(p.get("income", "")),
                "household_size": str(p.get("household_size", "")),
                "utilization":    str(p.get("utilization", "sometimes")),
                "tobacco_use":    str(p.get("tobacco_use", False)),
                "is_premium":     str(p.get("is_premium", False)),
                "num_drugs":      str(len(p.get("drugs", []))),
                "num_doctors":    str(len(p.get("doctors", []))),
            }
            for k, v in params.items():
                _safe(c.log_param, self._run_id, k, v)
        except Exception:
            self._run_id = None
        return self

    def __exit__(self, exc_type, *_):
        c = _get_client()
        if c is None or self._run_id is None:
            return
        status = "FAILED" if exc_type else "FINISHED"
        _safe(c.set_terminated, self._run_id, status)

    @property
    def run_id(self) -> "str | None":
        return self._run_id


def log_phase1(run: analysis_run, data: dict, latency_s: float):
    """Phase 1 — data collection: subsidy, plans, risk flags."""
    c = _get_client()
    if c is None or run.run_id is None:
        return
    rid = run.run_id
    subsidy = data.get("subsidy", {})
    plans   = data.get("plans",   [])
    loc     = data.get("location", {})

    metrics = {
        "phase1_latency_s": round(latency_s, 2),
        "fpl_percentage":   round(subsidy.get("fpl_percentage", 0), 1),
        "monthly_aptc":     round(subsidy.get("monthly_aptc", 0), 2),
        "annual_aptc":      round(subsidy.get("monthly_aptc", 0) * 12, 2),
        "num_plans":        float(len(plans)),
        "num_hsa_plans":    float(sum(1 for p in plans if p.get("hsa_eligible"))),
        "num_risk_flags":   float(len(data.get("risk_flags", []))),
        "csr_eligible":     1.0 if subsidy.get("csr_variant") else 0.0,
    }
    for k, v in metrics.items():
        _safe(c.log_metric, rid, k, v)

    _safe(c.set_tag, rid, "state",        loc.get("state", ""))
    _safe(c.set_tag, rid, "csr_variant",  subsidy.get("csr_variant") or "none")
    _safe(c.set_tag, rid, "is_medicaid",  str(subsidy.get("is_medicaid_eligible", False)))

    if plans:
        summary = [
            {
                "rank":                  i + 1,
                "name":                  p.get("name"),
                "metal_level":           p.get("metal_level"),
                "type":                  p.get("type"),
                "premium_after_subsidy": p.get("premium_w_credit"),
                "deductible":            p.get("deductible"),
                "oop_max":               p.get("oop_max"),
                "true_annual_cost":      p.get("true_annual_cost"),
                "scenario_healthy":      p.get("scenario_healthy"),
                "scenario_clinical":     p.get("scenario_clinical"),
                "scenario_worst":        p.get("scenario_worst"),
                "hsa_eligible":          p.get("hsa_eligible"),
            }
            for i, p in enumerate(plans)
        ]
        _safe(c.log_text, rid, json.dumps(summary, indent=2), "plans_ranked.json")


def log_phase15(run: analysis_run, ranking: dict, latency_s: float):
    """Phase 1.5 — LLM ranking: EV scores, top pick, red flags."""
    c = _get_client()
    if c is None or run.run_id is None:
        return
    rid = run.run_id

    _safe(c.log_metric, rid, "phase15_latency_s", round(latency_s, 2))

    if not ranking:
        _safe(c.set_tag, rid, "phase15_status", "skipped")
        return

    ev_list = ranking.get("expected_value_ranking", [])
    top_rec = ranking.get("top_recommendation", {})

    _safe(c.log_metric, rid, "phase15_plans_ranked", float(len(ev_list)))
    _safe(c.log_metric, rid, "top_plan_ev_score",    float(ev_list[0].get("ev_score", 0)) if ev_list else 0.0)
    _safe(c.log_metric, rid, "ranking_red_flags",    float(len(ranking.get("red_flags", []))))

    _safe(c.set_tag, rid, "top_plan_name",  top_rec.get("plan_name", ""))
    _safe(c.set_tag, rid, "csr_override",   ranking.get("csr_override") or "none")
    _safe(c.set_tag, rid, "phase15_status", "ok")

    _safe(c.log_text, rid, json.dumps(ranking, indent=2), "llm_ranking.json")


def log_phase2(run: analysis_run, recommendation: str, synthesis_prompt: str, latency_s: float):
    """Phase 2 — Gemini synthesis: latency, char counts, full artifacts."""
    c = _get_client()
    if c is None or run.run_id is None:
        return
    rid = run.run_id

    _safe(c.log_metric, rid, "phase2_latency_s",       round(latency_s, 2))
    _safe(c.log_metric, rid, "recommendation_chars",   float(len(recommendation)))
    _safe(c.log_metric, rid, "synthesis_prompt_chars", float(len(synthesis_prompt)))

    _safe(c.log_text, rid, recommendation,   "recommendation.md")
    _safe(c.log_text, rid, synthesis_prompt, "synthesis_prompt.txt")


def log_cache(run: analysis_run, cache_stats: dict):
    """Log cache hit/miss counters."""
    c = _get_client()
    if c is None or run.run_id is None or not cache_stats:
        return
    rid = run.run_id
    for k, v in cache_stats.items():
        if isinstance(v, (int, float)):
            _safe(c.log_metric, rid, f"cache_{k}", float(v))
