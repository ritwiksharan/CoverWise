"""
RAG-based drug formulary store.

Seeds from CMS machine-readable formulary (MRF) JSON files published by insurers.
Two-layer lookup:
  1. SQLite  — exact (rxnorm_id, plan_id) coverage data
  2. ChromaDB — semantic drug-name search for brand/generic/strength variants

Fallback called by check_drug_coverage() when CMS API returns DataNotProvided.
"""

import os
import json
import sqlite3
import threading
import requests
import chromadb
from typing import Optional

_BASE_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_BASE_DIR, "..", "data")
DB_PATH = os.path.join(_DATA_DIR, "formulary.db")
CHROMA_PATH = os.path.join(_DATA_DIR, "formulary_chroma")

# CMS Machine Readable File index URLs keyed by 5-digit issuer ID.
# Sourced from Machine_Readable_PUF.xlsx (CMS 2024).
ISSUER_MRF_URLS: dict[str, str] = {
    "36096": "https://www.bcbsil.com/aca-json/il/index_il.json",          # BCBS IL
    "38344": "https://fm.formularynavigator.com/jsonFiles/publish/11/47/cms-data-index.json",
    "73836": "https://www.modahealth.com/cms-data-index.json",
    "99129": "https://www.aetna.com/json/mrf/index.json",                  # Aetna
    "14165": "https://www.uhc.com/content/dam/uhcdotcom/en/HealthReform/PDF/machine-readable-files/index.json",
    "62308": "https://www.cigna.com/static/www-cigna-com/docs/json/formulary/index.json",
    "77422": "https://www.humana.com/content/dam/humana/individual-family/mrf/index.json",
    "29530": "https://www.oscar.com/machine-readable/index.json",
    "84418": "https://www.ambetterhealth.com/content/dam/centene/machine-readable/index.json",
    "60052": "https://www.molina.com/content/dam/molinahealth/machine-readable/index.json",
    "10693": "https://www.kaiserpermanente.org/content/dam/public/kporg/en/machine-readable-files/formulary/index.json",
}

_seed_lock = threading.Lock()
_seeded_issuers: set[str] = set()
_seed_in_progress: set[str] = set()


# ── SQLite helpers ────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    os.makedirs(_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS drug_coverage (
                issuer_id         TEXT NOT NULL,
                rxnorm_id         TEXT NOT NULL,
                drug_name         TEXT NOT NULL,
                drug_name_lower   TEXT NOT NULL,
                plan_id           TEXT NOT NULL,
                drug_tier         TEXT,
                prior_authorization INTEGER DEFAULT 0,
                step_therapy      INTEGER DEFAULT 0,
                quantity_limit    INTEGER DEFAULT 0,
                PRIMARY KEY (issuer_id, rxnorm_id, plan_id)
            );
            CREATE INDEX IF NOT EXISTS idx_plan_rxnorm
                ON drug_coverage (plan_id, rxnorm_id);
            CREATE INDEX IF NOT EXISTS idx_drug_name
                ON drug_coverage (drug_name_lower);
            CREATE INDEX IF NOT EXISTS idx_issuer
                ON drug_coverage (issuer_id);

            CREATE TABLE IF NOT EXISTS seeded_issuers (
                issuer_id TEXT PRIMARY KEY,
                drug_count INTEGER,
                seeded_at TEXT
            );
        """)


_init_db()


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_lock = threading.Lock()


def _get_collection() -> chromadb.Collection:
    global _chroma_client
    with _chroma_lock:
        if _chroma_client is None:
            os.makedirs(CHROMA_PATH, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        return _chroma_client.get_or_create_collection(
            name="formulary_drugs",
            metadata={"hnsw:space": "cosine"},
        )


# ── Seeding ───────────────────────────────────────────────────────────────────

def _already_seeded(issuer_id: str) -> bool:
    if issuer_id in _seeded_issuers:
        return True
    with _get_db() as conn:
        row = conn.execute(
            "SELECT issuer_id FROM seeded_issuers WHERE issuer_id = ?", (issuer_id,)
        ).fetchone()
        if row:
            _seeded_issuers.add(issuer_id)
            return True
    return False


def _fetch_formulary_urls(mrf_index_url: str) -> list[str]:
    try:
        r = requests.get(mrf_index_url, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("formulary_urls", [])
    except Exception as e:
        print(f"[formulary] index fetch error {mrf_index_url}: {e}")
        return []


def _seed_formulary_url(issuer_id: str, formulary_url: str) -> int:
    """Download one formulary JSON and insert all drugs into SQLite + ChromaDB."""
    try:
        r = requests.get(formulary_url, timeout=60)
        if r.status_code != 200:
            print(f"[formulary] {formulary_url} → {r.status_code}")
            return 0
        drugs: list = r.json()
        if isinstance(drugs, dict):
            drugs = drugs.get("formulary", drugs.get("drugs", []))
        if not isinstance(drugs, list):
            return 0
    except Exception as e:
        print(f"[formulary] download error {formulary_url}: {e}")
        return 0

    rows = []
    chroma_docs, chroma_ids, chroma_metas = [], [], []
    seen_chroma = set()

    for drug in drugs:
        rxnorm_id = str(drug.get("rxnorm_id", "") or "")
        drug_name = (drug.get("drug_name") or "").strip()
        if not rxnorm_id or not drug_name:
            continue

        chroma_key = f"{issuer_id}::{rxnorm_id}"
        if chroma_key not in seen_chroma:
            seen_chroma.add(chroma_key)
            chroma_docs.append(drug_name)
            chroma_ids.append(chroma_key)
            chroma_metas.append({"rxnorm_id": rxnorm_id, "issuer_id": issuer_id,
                                  "drug_name_lower": drug_name.lower()})

        for plan in drug.get("plans", []):
            plan_id = plan.get("plan_id", "")
            if not plan_id:
                continue
            rows.append((
                issuer_id, rxnorm_id, drug_name, drug_name.lower(),
                plan_id,
                plan.get("drug_tier") or "",
                int(bool(plan.get("prior_authorization", False))),
                int(bool(plan.get("step_therapy", False))),
                int(bool(plan.get("quantity_limit", False))),
            ))

    if not rows:
        return 0

    with _get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO drug_coverage
               (issuer_id, rxnorm_id, drug_name, drug_name_lower, plan_id,
                drug_tier, prior_authorization, step_therapy, quantity_limit)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    if chroma_docs:
        col = _get_collection()
        col.upsert(documents=chroma_docs, ids=chroma_ids, metadatas=chroma_metas)

    return len(set(r[1] for r in rows))  # unique drug count


def seed_issuer(issuer_id: str, blocking: bool = True) -> int:
    """
    Download and index formulary data for a given 5-digit issuer ID.
    Returns the number of unique drugs indexed (0 if issuer unknown or already seeded).
    """
    if _already_seeded(issuer_id):
        return 0
    if issuer_id not in ISSUER_MRF_URLS:
        return 0

    with _seed_lock:
        if issuer_id in _seed_in_progress:
            return 0
        _seed_in_progress.add(issuer_id)

    def _run():
        try:
            mrf_index_url = ISSUER_MRF_URLS[issuer_id]
            formulary_urls = _fetch_formulary_urls(mrf_index_url)
            if not formulary_urls:
                print(f"[formulary] no formulary URLs found for issuer {issuer_id}")
                return 0

            total_drugs = 0
            for url in formulary_urls:
                total_drugs += _seed_formulary_url(issuer_id, url)

            from datetime import datetime, timezone
            with _get_db() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO seeded_issuers VALUES (?,?,?)",
                    (issuer_id, total_drugs, datetime.now(timezone.utc).isoformat()),
                )
            _seeded_issuers.add(issuer_id)
            print(f"[formulary] seeded issuer {issuer_id}: {total_drugs} drugs")
            return total_drugs
        except Exception as e:
            print(f"[formulary] seed error for {issuer_id}: {e}")
            return 0
        finally:
            with _seed_lock:
                _seed_in_progress.discard(issuer_id)

    if blocking:
        return _run()
    else:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return -1  # seeding in background


def seed_issuers_from_plan_ids(plan_ids: list[str], blocking: bool = False):
    """Trigger background seeding for all issuers referenced by a list of plan_ids."""
    for pid in plan_ids:
        issuer_id = pid[:5]
        if issuer_id in ISSUER_MRF_URLS and not _already_seeded(issuer_id):
            seed_issuer(issuer_id, blocking=blocking)


# ── Lookup ────────────────────────────────────────────────────────────────────

def _sqlite_lookup(rxnorm_id: str, plan_ids: list[str]) -> list[dict]:
    if not rxnorm_id or not plan_ids:
        return []
    placeholders = ",".join("?" * len(plan_ids))
    with _get_db() as conn:
        rows = conn.execute(
            f"""SELECT plan_id, drug_name, rxnorm_id, drug_tier, prior_authorization,
                       step_therapy, quantity_limit
                FROM drug_coverage
                WHERE rxnorm_id = ? AND plan_id IN ({placeholders})""",
            (rxnorm_id, *plan_ids),
        ).fetchall()
    return [dict(r) for r in rows]


def _sqlite_issuer_fallback(rxnorm_id: str, issuer_ids: list[str],
                             requested_plan_ids: list[str]) -> list[dict]:
    """
    When plan IDs change year-to-year, fall back to the first available plan
    from the same issuer. Marks results with proxy_plan=True.
    """
    if not rxnorm_id or not issuer_ids:
        return []
    placeholders = ",".join("?" * len(issuer_ids))
    with _get_db() as conn:
        rows = conn.execute(
            f"""SELECT plan_id, drug_name, rxnorm_id, drug_tier, prior_authorization,
                       step_therapy, quantity_limit, issuer_id
                FROM drug_coverage
                WHERE rxnorm_id = ? AND issuer_id IN ({placeholders})
                LIMIT 1""",
            (rxnorm_id, *issuer_ids),
        ).fetchall()
    if not rows:
        return []
    result = dict(rows[0])
    # Tag as proxy so callers can add a note
    result["proxy_plan"] = True
    result["original_plan_ids"] = requested_plan_ids
    return [result]


def _sqlite_name_lookup(drug_name: str, plan_ids: list[str]) -> list[dict]:
    """Exact or prefix substring match on drug_name_lower."""
    if not drug_name or not plan_ids:
        return []
    name_lower = drug_name.lower().strip()
    # Try exact then prefix
    for pattern in (name_lower, f"{name_lower}%"):
        placeholders = ",".join("?" * len(plan_ids))
        with _get_db() as conn:
            rows = conn.execute(
                f"""SELECT plan_id, drug_name, rxnorm_id, drug_tier,
                           prior_authorization, step_therapy, quantity_limit
                    FROM drug_coverage
                    WHERE drug_name_lower LIKE ? AND plan_id IN ({placeholders})""",
                (pattern, *plan_ids),
            ).fetchall()
        if rows:
            return [dict(r) for r in rows]
    return []


def _chroma_lookup(drug_name: str, issuer_ids: list[str]) -> list[str]:
    """Semantic search: drug_name → list of rxnorm_ids from ChromaDB."""
    try:
        col = _get_collection()
        if col.count() == 0:
            return []
        where = {"issuer_id": {"$in": issuer_ids}} if issuer_ids else None
        kw = {"query_texts": [drug_name], "n_results": 5}
        if where:
            kw["where"] = where
        results = col.query(**kw)
        rxnorm_ids = []
        if results and results["metadatas"]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                if dist < 0.4:
                    rxnorm_ids.append(meta["rxnorm_id"])
        return rxnorm_ids
    except Exception as e:
        print(f"[formulary] chroma lookup error: {e}")
        return []


def lookup_drug_coverage(drug_name: str, rxcui: str, plan_ids: list[str]) -> list[dict]:
    """
    Look up drug coverage from the formulary RAG store.

    Returns a list of dicts with keys:
      rxcui, plan_id, coverage, drug_tier, prior_authorization,
      step_therapy, quantity_limit, source
    Returns [] if nothing found.
    """
    if not plan_ids:
        return []

    issuer_ids = list({pid[:5] for pid in plan_ids})

    # Resolve best rxnorm_ids for this drug (try direct then semantic)
    candidate_rxnorms: list[str] = []
    if rxcui:
        candidate_rxnorms.append(rxcui)
    candidate_rxnorms.extend(_chroma_lookup(drug_name, issuer_ids))
    # Also try SQLite name-prefix for additional rxnorm_ids
    name_rows = _sqlite_name_lookup(drug_name, plan_ids)
    for nr in name_rows:
        rxn = nr.get("rxnorm_id", "")
        if rxn and rxn not in candidate_rxnorms:
            candidate_rxnorms.append(rxn)

    # Look up each plan_id individually so the issuer fallback runs per-plan
    rows: list[dict] = []
    satisfied_plans: set[str] = set()

    for rxn in candidate_rxnorms:
        remaining = [p for p in plan_ids if p not in satisfied_plans]
        if not remaining:
            break
        found = _sqlite_lookup(rxn, remaining)
        for r in found:
            satisfied_plans.add(r["plan_id"])
        rows.extend(found)

    # Per-plan issuer fallback for plans still not found (year-to-year ID change)
    for pid in plan_ids:
        if pid not in satisfied_plans:
            for rxn in candidate_rxnorms:
                fallback = _sqlite_issuer_fallback(rxn, [pid[:5]], [pid])
                if fallback:
                    rows.extend(fallback)
                    satisfied_plans.add(pid)
                    break

    if not rows:
        return []

    return [
        {
            "rxcui": row.get("rxnorm_id", rxcui),
            "plan_id": row["plan_id"],
            "coverage": "Covered" if row.get("drug_tier") else "DataNotProvided",
            "drug_tier": row.get("drug_tier"),
            "prior_authorization": bool(row.get("prior_authorization", 0)),
            "step_therapy": bool(row.get("step_therapy", 0)),
            "quantity_limit": bool(row.get("quantity_limit", 0)),
            "source": "rag_formulary",
            "drug_name_matched": row.get("drug_name"),
            # Proxy: formulary plan ID differs from requested plan ID (year change)
            **({"note": "Coverage from issuer formulary (plan year may differ)",
                "original_plan_ids": row.get("original_plan_ids", [])}
               if row.get("proxy_plan") else {}),
        }
        for row in rows
    ]


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    with _get_db() as conn:
        total_drugs = conn.execute("SELECT COUNT(DISTINCT rxnorm_id) FROM drug_coverage").fetchone()[0]
        total_plans = conn.execute("SELECT COUNT(DISTINCT plan_id) FROM drug_coverage").fetchone()[0]
        total_rows = conn.execute("SELECT COUNT(*) FROM drug_coverage").fetchone()[0]
        issuers = conn.execute(
            "SELECT issuer_id, drug_count, seeded_at FROM seeded_issuers"
        ).fetchall()
    try:
        chroma_count = _get_collection().count()
    except Exception:
        chroma_count = 0
    return {
        "unique_drugs": total_drugs,
        "unique_plans": total_plans,
        "total_coverage_rows": total_rows,
        "chroma_documents": chroma_count,
        "seeded_issuers": [dict(r) for r in issuers],
    }
