"""
Cache Manager - TTL-based caching for government API responses
Reduces redundant CMS/RxNorm/NPPES API calls significantly
"""

import time
import json
import hashlib
from typing import Any, Optional
from datetime import datetime

# In-memory cache store: {key: {data, expires_at, hits}}
_cache: dict = {}

# TTL settings per data type (seconds)
TTL_CONFIG = {
    "fpl_table": 365 * 24 * 3600,       # Federal Poverty Level — changes once a year
    "zip_fips": 365 * 24 * 3600,         # ZIP→FIPS crosswalk — never changes
    "rxnorm_drug": 30 * 24 * 3600,       # Drug ID resolution — very stable
    "plans": 24 * 3600,                   # CMS plan data — changes daily at most
    "npi_doctor": 7 * 24 * 3600,         # Doctor network — weekly refresh ok
    "formulary": 24 * 3600,              # Drug formulary per plan — daily
    "medicaid_threshold": 30 * 24 * 3600, # Medicaid income thresholds — monthly
    "default": 6 * 3600,                  # Default — 6 hours
}

_stats = {
    "hits": 0,
    "misses": 0,
    "total_calls": 0,
}

def _make_key(namespace: str, params: dict) -> str:
    param_str = json.dumps(params, sort_keys=True)
    hash_val = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return f"{namespace}:{hash_val}"

def get(namespace: str, params: dict) -> Optional[Any]:
    _stats["total_calls"] += 1
    key = _make_key(namespace, params)
    
    if key in _cache:
        entry = _cache[key]
        if time.time() < entry["expires_at"]:
            _stats["hits"] += 1
            _cache[key]["hits"] += 1
            return entry["data"]
        else:
            del _cache[key]
    
    _stats["misses"] += 1
    return None

def set(namespace: str, params: dict, data: Any) -> None:
    key = _make_key(namespace, params)
    ttl = TTL_CONFIG.get(namespace, TTL_CONFIG["default"])
    _cache[key] = {
        "data": data,
        "expires_at": time.time() + ttl,
        "hits": 0,
        "namespace": namespace,
        "cached_at": datetime.utcnow().isoformat(),
    }

def get_cache_stats() -> dict:
    total = _stats["total_calls"]
    hit_rate = (_stats["hits"] / total * 100) if total > 0 else 0
    return {
        "total_calls": total,
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "hit_rate_pct": round(hit_rate, 1),
        "cached_entries": len(_cache),
	"namespaces": list({v["namespace"] for v in _cache.values()}),
     }

def cached_call(namespace: str, params: dict, fetch_fn):
    """Helper: check cache first, call fetch_fn on miss, store result."""
    result = get(namespace, params)
    if result is not None:
        return result
    result = fetch_fn()
    if result is not None:
        set(namespace, params, result)
    return result
