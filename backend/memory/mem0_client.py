"""
mem0 Memory Client - Persistent cross-session user memory
Stores user profile facts so returning users skip intake questions
"""

import os
from typing import Optional

try:
    from mem0 import Memory
    _config = {
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": "coverwise_users", "path": "./data/mem0_chroma"}
        }
    }
    _mem0 = Memory.from_config(_config)
    MEM0_AVAILABLE = True
except Exception:
    _mem0 = None
    MEM0_AVAILABLE = False

_fallback_store: dict = {}


def store_user_profile(user_id: str, profile: dict) -> None:
    facts = []
    if profile.get("zip_code"):
        facts.append("User lives in ZIP code " + profile["zip_code"])
    if profile.get("age"):
        facts.append("User is " + str(profile["age"]) + " years old")
    if profile.get("income"):
        facts.append("User annual income is $" + f"{profile['income']:,.0f}")
    if profile.get("household_size"):
        facts.append("User household size is " + str(profile["household_size"]))
    if profile.get("drugs"):
        facts.append("User takes: " + ", ".join(profile["drugs"]))
    if profile.get("doctors"):
        facts.append("User doctors: " + ", ".join(profile["doctors"]))

    memory_text = ". ".join(facts)

    if MEM0_AVAILABLE and _mem0:
        try:
            _mem0.add(memory_text, user_id=user_id)
        except Exception as e:
            print("mem0 store error: " + str(e))
            _fallback_store[user_id] = facts
    else:
        _fallback_store[user_id] = facts


def get_user_memories(user_id: str) -> list:
    if MEM0_AVAILABLE and _mem0:
        try:
            results = _mem0.get_all(user_id=user_id)
            return [r.get("memory", "") for r in results]
        except Exception as e:
            print("mem0 retrieve error: " + str(e))
    return _fallback_store.get(user_id, [])


def build_memory_context(user_id: str) -> str:
    memories = get_user_memories(user_id)
    if not memories:
        return ""
    memory_lines = "\n".join("- " + m for m in memories)
    return "## Returning User Memory (from mem0)\n" + memory_lines + "\n"


def search_user_memory(user_id: str, query: str) -> list:
    if MEM0_AVAILABLE and _mem0:
        try:
            results = _mem0.search(query, user_id=user_id)
            return [r.get("memory", "") for r in results]
        except Exception as e:
            print("mem0 search error: " + str(e))
    return _fallback_store.get(user_id, [])
