"""
Intake Agent — Google ADK conversational intake.
Collects 8 profile fields via natural back-and-forth conversation.
Returning users (Mem0) are pre-filled and only asked what changed.
Profile confirmation gate fires before any CMS API calls.
"""

import os
import re
import asyncio
import vertexai
from typing import Optional

try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import ToolContext
    from google.genai.types import Content, Part
    ADK_AVAILABLE = True
except Exception as _import_err:
    print(f"[ADK import error] {_import_err}")
    ADK_AVAILABLE = False
    ToolContext = object

from memory.mem0_client import get_user_memories

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "coverwise-local")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

if ADK_AVAILABLE:
    vertexai.init(project=PROJECT_ID, location=REGION)

APP_NAME = "CoverWise"
REQUIRED_FIELDS = ["zip_code", "age", "household_size", "income", "doctors", "drugs", "utilization", "tobacco_use"]

INTAKE_INSTRUCTION = """You are CoverWise's intake assistant. Your only job is to collect 7 pieces of
information from the user, confirm them, then hand off to the analysis pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — START OF EVERY CONVERSATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call `check_returning_user` immediately. Do not say anything before calling it.

• If returning user: greet warmly, show what you already know, ask ONLY what may have
  changed (income, medications, address). Example:
  "Welcome back! I have: ZIP 77001, age 35, household of 2, income $45,000, taking
  lisinopril, Dr. Smith. Anything changed since last year, or shall I run a fresh search
  with the same info?"

• If new user: introduce yourself in one sentence, then ask the first missing field.
  "Hi! I'm CoverWise — I'll find your best health plan in about 90 seconds. What's your
  ZIP code?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — COLLECT FIELDS ONE AT A TIME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After each answer, call `store_field` immediately, then ask the next missing field.
Call `get_profile` to see what's still needed.

Fields and how to ask them naturally:
• zip_code      → "What's your ZIP code?"
• age           → "How old are you?"
• household_size→ "How many people are in your household, including yourself?"
• income        → "What's your total household income before taxes this year?
                   (An estimate is fine — you can update it later.)"
• doctors       → "Is there a specific doctor you want to keep? Name them, or say 'none'."
• drugs         → "Any prescription medications? List them separated by commas, or say 'none'."
• utilization   → "How often do you typically use healthcare?
                   rarely (0–1 visits/year) / sometimes (2–4) / frequently (5+) / chronic"
• tobacco_use   → "Do you use tobacco? (yes/no)"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — HANDLE CORRECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the user corrects something ("actually my income is $72,000, not $68,000"), call
`update_field` and confirm: "Got it, updated to $72,000."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — PROFILE CONFIRMATION GATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When `get_profile` shows missing_fields is empty, call `show_confirmation`.
Show the structured summary and ask: "Does this look correct? I'll start searching
for plans once you confirm."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — TRIGGER ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After user confirms ("yes", "correct", "looks good", "go ahead", etc.), call
`confirm_and_analyze`. Say: "Perfect — searching for every plan available in your
area. This takes about 5 seconds..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Ask ONE question per message. Never combine two questions.
• Be conversational. Don't use jargon or mention field names.
• Accept natural language: "$50k" → 50000, "just me" → 1, "a couple" → 2.
• If user asks a clarifying question mid-intake, answer briefly then continue.
• Never skip the confirmation gate — wrong income can change subsidies by thousands.
"""


# ── TOOLS ─────────────────────────────────────────────────────────────────────

def check_returning_user(tool_context: "ToolContext") -> dict:
    """Check if this user_id has memories from a prior session in Mem0."""
    user_id = tool_context.state.get("user_id", "")
    if not user_id:
        tool_context.state["returning"] = False
        return {"returning": False}

    memories = get_user_memories(user_id)
    if not memories:
        tool_context.state["returning"] = False
        return {"returning": False}

    tool_context.state["returning"] = True
    tool_context.state["prior_memories"] = memories

    # Pre-fill profile from Mem0 facts
    profile = tool_context.state.get("profile", {})
    for mem in memories:
        if "zip" in mem.lower() and "zip_code" not in profile:
            m = re.search(r'\b(\d{5})\b', mem)
            if m:
                profile["zip_code"] = m.group(1)
        if "income" in mem.lower() and "income" not in profile:
            m = re.search(r'\$([0-9,]+)', mem)
            if m:
                try:
                    profile["income"] = float(m.group(1).replace(",", ""))
                except ValueError:
                    pass
        if "household size" in mem.lower() and "household_size" not in profile:
            m = re.search(r'household size is (\d+)', mem, re.IGNORECASE)
            if m:
                profile["household_size"] = int(m.group(1))
        if "years old" in mem.lower() and "age" not in profile:
            m = re.search(r'(\d+) years old', mem, re.IGNORECASE)
            if m:
                profile["age"] = int(m.group(1))
        if "takes:" in mem.lower() and "drugs" not in profile:
            parts = mem.split("takes:")
            if len(parts) > 1:
                profile["drugs"] = [d.strip() for d in parts[1].split(",") if d.strip()]
        if "doctors:" in mem.lower() and "doctors" not in profile:
            parts = mem.split("doctors:")
            if len(parts) > 1:
                profile["doctors"] = [d.strip() for d in parts[1].split(",") if d.strip()]
        if "uses tobacco:" in mem.lower() and "tobacco_use" not in profile:
            profile["tobacco_use"] = "yes" in mem.lower()

    tool_context.state["profile"] = profile
    missing = [f for f in REQUIRED_FIELDS if f not in profile]
    return {"returning": True, "memories": memories, "pre_filled": profile, "still_missing": missing}


def store_field(field: str, value: str, tool_context: "ToolContext") -> dict:
    """Store a single profile field, normalising the value."""
    profile = tool_context.state.get("profile", {})

    if field == "zip_code":
        clean = re.sub(r"[^\d]", "", value)
        if len(clean) not in (4, 5):
            return {"error": "Invalid ZIP code. Please provide a 5-digit ZIP code (or 4 digits for New England)."}
        profile["zip_code"] = clean.zfill(5)
    elif field == "income":
        clean = value.replace("$", "").replace(",", "").strip()
        if clean.lower().endswith("k"):
            clean = clean[:-1] + "000"
        try:
            profile["income"] = float(clean)
        except ValueError:
            profile["income"] = 50000.0
    elif field == "age":
        try:
            profile["age"] = int(float(re.sub(r"[^\d.]", "", value) or "35"))
        except ValueError:
            profile["age"] = 35
    elif field == "household_size":
        word_map = {"one": 1, "just me": 1, "only me": 1, "two": 2, "a couple": 2,
                    "three": 3, "four": 4, "five": 5}
        v = value.lower().strip()
        profile["household_size"] = word_map.get(v, int(float(re.sub(r"[^\d.]", "", value) or "1")))
    elif field == "drugs":
        v = value.lower().strip()
        if v in ("none", "no", "n/a", "nothing", "no medications", "nope"):
            profile["drugs"] = []
        else:
            profile["drugs"] = [d.strip() for d in re.split(r"[,;]", value) if d.strip()]
    elif field == "doctors":
        v = value.lower().strip()
        if v in ("none", "no", "n/a", "no doctors", "nope"):
            profile["doctors"] = []
        else:
            profile["doctors"] = [d.strip() for d in re.split(r"[,;]", value) if d.strip()]
    elif field == "utilization":
        v = value.lower()
        if any(w in v for w in ("rare", "never", "almost never", "0", "1")):
            profile["utilization"] = "rarely"
        elif any(w in v for w in ("sometimes", "occasional", "2", "3", "4")):
            profile["utilization"] = "sometimes"
        elif any(w in v for w in ("frequent", "often", "5", "6", "7", "8")):
            profile["utilization"] = "frequently"
        elif any(w in v for w in ("chronic", "regular", "ongoing", "always", "constant")):
            profile["utilization"] = "chronic"
        else:
            profile["utilization"] = "sometimes"
    elif field == "tobacco_use":
        v = value.lower().strip()
        profile["tobacco_use"] = any(w in v for w in ("yes", "yep", "yeah", "true", "smoke", "tobacco"))
    else:
        profile[field] = value

    tool_context.state["profile"] = profile
    missing = [f for f in REQUIRED_FIELDS if f not in profile]
    return {"stored": field, "value": profile.get(field), "missing_fields": missing}


def update_field(field: str, new_value: str, tool_context: "ToolContext") -> dict:
    """Correct a previously stored field value (live profile correction)."""
    result = store_field(field, new_value, tool_context)
    corrections = tool_context.state.get("corrections", [])
    corrections.append(field)
    tool_context.state["corrections"] = corrections
    return {**result, "corrected": True}


def get_profile(tool_context: "ToolContext") -> dict:
    """Return the current profile and list of still-missing fields."""
    profile = tool_context.state.get("profile", {})
    missing = [f for f in REQUIRED_FIELDS if f not in profile]
    return {"profile": profile, "missing_fields": missing, "complete": len(missing) == 0}


def show_confirmation(tool_context: "ToolContext") -> dict:
    """Show the collected profile for user confirmation — the human-in-the-loop gate."""
    profile = tool_context.state.get("profile", {})
    missing = [f for f in REQUIRED_FIELDS if f not in profile]
    if missing:
        return {"ready": False, "missing_fields": missing}

    tool_context.state["awaiting_confirmation"] = True
    return {
        "ready": True,
        "profile": profile,
        "summary": {
            "ZIP": profile.get("zip_code"),
            "Age": profile.get("age"),
            "Household size": profile.get("household_size"),
            "Annual income": f"${profile.get('income', 0):,.0f}",
            "Medications": ", ".join(profile.get("drugs", [])) or "None",
            "Doctors": ", ".join(profile.get("doctors", [])) or "None",
            "Healthcare use": profile.get("utilization", "sometimes"),
            "Tobacco use": "Yes" if profile.get("tobacco_use") else "No",
        },
    }


def confirm_and_analyze(tool_context: "ToolContext") -> dict:
    """User confirmed — mark profile ready for analysis pipeline."""
    tool_context.state["confirmed"] = True
    tool_context.state["analysis_ready"] = True
    profile = tool_context.state.get("profile", {})
    return {"confirmed": True, "profile": profile}


# ── ADK RUNNER ────────────────────────────────────────────────────────────────

_session_service: Optional[object] = None
_runner: Optional[object] = None

def _ensure_runner():
    global _session_service, _runner
    if not ADK_AVAILABLE:
        return
    if _runner is not None:
        return

    agent = Agent(
        name="coverwise_intake",
        model="gemini-2.5-flash",
        description="Conversational intake agent for health insurance profile collection",
        instruction=INTAKE_INSTRUCTION,
        tools=[
            check_returning_user,
            store_field,
            update_field,
            get_profile,
            show_confirmation,
            confirm_and_analyze,
        ],
    )
    _session_service = InMemorySessionService()
    _runner = Runner(agent=agent, app_name=APP_NAME, session_service=_session_service)


async def start_session(user_id: str, session_id: str) -> dict:
    """Create a session and get the opening greeting from the intake agent."""
    if not ADK_AVAILABLE:
        return {"session_id": session_id, "message": _fallback_start(user_id), "status": "started"}

    _ensure_runner()

    # Create session with user_id pre-seeded in state
    try:
        await _session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
            state={"user_id": user_id, "profile": {}}
        )
    except Exception:
        _session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
            state={"user_id": user_id, "profile": {}}
        )

    # Send a silent init trigger so the agent runs check_returning_user
    init_msg = Content(role="user", parts=[Part(text="start")])
    reply = await _collect_final_text(user_id, session_id, init_msg)

    return {"session_id": session_id, "message": reply, "status": "started"}


async def send_message(user_id: str, session_id: str, message: str) -> dict:
    """Forward a user message to the intake agent and return its response."""
    if not ADK_AVAILABLE:
        return {"message": _fallback_message(message), "profile_ready": False, "profile": None}

    _ensure_runner()
    user_msg = Content(role="user", parts=[Part(text=message)])
    reply = await _collect_final_text(user_id, session_id, user_msg)

    # Check session state for analysis readiness
    profile_ready = False
    profile_data = None
    try:
        session = await _session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception:
        session = _session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    if session and session.state.get("analysis_ready"):
        profile_ready = True
        raw = session.state.get("profile", {})
        # Ensure required fields have correct types before handing off
        profile_data = {
            "user_id": user_id,
            "zip_code": str(raw.get("zip_code", "")),
            "age": int(raw.get("age", 35)),
            "income": float(raw.get("income", 50000)),
            "household_size": int(raw.get("household_size", 1)),
            "drugs": raw.get("drugs", []),
            "doctors": raw.get("doctors", []),
            "utilization": raw.get("utilization", "sometimes"),
            "tobacco_use": bool(raw.get("tobacco_use", False)),
        }

    return {"message": reply, "profile_ready": profile_ready, "profile": profile_data, "session_id": session_id}


async def _collect_final_text(user_id: str, session_id: str, msg: "Content") -> str:
    """Run the agent and collect the final text response."""
    reply = ""
    async for event in _runner.run_async(user_id=user_id, session_id=session_id, new_message=msg):
        if hasattr(event, "is_final_response") and event.is_final_response():
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        reply = part.text
                        break
    return (reply or "I'm here to help you find a health plan. What's your ZIP code?").replace("*", "")


# ── FALLBACKS (when ADK not installed) ───────────────────────────────────────

def _fallback_start(user_id: str) -> str:
    memories = get_user_memories(user_id)
    if memories:
        return (
            "Welcome back! I'll pre-fill your profile from last time. "
            "Please confirm or update your details in the form below."
        )
    return "Hi! I'm CoverWise. Fill out the form below to find your best health insurance plan."


def _fallback_message(message: str) -> str:
    return (
        "The conversational intake requires the google-adk package. "
        "Please use the form to enter your details, or install google-adk and restart."
    )
