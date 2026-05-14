"""
Intake Agent — Conversational intake using Vertex AI (Gemini) directly.
Works with OR without google-adk installed.

When google-adk is available (Cloud Run): uses the full ADK Runner with
tool-calling for structured field collection.

When google-adk is NOT available (local dev): falls back to a stateful
Gemini chat session via vertexai.generative_models — same UX, same
Gemini model, no ADK dependency required. Uses your Google Cloud credits.
"""

import os
import re
import json
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
except ImportError:
    ADK_AVAILABLE = False
    ToolContext = object

from memory.mem0_client import get_user_memories

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "coverwise-local")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

# Tell google-adk/google-genai to use Vertex AI ADC credentials, not a Gemini API key.
# Must be set before any ADK/genai code runs.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", REGION)

vertexai.init(project=PROJECT_ID, location=REGION)

APP_NAME = "CoverWise"
REQUIRED_FIELDS = ["zip_code", "age", "household_size", "income", "doctors", "drugs", "utilization", "tobacco_use"]

INTAKE_INSTRUCTION = """You are CoverWise's intake assistant. Your only job is to collect 7 pieces of
information from the user, confirm them, then hand off to the analysis pipeline.

STEP 1 — START OF EVERY CONVERSATION
Call `check_returning_user` immediately. Do not say anything before calling it.

STEP 2 — COLLECT FIELDS ONE AT A TIME
Fields: zip_code, age, household_size, income, doctors, drugs, utilization, tobacco_use
Ask ONE question at a time. Call store_field after each answer.

STEP 3 — PROFILE CONFIRMATION GATE
When all fields collected, call show_confirmation and ask user to confirm.

STEP 4 — TRIGGER ANALYSIS
After user confirms, call confirm_and_analyze.

RULES
- Ask ONE question per message.
- Accept natural language: "$50k" → 50000, "just me" → 1.
- Never skip the confirmation gate.
"""


# ── TOOLS (used by ADK path only) ─────────────────────────────────────────────

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
            return {"error": "Invalid ZIP code."}
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
        profile["drugs"] = [] if v in ("none", "no", "n/a", "nothing", "no medications", "nope") \
            else [d.strip() for d in re.split(r"[,;]", value) if d.strip()]
    elif field == "doctors":
        v = value.lower().strip()
        profile["doctors"] = [] if v in ("none", "no", "n/a", "no doctors", "nope") \
            else [d.strip() for d in re.split(r"[,;]", value) if d.strip()]
    elif field == "utilization":
        v = value.lower()
        if any(w in v for w in ("rare", "never", "0", "1")):
            profile["utilization"] = "rarely"
        elif any(w in v for w in ("sometimes", "occasional", "2", "3", "4")):
            profile["utilization"] = "sometimes"
        elif any(w in v for w in ("frequent", "often", "5", "6", "7", "8")):
            profile["utilization"] = "frequently"
        elif any(w in v for w in ("chronic", "regular", "ongoing", "always")):
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
    """Correct a previously stored field value."""
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
    """Show the collected profile for user confirmation."""
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


# ── ADK RUNNER (when google-adk is installed) ─────────────────────────────────

_session_service: Optional[object] = None
_runner: Optional[object] = None


def _ensure_runner():
    global _session_service, _runner
    if not ADK_AVAILABLE or _runner is not None:
        return
    agent = Agent(
        name="coverwise_intake",
        model="gemini-2.0-flash",
        description="Conversational intake agent for health insurance profile collection",
        instruction=INTAKE_INSTRUCTION,
        tools=[check_returning_user, store_field, update_field, get_profile,
               show_confirmation, confirm_and_analyze],
    )
    _session_service = InMemorySessionService()
    _runner = Runner(agent=agent, app_name=APP_NAME, session_service=_session_service)


async def start_session(user_id: str, session_id: str) -> dict:
    """Create a session and get the opening greeting from the intake agent."""
    if not ADK_AVAILABLE:
        msg = await _gemini_start_session(user_id, session_id)
        return {"session_id": session_id, "message": msg, "status": "started"}

    _ensure_runner()
    try:
        await _session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
            state={"user_id": user_id, "profile": {}})
    except Exception:
        _session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
            state={"user_id": user_id, "profile": {}})

    states_intro = (
        "Hi! I'm CoverWise — I'll find your best health plan in about 90 seconds. "
        "I support live plan analysis for 30 federal marketplace states (TX, FL, TN, GA, AZ, IL, "
        "OH, MI, NC, SC, AL, MS, AR, OK, KS, NE, IA, WI, MO, LA, SD, ND, MT, WY, UT, HI, AK, WV, NH, DE). "
        "For the 20 states with their own exchanges (NY, CA, WA, CO, CT, KY, ME, MD, MA, MN, NV, "
        "NJ, NM, PA, RI, VT, VA, DC, ID, OR), I will redirect you to your state exchange. "
        "What's your ZIP code?"
    )
    return {"session_id": session_id, "message": states_intro, "status": "started"}


async def send_message(user_id: str, session_id: str, message: str) -> dict:
    """Forward a user message to the intake agent and return its response."""
    if not ADK_AVAILABLE:
        return await _gemini_send_message(user_id, session_id, message)

    _ensure_runner()
    user_msg = Content(role="user", parts=[Part(text=message)])
    reply = await _collect_final_text(user_id, session_id, user_msg)

    profile_ready = False
    profile_data = None
    try:
        session = await _session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id)
    except Exception:
        session = _session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if session and session.state.get("analysis_ready"):
        profile_ready = True
        raw = session.state.get("profile", {})
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
    """Run the ADK agent and collect the final text response."""
    reply = ""
    async for event in _runner.run_async(user_id=user_id, session_id=session_id, new_message=msg):
        if hasattr(event, "is_final_response") and event.is_final_response():
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        reply = part.text
                        break
    return (reply or "I'm here to help you find a health plan. What's your ZIP code?").replace("*", "")


# ── GEMINI FALLBACK (no google-adk needed — uses vertexai directly) ───────────
#
# Stores conversation history in _gemini_sessions[session_id].
# When profile is complete and confirmed, Gemini outputs a PROFILE_READY: JSON
# marker that this code parses and hands off to the analysis pipeline.

_gemini_sessions: dict = {}

_GEMINI_SYSTEM = """You are CoverWise, a friendly health insurance advisor chatbot.
Your job: collect the following 8 fields from the user through natural conversation,
then confirm with the user, then output the profile.

Fields to collect:
- zip_code (5-digit ZIP)
- age (integer)
- household_size (integer, including user)
- income (annual household income before taxes, as a number)
- doctors (list of doctor names they want to keep, or empty list)
- drugs (list of prescription medications, or empty list)
- utilization: one of "rarely" / "sometimes" / "frequently" / "chronic"
- tobacco_use: true or false

Rules:
- Ask ONE question at a time. Short, friendly, conversational.
- Accept natural language: "$50k"=50000, "just me"=1, "a couple"=2, "none"=[]
- After collecting ALL 8 fields, show a summary like:
  "Here's what I have:
   ZIP: 77001 | Age: 35 | Household: 2 | Income: $45,000
   Medications: lisinopril | Doctors: Dr. Smith
   Healthcare use: sometimes | Tobacco: No
   Does this look correct?"
- After user confirms (yes/correct/looks good/go ahead), output EXACTLY this on its own line:
  PROFILE_READY:{"zip_code":"XXXXX","age":NN,"household_size":N,"income":NNNNN,"doctors":["..."],"drugs":["..."],"utilization":"sometimes","tobacco_use":false}
- income must be a plain number (no $ or commas)
- Do NOT output PROFILE_READY until the user has confirmed the summary.
- Do not mention "fields", "JSON", or technical terms to the user."""

_GEMINI_WELCOME = (
    "Hi! I'm CoverWise — I'll find your best health plan in about 90 seconds. "
    "I support live plan analysis for 30 federal marketplace states (TX, FL, TN, GA, AZ, IL, "
    "OH, MI, NC, SC, AL, MS, AR, OK, KS, NE, IA, WI, MO, LA, SD, ND, MT, WY, UT, HI, AK, WV, NH, DE). "
    "For the 20 states with their own exchanges (NY, CA, WA, CO, CT, KY, ME, MD, MA, MN, NV, "
    "NJ, NM, PA, RI, VT, VA, DC, ID, OR), I'll redirect you to your state exchange. "
    "What's your ZIP code?"
)


async def _gemini_start_session(user_id: str, session_id: str) -> str:
    """Initialise an in-memory Gemini chat session."""
    _gemini_sessions[session_id] = {
        "user_id": user_id,
        "history": [],
        "profile": {},
        "analysis_ready": False,
        "seeded": False,
    }
    return _GEMINI_WELCOME


async def _gemini_send_message(user_id: str, session_id: str, message: str) -> dict:
    """Send a message to the stateful Gemini chat via Vertex AI REST API (uses ADC — no API key needed)."""
    import google.auth
    import google.auth.transport.requests
    import urllib.request

    if session_id not in _gemini_sessions:
        await _gemini_start_session(user_id, session_id)

    sess = _gemini_sessions[session_id]

    # Build contents array: system prime + history + new user message
    contents = []
    if not sess["seeded"]:
        contents.append({"role": "user",  "parts": [{"text": _GEMINI_SYSTEM}]})
        contents.append({"role": "model", "parts": [{"text": _GEMINI_WELCOME}]})
        sess["seeded"] = True
    else:
        contents.append({"role": "user",  "parts": [{"text": _GEMINI_SYSTEM}]})
        contents.append({"role": "model", "parts": [{"text": _GEMINI_WELCOME}]})
        for turn in sess["history"]:
            contents.append({"role": turn["role"], "parts": [{"text": turn["text"]}]})

    contents.append({"role": "user", "parts": [{"text": message}]})

    def _call_vertex():
        # Get ADC token (works with `gcloud auth application-default login` locally
        # and with the Cloud Run service account automatically)
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        token = creds.token

        model_id = "gemini-2.0-flash-001"
        url = (
            f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
            f"/locations/{REGION}/publishers/google/models/{model_id}:generateContent"
        )
        payload = json.dumps({"contents": contents}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    data = await asyncio.to_thread(_call_vertex)

    # Extract text from response
    try:
        reply_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        reply_text = "Sorry, I had trouble understanding that. Could you try again?"

    # Persist turn
    sess["history"].append({"role": "user", "text": message})
    sess["history"].append({"role": "model", "text": reply_text})

    # Check for profile completion signal
    profile_ready = False
    profile_data = None

    if "PROFILE_READY:" in reply_text:
        try:
            json_str = reply_text.split("PROFILE_READY:", 1)[1].strip().splitlines()[0]
            raw = json.loads(json_str)
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
            profile_ready = True
            sess["analysis_ready"] = True
            sess["profile"] = profile_data
            # Strip the marker from what the user sees
            reply_text = reply_text.split("PROFILE_READY:")[0].strip()
            if not reply_text:
                reply_text = "Perfect — searching for every plan available in your area. This takes about 5 seconds..."
        except (json.JSONDecodeError, ValueError, KeyError):
            pass  # malformed — keep chatting

    return {
        "message": reply_text.replace("**", "").replace("*", ""),
        "profile_ready": profile_ready,
        "profile": profile_data,
        "session_id": session_id,
    }
