"""Gemini-powered injury intake chat agent for BRACE onboarding."""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from db import get_collection
except ImportError:
    from backend.db import get_collection

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Gemini client (lazy-init, same pattern as gemini_classifier.py)
_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai
            api_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
            _genai_client = genai.Client(api_key=api_key)
        except ImportError:
            raise RuntimeError("google-genai package not installed")
    return _genai_client


# System prompt for the injury intake agent
INTAKE_SYSTEM_PROMPT = """You are BRACE, a friendly sports medicine intake assistant. Your job is to learn about the user's injury history through conversation.

BRACE is a real-time biomechanical analysis system that uses computer vision to track body movement and detect injury risk. Here's what BRACE can measure and why each metric matters:

**BRACE Measurement Capabilities:**
- **FPPA (Frontal Plane Projection Angle):** Measures knee valgus — how much the knee collapses inward during movement. Critical for ACL, knee, and ankle injuries. Normal: <15°, At-risk: >25°.
- **Hip Drop (Pelvic Obliquity):** Measures how much the pelvis tilts during single-leg stance. Important for hip, lower back, and gluteal weakness. Normal: <8°, At-risk: >12°.
- **Trunk Lean:** Measures lateral lean of the torso from vertical. Key for lower back injuries and core stability issues. Normal: <15°, At-risk: >25°.
- **Bilateral Asymmetry Index:** Compares left vs right joint angles to detect compensation patterns. Critical for hamstring strains and any unilateral injury. Normal: <15%, At-risk: >25%.
- **Angular Velocity:** Tracks how fast joints move (degrees/second). Important for shoulder, elbow, and general joint injuries to detect dangerous speed spikes. At-risk: >500°/s.

Ask about:
1. Previous injuries (ACL, shoulder, ankle, lower back, etc.)
2. Which side (left, right, or both)
3. Severity (mild, moderate, severe)
4. When it happened (acute = <6 weeks, chronic = >6 months, recovered = fully healed)
5. Current symptoms or limitations

Be conversational and empathetic. Ask one question at a time. When you have enough information, briefly explain which BRACE metrics are most relevant to their specific injuries (1-2 sentences per injury), then output the JSON profile.

After gathering sufficient information, output a JSON block with the extracted profile:
```json
{"injuries": [{"type": "acl", "side": "left", "severity": "severe", "timeframe": "chronic"}], "complete": true}
```

Injury types: acl, shoulder, ankle, lower_back, knee_general, hip, hamstring, wrist
Sides: left, right, bilateral, unknown
Severity: mild, moderate, severe
Timeframe: acute, chronic, recovered"""


class InjuryChatAgent:
    """Wraps Gemini 2.5 Pro for injury intake conversation."""

    def __init__(self):
        self._model = "gemini-2.5-pro"

    async def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send messages to Gemini and parse response.

        Args:
            messages: List of {role: "user"|"assistant", content: str}

        Returns:
            {response: str, extracted_profile: dict|None, profile_complete: bool}
        """
        client = _get_genai_client()

        # Build Gemini contents
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        # Call Gemini
        response = client.models.generate_content(
            model=self._model,
            contents=contents,
            config={"system_instruction": INTAKE_SYSTEM_PROMPT},
        )

        response_text = response.text

        # Try to extract JSON profile from response
        extracted_profile = None
        profile_complete = False

        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "injuries" in parsed:
                    extracted_profile = parsed
                    profile_complete = parsed.get("complete", False)
            except json.JSONDecodeError:
                pass

        return {
            "response": response_text,
            "extracted_profile": extracted_profile,
            "profile_complete": profile_complete,
        }


# ---------------------------------------------------------------------------
# Risk modifier mapping
# ---------------------------------------------------------------------------

@dataclass
class InjuryModifiers:
    """Threshold multipliers based on injury profile."""
    fppa_scale: float = 1.0
    hip_drop_scale: float = 1.0
    trunk_lean_scale: float = 1.0
    asymmetry_scale: float = 1.0
    angular_velocity_scale: float = 1.0
    monitor_joints: list[str] = field(default_factory=list)


def profile_to_risk_modifiers(injury_profile: dict | None) -> dict:
    """Convert injury profile to risk threshold multipliers.

    Lower multiplier = more sensitive (lower threshold).
    """
    if not injury_profile or "injuries" not in injury_profile:
        return asdict(InjuryModifiers())

    mods = InjuryModifiers()

    for injury in injury_profile["injuries"]:
        injury_type = injury.get("type", "")
        severity = injury.get("severity", "mild")

        # Severity factor: severe=0.65, moderate=0.8, mild=0.9
        factor = {"severe": 0.65, "moderate": 0.8, "mild": 0.9}.get(severity, 0.9)

        if injury_type == "acl":
            mods.fppa_scale *= factor
            mods.monitor_joints.extend(["left_knee", "right_knee"])
        elif injury_type == "shoulder":
            mods.angular_velocity_scale *= factor
            side = injury.get("side", "bilateral")
            if side in ("left", "bilateral"):
                mods.monitor_joints.append("left_elbow")
            if side in ("right", "bilateral"):
                mods.monitor_joints.append("right_elbow")
        elif injury_type == "lower_back":
            mods.trunk_lean_scale *= factor
            mods.hip_drop_scale *= factor
        elif injury_type == "ankle":
            mods.monitor_joints.extend(["left_ankle", "right_ankle"])
        elif injury_type == "hip":
            mods.hip_drop_scale *= factor
        elif injury_type == "hamstring":
            mods.asymmetry_scale *= factor
        elif injury_type == "knee_general":
            mods.fppa_scale *= factor

    return asdict(mods)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """You are a sports medicine research assistant for BRACE, a real-time biomechanical analysis system.

Given a user's injury profile, generate evidence-based guidelines specific to what BRACE can measure.

BRACE measures these metrics in real-time during exercise:
- **FPPA (Frontal Plane Projection Angle):** Knee valgus angle. Normal <15°, medium risk 15-25°, high risk >25°. Relevant for ACL, knee, and ankle injuries.
- **Hip Drop (Pelvic Obliquity):** Pelvis tilt during single-leg stance. Normal <8°, medium risk 8-12°, high risk >12°. Relevant for hip, lower back injuries.
- **Trunk Lean:** Lateral torso lean from vertical. Normal <15°, medium risk 15-25°, high risk >25°. Relevant for lower back injuries.
- **Bilateral Asymmetry Index:** Left vs right joint angle difference. Normal <15%, medium risk 15-25%, high risk >25%. Relevant for hamstring, unilateral injuries.
- **Angular Velocity:** Joint movement speed (deg/s). Risk >500°/s. Relevant for shoulder, elbow, joint injuries.

For each injury in the user's profile, return a JSON block:
```json
{
  "guidelines": [
    {
      "injury_type": "acl",
      "metric": "FPPA",
      "title": "Knee Alignment Monitoring",
      "explanation": "Why BRACE monitors this metric for this injury (2-3 sentences)",
      "precautions": ["Precaution 1", "Precaution 2", "Precaution 3"]
    }
  ],
  "summary": "A brief overall summary paragraph (2-3 sentences)"
}
```

Keep explanations concise and actionable. Focus on what BRACE can specifically monitor and flag."""


class ChatRequest(BaseModel):
    user_id: str | None = None
    messages: list[dict[str, str]]


class ChatResponse(BaseModel):
    response: str
    extracted_profile: dict | None = None
    profile_complete: bool = False


class ConfirmProfileRequest(BaseModel):
    user_id: str | None = None
    injury_profile: dict


class ResearchRequest(BaseModel):
    injury_profile: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# Singleton agent
_agent = InjuryChatAgent()


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Send messages to the injury intake chat agent."""
    if not req.messages:
        raise HTTPException(status_code=422, detail="Messages cannot be empty")

    result = await _agent.chat(req.messages)
    return ChatResponse(**result)


@router.post("/research")
async def research_endpoint(req: ResearchRequest):
    """Generate Gemini research guidelines for an injury profile."""
    if not req.injury_profile or not req.injury_profile.get("injuries"):
        return {"guidelines": [], "summary": "No injuries to research."}

    client = _get_genai_client()

    # Build a user message describing the injury profile
    injuries_desc = []
    for inj in req.injury_profile["injuries"]:
        desc = f"- {inj.get('type', 'unknown')}"
        if inj.get("side") and inj["side"] != "unknown":
            desc += f" ({inj['side']})"
        if inj.get("severity"):
            desc += f", {inj['severity']}"
        if inj.get("timeframe"):
            desc += f", {inj['timeframe']}"
        injuries_desc.append(desc)

    user_msg = f"Generate BRACE monitoring guidelines for this injury profile:\n" + "\n".join(injuries_desc)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": user_msg}]}],
        config={"system_instruction": RESEARCH_SYSTEM_PROMPT},
    )

    # Parse JSON from response
    response_text = response.text
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing the whole response as JSON
    try:
        parsed = json.loads(response_text)
        return parsed
    except json.JSONDecodeError:
        return {"guidelines": [], "summary": response_text[:500]}


@router.post("/confirm-profile")
async def confirm_profile(req: ConfirmProfileRequest):
    """Save confirmed injury profile to user's MongoDB document."""
    risk_modifiers = profile_to_risk_modifiers(req.injury_profile)

    if req.user_id:
        from bson import ObjectId
        users = get_collection("users")
        await users.update_one(
            {"_id": ObjectId(req.user_id)},
            {"$set": {
                "injury_profile": req.injury_profile,
                "risk_modifiers": risk_modifiers,
            }}
        )

    return {"ok": True, "risk_modifiers": risk_modifiers}


@router.post("/save-research")
async def save_research(
    body: dict,
):
    """Save research guidelines to user's MongoDB document."""
    user_id = body.get("user_id")
    guidelines = body.get("research_guidelines")
    if user_id and guidelines:
        from bson import ObjectId
        users = get_collection("users")
        await users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"research_guidelines": guidelines}}
        )
    return {"ok": True}
