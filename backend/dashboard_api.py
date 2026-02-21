"""Dashboard API router for workout history, trends, and Gemini guidelines."""

from __future__ import annotations

import os
import time
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from bson import ObjectId

try:
    from db import get_collection, make_workout_summary_doc, make_guideline_doc
except ImportError:
    from backend.db import get_collection, make_workout_summary_doc, make_guideline_doc

try:
    from auth import get_current_user
except ImportError:
    from backend.auth import get_current_user


router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY", "")

# Rate limit for guideline generation
_guidelines_lock = threading.Lock()
_last_guideline_call: float = 0.0
_MIN_GUIDELINE_INTERVAL = 3.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GuidelineRequest(BaseModel):
    injury_type: str
    severity: str
    location: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_doc(doc: dict) -> dict:
    """Convert MongoDB _id to string id and datetime to ISO string."""
    from datetime import datetime
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    for key, val in doc.items():
        if isinstance(val, datetime):
            doc[key] = val.isoformat()
    return doc


async def save_workout_summary(
    user_id: str,
    manager: Any,
    duration_sec: float,
    video_name: str | None = None,
) -> str | None:
    """Extract data from a SubjectManager and save a workout summary.

    Returns the inserted document ID or None if there was nothing to save.
    """
    # Collect data across all analyzers
    all_clusters: dict = {}
    all_injury_risks: list = []
    all_activity_labels: dict = {}
    biomechanics_samples: list = []
    fatigue_timeline: dict | None = None
    form_scores: list = []
    guideline_name: str | None = None
    fatigue_score: float | None = None

    analyzers = getattr(manager, "_analyzers", {})
    if not analyzers:
        return None

    has_data = False
    for track_id, analyzer in analyzers.items():
        summary = analyzer.get_final_summary()
        if summary.get("valid_frames", 0) < 10:
            continue
        has_data = True

        # Merge cluster summaries
        for cid, cinfo in summary.get("cluster_summary", {}).items():
            key = f"{track_id}_{cid}"
            all_clusters[key] = cinfo

        # Activity labels
        labels = getattr(analyzer, "_activity_labels", {})
        for cid, label in labels.items():
            all_activity_labels[f"{track_id}_{cid}"] = label

        # Quality tracker data
        qt = getattr(analyzer, "_quality_tracker", None)
        if qt is None:
            continue

        # Injury risks from latest frame quality
        fq = qt.get_frame_quality()
        risks = fq.get("injury_risks", [])
        for r in risks:
            all_injury_risks.append(r)

        # Form score
        fs = fq.get("form_score")
        if fs is not None:
            form_scores.append(fs)

        # Guideline name
        ag = fq.get("active_guideline")
        if ag and guideline_name is None:
            guideline_name = ag.get("name")

        # Fatigue timeline (use the longest one)
        ft_ts = getattr(qt, "_fatigue_timeline_timestamps", [])
        ft_fat = getattr(qt, "_fatigue_timeline_fatigue", [])
        ft_form = getattr(qt, "_fatigue_timeline_form", [])
        if len(ft_ts) > 0:
            current_ft = {
                "timestamps": list(ft_ts),
                "fatigue": list(ft_fat),
                "form_scores": list(ft_form),
            }
            if fatigue_timeline is None or len(ft_ts) > len(fatigue_timeline.get("timestamps", [])):
                fatigue_timeline = current_ft

        # Biomechanics timeline: sample from internal state
        biomech_hist = getattr(qt, "_biomechanics_history", [])
        if biomech_hist:
            biomechanics_samples.extend(biomech_hist)

        # Fatigue score: use last fatigue value
        if ft_fat:
            fatigue_score = ft_fat[-1]

    if not has_data:
        return None

    form_score_avg = round(sum(form_scores) / len(form_scores), 1) if form_scores else None

    doc = make_workout_summary_doc(
        user_id=user_id,
        duration_sec=duration_sec,
        clusters=all_clusters,
        injury_risks=all_injury_risks,
        fatigue_score=fatigue_score,
        video_name=video_name,
        activity_labels=all_activity_labels if all_activity_labels else None,
        biomechanics_timeline=biomechanics_samples if biomechanics_samples else None,
        fatigue_timeline=fatigue_timeline,
        form_score_avg=form_score_avg,
        guideline_name=guideline_name,
    )

    coll = get_collection("workout_summaries")
    result = await coll.insert_one(doc)
    return str(result.inserted_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/workouts")
async def list_workouts(
    user: dict = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List user's workouts, paginated and sorted by date descending."""
    coll = get_collection("workout_summaries")
    user_id = str(user["_id"])

    cursor = coll.find({"user_id": user_id}).sort("created_at", -1).skip(offset).limit(limit)
    workouts = []
    async for doc in cursor:
        # Build compact list item
        activity = "unknown"
        labels = doc.get("activity_labels") or {}
        if labels:
            # Most common label
            from collections import Counter
            counts = Counter(labels.values())
            activity = counts.most_common(1)[0][0] if counts else "unknown"

        risk_count = len(doc.get("injury_risks") or [])

        workouts.append({
            "id": str(doc["_id"]),
            "created_at": doc["created_at"].isoformat(),
            "duration_sec": doc.get("duration_sec", 0),
            "activity": activity,
            "form_score_avg": doc.get("form_score_avg"),
            "fatigue_score": doc.get("fatigue_score"),
            "risk_count": risk_count,
            "video_name": doc.get("video_name"),
        })

    total = await coll.count_documents({"user_id": user_id})
    return {"workouts": workouts, "total": total}


@router.get("/workouts/{workout_id}")
async def get_workout(
    workout_id: str,
    user: dict = Depends(get_current_user),
):
    """Get full workout detail."""
    coll = get_collection("workout_summaries")
    try:
        doc = await coll.find_one({"_id": ObjectId(workout_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid workout ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Workout not found")
    if doc.get("user_id") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Access denied")

    return _serialize_doc(doc)


@router.get("/trends")
async def get_trends(
    user: dict = Depends(get_current_user),
    n: int = Query(20, ge=1, le=100),
):
    """Aggregated trends across last N workouts."""
    coll = get_collection("workout_summaries")
    user_id = str(user["_id"])

    cursor = coll.find({"user_id": user_id}).sort("created_at", -1).limit(n)

    dates: list[str] = []
    form_scores: list[float | None] = []
    fatigue_scores: list[float | None] = []
    injury_counts: list[int] = []

    async for doc in cursor:
        dates.append(doc["created_at"].isoformat())
        form_scores.append(doc.get("form_score_avg"))
        fatigue_scores.append(doc.get("fatigue_score"))
        injury_counts.append(len(doc.get("injury_risks") or []))

    # Reverse to chronological order
    dates.reverse()
    form_scores.reverse()
    fatigue_scores.reverse()
    injury_counts.reverse()

    return {
        "dates": dates,
        "form_scores": form_scores,
        "fatigue_scores": fatigue_scores,
        "injury_counts": injury_counts,
    }


@router.get("/injury-history")
async def get_injury_history(
    user: dict = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
):
    """All injury risk events across sessions."""
    coll = get_collection("workout_summaries")
    user_id = str(user["_id"])

    cursor = coll.find(
        {"user_id": user_id, "injury_risks": {"$ne": []}},
    ).sort("created_at", -1).limit(limit)

    events: list[dict] = []
    async for doc in cursor:
        workout_id = str(doc["_id"])
        workout_date = doc["created_at"].isoformat()
        for risk in (doc.get("injury_risks") or []):
            events.append({
                "workout_id": workout_id,
                "workout_date": workout_date,
                "joint": risk.get("joint", "unknown"),
                "risk": risk.get("risk", "unknown"),
                "severity": risk.get("severity", "low"),
                "value": risk.get("value"),
                "threshold": risk.get("threshold"),
            })

    return {"events": events}


@router.post("/guidelines")
async def get_guidelines(
    req: GuidelineRequest,
    user: dict = Depends(get_current_user),
):
    """On-demand Gemini research for an injury. Returns cached or generates new."""
    injury_type = req.injury_type.strip().lower()
    severity = req.severity.strip().lower()
    location = req.location.strip().lower()

    if not injury_type or not severity or not location:
        raise HTTPException(status_code=400, detail="All fields are required")

    coll = get_collection("gemini_guidelines")

    # Check cache
    cached = await coll.find_one({
        "injury_type": injury_type,
        "injury_severity": severity,
        "injury_location": location,
    })
    if cached:
        return _serialize_doc(cached).get("content", {})

    # Rate limit
    global _last_guideline_call
    with _guidelines_lock:
        now = time.monotonic()
        wait = _MIN_GUIDELINE_INTERVAL - (now - _last_guideline_call)
        if wait > 0:
            import asyncio
            await asyncio.sleep(wait)
        _last_guideline_call = time.monotonic()

    # Call Gemini
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = (
            f"You are a sports medicine expert. Provide evidence-based rehabilitation guidelines for:\n"
            f"- Injury: {injury_type}\n"
            f"- Severity: {severity}\n"
            f"- Location: {location}\n\n"
            f"Respond in JSON format with these exact keys:\n"
            f'{{"summary": "brief overview of the injury and recovery outlook",'
            f'"safe_rom": [{{"joint": "name", "motion": "flexion/extension/etc", "min_degrees": 0, "max_degrees": 90}}],'
            f'"red_flags": ["list of warning signs requiring medical attention"],'
            f'"rehab_protocols": [{{"phase": "name", "duration": "2-4 weeks", "goals": "description", "exercises": ["list"]}}],'
            f'"recommended_exercises": [{{"name": "exercise name", "sets": "3", "reps": "10-12", "notes": "key cues"}}],'
            f'"activities_to_avoid": ["list of activities to avoid during recovery"],'
            f'"references": ["published guidelines or textbook references"]}}\n\n'
            f"Return ONLY valid JSON, no markdown fences."
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[prompt],
        )

        import json
        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
        if text.startswith("json"):
            text = text[4:].strip()

        content = json.loads(text)

    except Exception as e:
        print(f"[dashboard] Gemini guidelines failed: {e}", flush=True)
        raise HTTPException(status_code=502, detail="Failed to generate guidelines")

    # Cache in MongoDB
    doc = make_guideline_doc(injury_type, severity, location, content)
    try:
        await coll.insert_one(doc)
    except Exception:
        pass  # Duplicate key race condition — fine

    return content
