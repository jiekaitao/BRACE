"""Dashboard API endpoints for workout history, trends, and guidelines.

Provides REST endpoints for the frontend dashboard to retrieve
workout summaries, biomechanical trends, and personalized guidelines.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Workout History
# ---------------------------------------------------------------------------

@router.get("/workouts")
async def list_workouts(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List recent workout summaries for a user."""
    try:
        from db import get_collection
        workouts = get_collection("workout_summaries")
        cursor = (
            workouts.find({"user_id": user_id})
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return {"workouts": results, "count": len(results)}
    except Exception as e:
        return {"error": str(e), "workouts": []}


@router.get("/workouts/{workout_id}")
async def get_workout(workout_id: str):
    """Get a single workout summary by ID."""
    try:
        from bson import ObjectId
        from db import get_collection
        workouts = get_collection("workout_summaries")
        doc = await workouts.find_one({"_id": ObjectId(workout_id)})
        if doc is None:
            return {"error": "Workout not found"}
        doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Biomechanical Trends
# ---------------------------------------------------------------------------

@router.get("/trends")
async def get_trends(
    user_id: str = Query(..., description="User ID"),
    metric: str = Query("fatigue_score", description="Metric to trend"),
    limit: int = Query(30, ge=1, le=100),
):
    """Get biomechanical trends over recent workouts.

    Returns time-series data for the specified metric.
    """
    try:
        from db import get_collection
        workouts = get_collection("workout_summaries")
        cursor = (
            workouts.find(
                {"user_id": user_id},
                {"created_at": 1, metric: 1, "duration_sec": 1, "video_name": 1},
            )
            .sort("created_at", -1)
            .limit(limit)
        )
        points = []
        async for doc in cursor:
            value = doc.get(metric)
            if value is not None:
                points.append({
                    "date": doc["created_at"].isoformat() if doc.get("created_at") else None,
                    "value": value,
                    "duration_sec": doc.get("duration_sec"),
                    "video_name": doc.get("video_name"),
                })
        # Return in chronological order
        points.reverse()
        return {"metric": metric, "points": points}
    except Exception as e:
        return {"error": str(e), "points": []}


@router.get("/trends/injury-risks")
async def get_injury_risk_trends(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(30, ge=1, le=100),
):
    """Get injury risk frequency trends over recent workouts."""
    try:
        from db import get_collection
        workouts = get_collection("workout_summaries")
        cursor = (
            workouts.find(
                {"user_id": user_id},
                {"created_at": 1, "injury_risks": 1},
            )
            .sort("created_at", -1)
            .limit(limit)
        )
        points = []
        async for doc in cursor:
            risks = doc.get("injury_risks", [])
            risk_counts: dict[str, int] = {}
            for risk in risks:
                name = risk.get("risk_name", "unknown") if isinstance(risk, dict) else str(risk)
                risk_counts[name] = risk_counts.get(name, 0) + 1
            points.append({
                "date": doc["created_at"].isoformat() if doc.get("created_at") else None,
                "total_risks": len(risks),
                "risk_counts": risk_counts,
            })
        points.reverse()
        return {"points": points}
    except Exception as e:
        return {"error": str(e), "points": []}


# ---------------------------------------------------------------------------
# Personalized Guidelines
# ---------------------------------------------------------------------------

@router.get("/guidelines")
async def get_guidelines(
    user_id: str = Query(..., description="User ID"),
    activity: str = Query("", description="Filter by activity"),
):
    """Get personalized movement guidelines for a user."""
    try:
        from db import get_collection
        guidelines = get_collection("guidelines")
        query: dict[str, Any] = {"user_id": user_id}
        if activity:
            query["activity"] = activity
        cursor = guidelines.find(query).sort("created_at", -1)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return {"guidelines": results}
    except Exception as e:
        return {"error": str(e), "guidelines": []}


@router.post("/guidelines")
async def create_guideline(body: dict):
    """Create or update a personalized guideline."""
    try:
        from db import get_collection, make_guideline_doc
        guidelines = get_collection("guidelines")
        doc = make_guideline_doc(
            user_id=body["user_id"],
            activity=body["activity"],
            guidelines=body["guidelines"],
            injury_context=body.get("injury_context"),
        )
        # Upsert: replace existing guideline for same user+activity
        result = await guidelines.replace_one(
            {"user_id": body["user_id"], "activity": body["activity"]},
            doc,
            upsert=True,
        )
        return {"status": "ok", "upserted": result.upserted_id is not None}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Game-specific Dashboard
# ---------------------------------------------------------------------------

@router.get("/games")
async def list_games(
    user_id: str = Query("", description="User ID (optional)"),
    limit: int = Query(20, ge=1, le=100),
):
    """List recent game analyses."""
    try:
        from db import get_collection
        games = get_collection("games")
        query: dict[str, Any] = {}
        if user_id:
            query["user_id"] = user_id
        cursor = games.find(query).sort("created_at", -1).limit(limit)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return {"games": results}
    except Exception as e:
        return {"error": str(e), "games": []}


@router.get("/games/{game_id}/players")
async def game_players(game_id: str):
    """List players detected in a game."""
    try:
        from db import get_collection
        players = get_collection("game_players")
        cursor = players.find({"game_id": game_id}).sort("subject_id", 1)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return {"players": results}
    except Exception as e:
        return {"error": str(e), "players": []}


@router.get("/games/{game_id}/players/{subject_id}")
async def game_player_detail(game_id: str, subject_id: int):
    """Get detailed analysis for a specific player in a game."""
    try:
        from db import get_collection
        players = get_collection("game_players")
        player = await players.find_one(
            {"game_id": game_id, "subject_id": subject_id}
        )
        if player is None:
            return {"error": "Player not found"}
        player["_id"] = str(player["_id"])

        # Get frame-level data for biomechanics charts
        frames = get_collection("player_frames")
        frame_cursor = (
            frames.find(
                {"game_id": game_id, "subject_id": subject_id},
                {"quality": 1, "video_time": 1, "frame_index": 1, "activity_label": 1},
            )
            .sort("frame_index", 1)
            .limit(5000)
        )
        frame_data = []
        async for fdoc in frame_cursor:
            fdoc.pop("_id", None)
            frame_data.append(fdoc)

        player["frame_data"] = frame_data
        return player
    except Exception as e:
        return {"error": str(e)}
