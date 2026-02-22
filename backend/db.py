"""MongoDB client helpers for BRACE.

Provides async (motor) and sync (pymongo) access to the brace database.
Connection string is read from the MONGODB_URI environment variable.
"""

import os
from datetime import datetime, timezone

import motor.motor_asyncio
import pymongo

MONGODB_URI: str = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017/brace"
)


# ---------------------------------------------------------------------------
# Async helpers (motor)
# ---------------------------------------------------------------------------

def get_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    """Return a motor async client connected to MONGODB_URI."""
    return motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)


def get_db():
    """Return the default database from the motor client."""
    return get_client().get_default_database()


def get_collection(name: str):
    """Return a motor collection by name."""
    return get_db()[name]


# ---------------------------------------------------------------------------
# Sync helpers (pymongo) -- useful for tests and scripts
# ---------------------------------------------------------------------------

def get_sync_db():
    """Return the default database from a pymongo sync client."""
    client = pymongo.MongoClient(MONGODB_URI)
    return client.get_default_database()


def get_sync_collection(name: str):
    """Return a pymongo sync collection by name."""
    return get_sync_db()[name]


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------

async def ensure_indexes() -> None:
    """Create indexes on all collections. Safe to call repeatedly."""
    db = get_db()

    # users: unique on username
    await db["users"].create_index("username", unique=True)

    # chat_sessions: index on user_id for fast lookups
    await db["chat_sessions"].create_index("user_id")

    # workout_summaries: compound index for per-user time-ordered queries
    await db["workout_summaries"].create_index([
        ("user_id", 1),
        ("created_at", -1),
    ])

    # games: index on session_id and status
    await db["games"].create_index("session_id", unique=True)
    await db["games"].create_index("status")
    await db["games"].create_index([("user_id", 1), ("created_at", -1)])

    # game_players: compound index for per-game player lookups
    await db["game_players"].create_index([
        ("game_id", 1),
        ("subject_id", 1),
    ], unique=True)

    # player_frames: compound index for per-game, per-player frame data
    await db["player_frames"].create_index([
        ("game_id", 1),
        ("subject_id", 1),
        ("frame_index", 1),
    ])

    # guidelines: per-user activity guidelines
    await db["guidelines"].create_index([
        ("user_id", 1),
        ("activity", 1),
    ])


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def make_user_doc(username: str) -> dict:
    """Create a user document."""
    return {
        "username": username,
        "created_at": datetime.now(timezone.utc),
        "injury_profile": None,
        "risk_modifiers": None,
    }


def make_chat_session_doc(user_id: str) -> dict:
    """Create a chat session document."""
    now = datetime.now(timezone.utc)
    return {
        "user_id": user_id,
        "messages": [],
        "extracted_profile": None,
        "created_at": now,
        "updated_at": now,
    }


def make_workout_summary_doc(
    user_id: str,
    duration_sec: float,
    clusters: dict,
    injury_risks: list,
    fatigue_score: float | None = None,
    concussion_rating: float | None = None,
    video_name: str | None = None,
    game_id: str | None = None,
    activity_label: str | None = None,
) -> dict:
    """Create a workout summary document."""
    return {
        "user_id": user_id,
        "video_name": video_name,
        "duration_sec": duration_sec,
        "clusters": clusters,
        "injury_risks": injury_risks,
        "fatigue_score": fatigue_score,
        "concussion_rating": concussion_rating,
        "game_id": game_id,
        "activity_label": activity_label,
        "created_at": datetime.now(timezone.utc),
    }


def make_game_doc(
    session_id: str,
    video_name: str,
    sport: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Create a game analysis document."""
    return {
        "session_id": session_id,
        "video_name": video_name,
        "sport": sport,
        "user_id": user_id,
        "status": "pending",  # "pending" | "processing" | "complete" | "error"
        "player_count": 0,
        "total_frames": 0,
        "progress": 0.0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def make_game_player_doc(
    game_id: str,
    subject_id: int,
    label: str,
    jersey_number: int | None = None,
    jersey_color: str | None = None,
    risk_status: str = "GREEN",
    total_frames: int = 0,
    injury_events: list | None = None,
    workload: dict | None = None,
) -> dict:
    """Create a game player document."""
    return {
        "game_id": game_id,
        "subject_id": subject_id,
        "label": label,
        "jersey_number": jersey_number,
        "jersey_color": jersey_color,
        "risk_status": risk_status,
        "total_frames": total_frames,
        "injury_events": injury_events or [],
        "workload": workload or {},
        "created_at": datetime.now(timezone.utc),
    }


def make_player_frame_doc(
    game_id: str,
    subject_id: int,
    frame_index: int,
    video_time: float,
    quality: dict | None = None,
    activity_label: str | None = None,
    bbox: dict | None = None,
    velocity: float = 0.0,
) -> dict:
    """Create a per-frame player data document."""
    return {
        "game_id": game_id,
        "subject_id": subject_id,
        "frame_index": frame_index,
        "video_time": video_time,
        "quality": quality,
        "activity_label": activity_label,
        "bbox": bbox,
        "velocity": velocity,
    }


def make_guideline_doc(
    user_id: str,
    activity: str,
    guidelines: list[str],
    injury_context: str | None = None,
) -> dict:
    """Create an activity-specific guideline document."""
    return {
        "user_id": user_id,
        "activity": activity,
        "guidelines": guidelines,
        "injury_context": injury_context,
        "created_at": datetime.now(timezone.utc),
    }
