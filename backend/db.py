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

    # games: index on status and created_at for listing/filtering
    await db["games"].create_index("status")
    await db["games"].create_index("created_at")

    # game_players: unique compound on (game_id, subject_id), index on (game_id, team_id)
    await db["game_players"].create_index(
        [("game_id", 1), ("subject_id", 1)], unique=True,
    )
    await db["game_players"].create_index([("game_id", 1), ("team_id", 1)])

    # player_frames: compound on (game_id, subject_id, frame_idx), TTL 30 days
    await db["player_frames"].create_index(
        [("game_id", 1), ("subject_id", 1), ("frame_idx", 1)],
    )
    await db["player_frames"].create_index(
        "created_at", expireAfterSeconds=30 * 24 * 3600,
    )

    # gemini_guidelines: unique compound for cached guideline lookups
    await db["gemini_guidelines"].create_index(
        [("injury_type", 1), ("injury_severity", 1), ("injury_location", 1)],
        unique=True,
    )


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


def make_game_doc(session_id: str, video_path: str) -> dict:
    """Create a game document for basketball analysis."""
    return {
        "session_id": session_id,
        "video_path": video_path,
        "status": "queued",          # queued → processing → complete | failed
        "progress": 0.0,
        "frame_idx": 0,
        "total_frames": 0,
        "player_count": 0,
        "team_colors": [],           # [{team_id, color_name, rgb}]
        "duration_sec": 0.0,
        "error": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def make_game_player_doc(
    game_id: str,
    subject_id: int,
    jersey_number: int | None = None,
    team_id: int | None = None,
    jersey_color: str | None = None,
) -> dict:
    """Create a game player document."""
    return {
        "game_id": game_id,
        "subject_id": subject_id,
        "jersey_number": jersey_number,
        "team_id": team_id,
        "jersey_color": jersey_color,
        "injury_events": [],
        "final_quality": None,
        "analysis_summary": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def make_player_frame_doc(
    game_id: str,
    subject_id: int,
    frame_idx: int,
    quality: dict | None = None,
    biomechanics: dict | None = None,
) -> dict:
    """Create a player frame snapshot document."""
    return {
        "game_id": game_id,
        "subject_id": subject_id,
        "frame_idx": frame_idx,
        "quality": quality,
        "biomechanics": biomechanics,
        "created_at": datetime.now(timezone.utc),
    }


def make_workout_summary_doc(
    user_id: str,
    duration_sec: float,
    clusters: dict,
    injury_risks: list,
    fatigue_score: float | None = None,
    video_name: str | None = None,
    activity_labels: dict | None = None,
    biomechanics_timeline: list | None = None,
    fatigue_timeline: dict | None = None,
    form_score_avg: float | None = None,
    guideline_name: str | None = None,
) -> dict:
    """Create a workout summary document."""
    return {
        "user_id": user_id,
        "video_name": video_name,
        "duration_sec": duration_sec,
        "clusters": clusters,
        "injury_risks": injury_risks,
        "fatigue_score": fatigue_score,
        "activity_labels": activity_labels,
        "biomechanics_timeline": biomechanics_timeline,
        "fatigue_timeline": fatigue_timeline,
        "form_score_avg": form_score_avg,
        "guideline_name": guideline_name,
        "created_at": datetime.now(timezone.utc),
    }


def make_guideline_doc(
    injury_type: str,
    injury_severity: str,
    injury_location: str,
    content: dict,
) -> dict:
    """Create a cached Gemini guideline document."""
    return {
        "injury_type": injury_type,
        "injury_severity": injury_severity,
        "injury_location": injury_location,
        "content": content,
        "created_at": datetime.now(timezone.utc),
    }
