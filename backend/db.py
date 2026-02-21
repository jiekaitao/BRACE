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
    video_name: str | None = None,
) -> dict:
    """Create a workout summary document."""
    return {
        "user_id": user_id,
        "video_name": video_name,
        "duration_sec": duration_sec,
        "clusters": clusters,
        "injury_risks": injury_risks,
        "fatigue_score": fatigue_score,
        "created_at": datetime.now(timezone.utc),
    }
