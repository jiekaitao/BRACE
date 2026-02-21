"""Username-based authentication with session tokens stored in MongoDB."""

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from bson import ObjectId

try:
    from db import get_collection, make_user_doc
except ImportError:
    from backend.db import get_collection, make_user_doc

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

import re

_USERNAME_RE = re.compile(r"^[a-z]+$")


def _validate_username(username: str) -> str:
    """Strip, lowercase, and enforce lowercase-letters-only."""
    username = username.strip().lower()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if not _USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must contain only lowercase letters (a-z), no numbers or special characters",
        )
    return username


class RegisterRequest(BaseModel):
    username: str


class LoginRequest(BaseModel):
    username: str


class AuthResponse(BaseModel):
    user_id: str
    username: str
    token: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    injury_profile: dict | None = None
    risk_modifiers: dict | None = None


class ProfileUpdateRequest(BaseModel):
    injury_profile: dict | None = None
    risk_modifiers: dict | None = None


# ---------------------------------------------------------------------------
# User CRUD (async, motor)
# ---------------------------------------------------------------------------

async def create_user(username: str) -> dict:
    """Insert a new user document, return it with _id."""
    coll = get_collection("users")
    doc = make_user_doc(username)
    result = await coll.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def get_user(username: str) -> dict | None:
    """Find a user by username."""
    coll = get_collection("users")
    return await coll.find_one({"username": username})


async def get_user_by_id(user_id: str) -> dict | None:
    """Find a user by _id (string ObjectId)."""
    coll = get_collection("users")
    return await coll.find_one({"_id": ObjectId(user_id)})


# ---------------------------------------------------------------------------
# Session tokens
# ---------------------------------------------------------------------------

async def create_session(user_id: str) -> str:
    """Generate a UUID token, store in sessions collection, return token."""
    coll = get_collection("sessions")
    token = str(uuid.uuid4())
    await coll.insert_one({
        "token": token,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc),
    })
    return token


async def validate_session(token: str) -> str | None:
    """Lookup token in sessions, return user_id if found."""
    coll = get_collection("sessions")
    doc = await coll.find_one({"token": token})
    if doc is None:
        return None
    return doc["user_id"]


async def delete_session(token: str) -> None:
    """Delete a session token."""
    coll = get_collection("sessions")
    await coll.delete_one({"token": token})


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_current_user(authorization: str = Header(None)) -> dict:
    """Extract user from Authorization: Bearer <token> header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    user_id = await validate_session(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """Register a new user. Duplicate username returns 409."""
    username = _validate_username(req.username)
    existing = await get_user(username)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username already taken")
    user = await create_user(username)
    token = await create_session(str(user["_id"]))
    return AuthResponse(
        user_id=str(user["_id"]),
        username=username,
        token=token,
    )


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Login by username. Auto-creates user if not found."""
    username = _validate_username(req.username)
    user = await get_user(username)
    if user is None:
        user = await create_user(username)
    token = await create_session(str(user["_id"]))
    return AuthResponse(
        user_id=str(user["_id"]),
        username=user["username"],
        token=token,
    )


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    """Return current user info."""
    return UserResponse(
        user_id=str(user["_id"]),
        username=user["username"],
        injury_profile=user.get("injury_profile"),
        risk_modifiers=user.get("risk_modifiers"),
    )


@router.post("/logout")
async def logout(user: dict = Depends(get_current_user), authorization: str = Header(None)):
    """Invalidate the current session token."""
    token = authorization.split(" ", 1)[1]
    await delete_session(token)
    return {"ok": True}


@router.get("/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    """Return the user's injury profile and risk modifiers."""
    return {
        "injury_profile": user.get("injury_profile"),
        "risk_modifiers": user.get("risk_modifiers"),
    }


@router.put("/profile")
async def update_profile(
    req: ProfileUpdateRequest,
    user: dict = Depends(get_current_user),
):
    """Update the user's injury profile and/or risk modifiers."""
    coll = get_collection("users")
    update_fields: dict[str, Any] = {}
    if req.injury_profile is not None:
        update_fields["injury_profile"] = req.injury_profile
    if req.risk_modifiers is not None:
        update_fields["risk_modifiers"] = req.risk_modifiers
    if update_fields:
        await coll.update_one(
            {"_id": user["_id"]},
            {"$set": update_fields},
        )
    return {"ok": True}
