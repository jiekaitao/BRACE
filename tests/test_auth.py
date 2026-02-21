"""Tests for username-based authentication: auth.py endpoints and helpers.

All tests use mocking -- no running MongoDB instance required.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ---------------------------------------------------------------------------
# Helpers: fake ObjectId and user docs
# ---------------------------------------------------------------------------

_FAKE_OID = ObjectId()
_FAKE_OID_STR = str(_FAKE_OID)
_FAKE_TOKEN = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

_FAKE_USER_DOC = {
    "_id": _FAKE_OID,
    "username": "alice",
    "created_at": datetime.now(timezone.utc),
    "injury_profile": None,
    "risk_modifiers": None,
}


def _make_test_app():
    """Create a minimal FastAPI app with the auth router for testing."""
    import auth

    test_app = FastAPI()
    test_app.include_router(auth.router)
    return test_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_users():
    """Return a mock motor collection for 'users'."""
    return AsyncMock()


@pytest.fixture
def mock_sessions():
    """Return a mock motor collection for 'sessions'."""
    return AsyncMock()


@pytest.fixture
def patched_collections(mock_users, mock_sessions):
    """Patch auth.get_collection to return mock collections."""
    import auth

    def _get_collection(name):
        if name == "users":
            return mock_users
        elif name == "sessions":
            return mock_sessions
        return MagicMock()

    with patch.object(auth, "get_collection", side_effect=_get_collection):
        yield {"users": mock_users, "sessions": mock_sessions}


@pytest.fixture
def client(patched_collections):
    """Return a TestClient with mocked DB collections."""
    app = _make_test_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test: create_user
# ---------------------------------------------------------------------------

class TestCreateUser:
    """Tests for the create_user helper function."""

    @pytest.mark.asyncio
    async def test_create_user_inserts_and_returns(self, patched_collections):
        import auth

        mock_users = patched_collections["users"]
        mock_users.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id=_FAKE_OID)
        )

        result = await auth.create_user("alice")
        mock_users.insert_one.assert_awaited_once()
        doc = mock_users.insert_one.call_args[0][0]
        assert doc["username"] == "alice"
        assert result["_id"] == _FAKE_OID

    @pytest.mark.asyncio
    async def test_create_user_has_required_fields(self, patched_collections):
        import auth

        mock_users = patched_collections["users"]
        mock_users.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id=_FAKE_OID)
        )

        result = await auth.create_user("bob")
        doc = mock_users.insert_one.call_args[0][0]
        assert "username" in doc
        assert "created_at" in doc
        assert doc["injury_profile"] is None
        assert doc["risk_modifiers"] is None


# ---------------------------------------------------------------------------
# Test: get_user / get_user_by_id
# ---------------------------------------------------------------------------

class TestGetUser:
    """Tests for user lookup helpers."""

    @pytest.mark.asyncio
    async def test_get_user_found(self, patched_collections):
        import auth

        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)

        result = await auth.get_user("alice")
        assert result is not None
        assert result["username"] == "alice"
        patched_collections["users"].find_one.assert_awaited_once_with({"username": "alice"})

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, patched_collections):
        import auth

        patched_collections["users"].find_one = AsyncMock(return_value=None)

        result = await auth.get_user("ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_by_id_found(self, patched_collections):
        import auth

        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)

        result = await auth.get_user_by_id(_FAKE_OID_STR)
        assert result is not None
        assert result["username"] == "alice"
        patched_collections["users"].find_one.assert_awaited_once_with(
            {"_id": ObjectId(_FAKE_OID_STR)}
        )

    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(self, patched_collections):
        import auth

        patched_collections["users"].find_one = AsyncMock(return_value=None)

        result = await auth.get_user_by_id(_FAKE_OID_STR)
        assert result is None


# ---------------------------------------------------------------------------
# Test: session CRUD
# ---------------------------------------------------------------------------

class TestSessionCRUD:
    """Tests for create_session, validate_session, delete_session."""

    @pytest.mark.asyncio
    async def test_create_session_returns_token(self, patched_collections):
        import auth

        patched_collections["sessions"].insert_one = AsyncMock()

        token = await auth.create_session(_FAKE_OID_STR)
        assert isinstance(token, str)
        assert len(token) == 36  # UUID format
        patched_collections["sessions"].insert_one.assert_awaited_once()
        doc = patched_collections["sessions"].insert_one.call_args[0][0]
        assert doc["token"] == token
        assert doc["user_id"] == _FAKE_OID_STR

    @pytest.mark.asyncio
    async def test_validate_session_valid(self, patched_collections):
        import auth

        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )

        user_id = await auth.validate_session(_FAKE_TOKEN)
        assert user_id == _FAKE_OID_STR
        patched_collections["sessions"].find_one.assert_awaited_once_with(
            {"token": _FAKE_TOKEN}
        )

    @pytest.mark.asyncio
    async def test_validate_session_invalid(self, patched_collections):
        import auth

        patched_collections["sessions"].find_one = AsyncMock(return_value=None)

        user_id = await auth.validate_session("bad-token")
        assert user_id is None

    @pytest.mark.asyncio
    async def test_delete_session(self, patched_collections):
        import auth

        patched_collections["sessions"].delete_one = AsyncMock()

        await auth.delete_session(_FAKE_TOKEN)
        patched_collections["sessions"].delete_one.assert_awaited_once_with(
            {"token": _FAKE_TOKEN}
        )


# ---------------------------------------------------------------------------
# Test: get_current_user dependency
# ---------------------------------------------------------------------------

class TestGetCurrentUser:
    """Tests for the FastAPI dependency that extracts user from Authorization header."""

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self, patched_collections):
        import auth

        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)

        user = await auth.get_current_user(authorization=f"Bearer {_FAKE_TOKEN}")
        assert user["username"] == "alice"

    @pytest.mark.asyncio
    async def test_missing_header_raises_401(self, patched_collections):
        import auth
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await auth.get_current_user(authorization=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_header_raises_401(self, patched_collections):
        import auth
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await auth.get_current_user(authorization="Token abc")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self, patched_collections):
        import auth
        from fastapi import HTTPException

        patched_collections["sessions"].find_one = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await auth.get_current_user(authorization="Bearer bad-token")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_user_not_found_raises_401(self, patched_collections):
        import auth
        from fastapi import HTTPException

        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await auth.get_current_user(authorization=f"Bearer {_FAKE_TOKEN}")
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Test: POST /api/auth/register
# ---------------------------------------------------------------------------

class TestRegisterEndpoint:
    """Tests for the registration endpoint."""

    def test_register_success(self, client, patched_collections):
        import auth

        patched_collections["users"].find_one = AsyncMock(return_value=None)
        patched_collections["users"].insert_one = AsyncMock(
            return_value=MagicMock(inserted_id=_FAKE_OID)
        )
        patched_collections["sessions"].insert_one = AsyncMock()

        resp = client.post("/api/auth/register", json={"username": "alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == _FAKE_OID_STR
        assert data["username"] == "alice"
        assert "token" in data

    def test_register_duplicate_username(self, client, patched_collections):
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)

        resp = client.post("/api/auth/register", json={"username": "alice"})
        assert resp.status_code == 409

    def test_register_empty_username(self, client, patched_collections):
        resp = client.post("/api/auth/register", json={"username": ""})
        assert resp.status_code == 400

    def test_register_whitespace_username(self, client, patched_collections):
        resp = client.post("/api/auth/register", json={"username": "   "})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test: POST /api/auth/login
# ---------------------------------------------------------------------------

class TestLoginEndpoint:
    """Tests for the login endpoint."""

    def test_login_success(self, client, patched_collections):
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)
        patched_collections["sessions"].insert_one = AsyncMock()

        resp = client.post("/api/auth/login", json={"username": "alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == _FAKE_OID_STR
        assert data["username"] == "alice"
        assert "token" in data

    def test_login_nonexistent_user(self, client, patched_collections):
        patched_collections["users"].find_one = AsyncMock(return_value=None)

        resp = client.post("/api/auth/login", json={"username": "ghost"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test: GET /api/auth/me
# ---------------------------------------------------------------------------

class TestMeEndpoint:
    """Tests for the /me endpoint."""

    def test_me_authenticated(self, client, patched_collections):
        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)

        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {_FAKE_TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "alice"
        assert data["user_id"] == _FAKE_OID_STR

    def test_me_unauthenticated(self, client, patched_collections):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: POST /api/auth/logout
# ---------------------------------------------------------------------------

class TestLogoutEndpoint:
    """Tests for the logout endpoint."""

    def test_logout_success(self, client, patched_collections):
        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)
        patched_collections["sessions"].delete_one = AsyncMock()

        resp = client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {_FAKE_TOKEN}"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        patched_collections["sessions"].delete_one.assert_awaited_once()

    def test_logout_unauthenticated(self, client, patched_collections):
        resp = client.post("/api/auth/logout")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: GET /api/auth/profile
# ---------------------------------------------------------------------------

class TestGetProfileEndpoint:
    """Tests for the profile GET endpoint."""

    def test_get_profile_empty(self, client, patched_collections):
        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)

        resp = client.get(
            "/api/auth/profile",
            headers={"Authorization": f"Bearer {_FAKE_TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["injury_profile"] is None
        assert data["risk_modifiers"] is None

    def test_get_profile_with_data(self, client, patched_collections):
        user_with_profile = {
            **_FAKE_USER_DOC,
            "injury_profile": {"knee": "acl_tear"},
            "risk_modifiers": {"age": 25, "sport": "basketball"},
        }
        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=user_with_profile)

        resp = client.get(
            "/api/auth/profile",
            headers={"Authorization": f"Bearer {_FAKE_TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["injury_profile"] == {"knee": "acl_tear"}
        assert data["risk_modifiers"] == {"age": 25, "sport": "basketball"}

    def test_get_profile_unauthenticated(self, client, patched_collections):
        resp = client.get("/api/auth/profile")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: PUT /api/auth/profile
# ---------------------------------------------------------------------------

class TestUpdateProfileEndpoint:
    """Tests for the profile PUT endpoint."""

    def test_update_injury_profile(self, client, patched_collections):
        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)
        patched_collections["users"].update_one = AsyncMock()

        resp = client.put(
            "/api/auth/profile",
            json={"injury_profile": {"knee": "acl_tear"}},
            headers={"Authorization": f"Bearer {_FAKE_TOKEN}"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        patched_collections["users"].update_one.assert_awaited_once()
        call_args = patched_collections["users"].update_one.call_args
        assert call_args[0][0] == {"_id": _FAKE_OID}
        update = call_args[0][1]
        assert update["$set"]["injury_profile"] == {"knee": "acl_tear"}

    def test_update_risk_modifiers(self, client, patched_collections):
        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)
        patched_collections["users"].update_one = AsyncMock()

        resp = client.put(
            "/api/auth/profile",
            json={"risk_modifiers": {"age": 30}},
            headers={"Authorization": f"Bearer {_FAKE_TOKEN}"},
        )
        assert resp.status_code == 200
        call_args = patched_collections["users"].update_one.call_args
        update = call_args[0][1]
        assert update["$set"]["risk_modifiers"] == {"age": 30}

    def test_update_both_fields(self, client, patched_collections):
        patched_collections["sessions"].find_one = AsyncMock(
            return_value={"token": _FAKE_TOKEN, "user_id": _FAKE_OID_STR}
        )
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)
        patched_collections["users"].update_one = AsyncMock()

        resp = client.put(
            "/api/auth/profile",
            json={
                "injury_profile": {"shoulder": "rotator_cuff"},
                "risk_modifiers": {"sport": "swimming"},
            },
            headers={"Authorization": f"Bearer {_FAKE_TOKEN}"},
        )
        assert resp.status_code == 200
        call_args = patched_collections["users"].update_one.call_args
        update = call_args[0][1]
        assert update["$set"]["injury_profile"] == {"shoulder": "rotator_cuff"}
        assert update["$set"]["risk_modifiers"] == {"sport": "swimming"}

    def test_update_profile_unauthenticated(self, client, patched_collections):
        resp = client.put("/api/auth/profile", json={"injury_profile": {}})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for auth flows."""

    def test_multiple_logins_same_user(self, client, patched_collections):
        """Multiple logins for the same user should return different tokens."""
        patched_collections["users"].find_one = AsyncMock(return_value=_FAKE_USER_DOC)
        patched_collections["sessions"].insert_one = AsyncMock()

        resp1 = client.post("/api/auth/login", json={"username": "alice"})
        resp2 = client.post("/api/auth/login", json={"username": "alice"})

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        token1 = resp1.json()["token"]
        token2 = resp2.json()["token"]
        # Tokens should be different UUIDs
        assert token1 != token2

    def test_register_then_login_flow(self, client, patched_collections):
        """Registration followed by login should both succeed."""
        # Register
        patched_collections["users"].find_one = AsyncMock(return_value=None)
        patched_collections["users"].insert_one = AsyncMock(
            return_value=MagicMock(inserted_id=_FAKE_OID)
        )
        patched_collections["sessions"].insert_one = AsyncMock()

        reg_resp = client.post("/api/auth/register", json={"username": "bob"})
        assert reg_resp.status_code == 200

        # Login (user now exists)
        patched_collections["users"].find_one = AsyncMock(return_value={
            **_FAKE_USER_DOC,
            "username": "bob",
        })

        login_resp = client.post("/api/auth/login", json={"username": "bob"})
        assert login_resp.status_code == 200
        assert login_resp.json()["username"] == "bob"
