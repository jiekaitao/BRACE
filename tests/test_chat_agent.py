"""Tests for Gemini-powered injury intake chat agent."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from chat_agent import (
    InjuryChatAgent,
    InjuryModifiers,
    profile_to_risk_modifiers,
    INTAKE_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gemini_response(text: str) -> MagicMock:
    """Create a mock Gemini API response."""
    resp = MagicMock()
    resp.text = text
    return resp


def _profile(injuries: list[dict], complete: bool = True) -> dict:
    """Build a profile dict."""
    return {"injuries": injuries, "complete": complete}


def _injury(type: str, side: str = "unknown", severity: str = "mild",
            timeframe: str = "current") -> dict:
    return {"type": type, "side": side, "severity": severity,
            "timeframe": timeframe}


# ---------------------------------------------------------------------------
# InjuryChatAgent.chat — profile extraction
# ---------------------------------------------------------------------------

class TestProfileExtraction:
    """Test that Gemini responses with JSON blocks are correctly parsed."""

    @pytest.mark.asyncio
    async def test_extract_acl_injury(self):
        agent = InjuryChatAgent()
        response_text = (
            "I understand you tore your ACL. Let me summarize:\n"
            '```json\n'
            '{"injuries": [{"type": "acl", "side": "unknown", '
            '"severity": "severe", "timeframe": "chronic"}], "complete": true}\n'
            '```'
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "I tore my ACL 2 years ago"}
            ])

        assert result["profile_complete"] is True
        profile = result["extracted_profile"]
        assert profile is not None
        assert len(profile["injuries"]) == 1
        inj = profile["injuries"][0]
        assert inj["type"] == "acl"
        assert inj["side"] == "unknown"
        assert inj["severity"] == "severe"
        assert inj["timeframe"] == "chronic"

    @pytest.mark.asyncio
    async def test_extract_shoulder_injury(self):
        agent = InjuryChatAgent()
        response_text = (
            "Got it — right shoulder pain.\n"
            '```json\n'
            '{"injuries": [{"type": "shoulder", "side": "right", '
            '"severity": "mild", "timeframe": "current"}], "complete": true}\n'
            '```'
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "my right shoulder hurts sometimes"}
            ])

        profile = result["extracted_profile"]
        assert profile is not None
        inj = profile["injuries"][0]
        assert inj["type"] == "shoulder"
        assert inj["side"] == "right"
        assert inj["severity"] == "mild"
        assert inj["timeframe"] == "current"

    @pytest.mark.asyncio
    async def test_extract_ankle_injury(self):
        agent = InjuryChatAgent()
        response_text = (
            "That sounds recent. Here's what I got:\n"
            '```json\n'
            '{"injuries": [{"type": "ankle", "side": "left", '
            '"severity": "moderate", "timeframe": "acute"}], "complete": true}\n'
            '```'
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "I sprained my left ankle last week"}
            ])

        profile = result["extracted_profile"]
        assert profile is not None
        inj = profile["injuries"][0]
        assert inj["type"] == "ankle"
        assert inj["side"] == "left"
        assert inj["severity"] == "moderate"
        assert inj["timeframe"] == "acute"

    @pytest.mark.asyncio
    async def test_extract_lower_back_injury(self):
        agent = InjuryChatAgent()
        response_text = (
            "Chronic lower back pain — noted.\n"
            '```json\n'
            '{"injuries": [{"type": "lower_back", "side": "bilateral", '
            '"severity": "moderate", "timeframe": "chronic"}], "complete": true}\n'
            '```'
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "chronic lower back pain"}
            ])

        profile = result["extracted_profile"]
        assert profile is not None
        inj = profile["injuries"][0]
        assert inj["type"] == "lower_back"
        assert inj["side"] == "bilateral"
        assert inj["severity"] == "moderate"
        assert inj["timeframe"] == "chronic"

    @pytest.mark.asyncio
    async def test_extract_multiple_injuries(self):
        agent = InjuryChatAgent()
        response_text = (
            "Two injuries noted:\n"
            '```json\n'
            '{"injuries": ['
            '{"type": "acl", "side": "left", "severity": "severe", "timeframe": "chronic"}, '
            '{"type": "shoulder", "side": "right", "severity": "moderate", "timeframe": "current"}'
            '], "complete": true}\n'
            '```'
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "ACL tear and shoulder impingement"}
            ])

        profile = result["extracted_profile"]
        assert profile is not None
        assert len(profile["injuries"]) == 2
        types = {inj["type"] for inj in profile["injuries"]}
        assert types == {"acl", "shoulder"}

    @pytest.mark.asyncio
    async def test_no_injuries_healthy(self):
        agent = InjuryChatAgent()
        response_text = (
            "Great to hear you're healthy!\n"
            '```json\n'
            '{"injuries": [], "complete": true}\n'
            '```'
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "I'm perfectly healthy"}
            ])

        assert result["profile_complete"] is True
        profile = result["extracted_profile"]
        assert profile is not None
        assert len(profile["injuries"]) == 0

    @pytest.mark.asyncio
    async def test_no_json_in_response(self):
        """When Gemini asks a follow-up question, no profile is extracted."""
        agent = InjuryChatAgent()
        response_text = "Can you tell me which side was affected?"
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "I hurt my knee"}
            ])

        assert result["extracted_profile"] is None
        assert result["profile_complete"] is False
        assert "which side" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_malformed_json_ignored(self):
        agent = InjuryChatAgent()
        response_text = (
            "Let me check...\n```json\n{not valid json}\n```"
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "test"}
            ])

        assert result["extracted_profile"] is None
        assert result["profile_complete"] is False

    @pytest.mark.asyncio
    async def test_json_without_injuries_key_ignored(self):
        agent = InjuryChatAgent()
        response_text = (
            '```json\n{"something_else": true}\n```'
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            result = await agent.chat([
                {"role": "user", "content": "test"}
            ])

        assert result["extracted_profile"] is None


# ---------------------------------------------------------------------------
# InjuryChatAgent.chat — message passing
# ---------------------------------------------------------------------------

class TestChatMessagePassing:
    """Test that messages are correctly forwarded to Gemini."""

    @pytest.mark.asyncio
    async def test_messages_forwarded_to_gemini(self):
        agent = InjuryChatAgent()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "My knee hurts"},
        ]
        with patch("chat_agent._get_genai_client") as mock_client:
            gen = mock_client.return_value.models.generate_content
            gen.return_value = _make_gemini_response("Tell me more")
            await agent.chat(messages)

        call_kwargs = gen.call_args
        contents = call_kwargs.kwargs.get("contents") or call_kwargs[1].get("contents")
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"  # assistant -> model
        assert contents[2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_system_prompt_sent(self):
        agent = InjuryChatAgent()
        with patch("chat_agent._get_genai_client") as mock_client:
            gen = mock_client.return_value.models.generate_content
            gen.return_value = _make_gemini_response("Hi!")
            await agent.chat([{"role": "user", "content": "hello"}])

        call_kwargs = gen.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert "system_instruction" in config
        assert "BRACE" in config["system_instruction"]

    @pytest.mark.asyncio
    async def test_model_name_is_gemini_flash(self):
        agent = InjuryChatAgent()
        with patch("chat_agent._get_genai_client") as mock_client:
            gen = mock_client.return_value.models.generate_content
            gen.return_value = _make_gemini_response("Hi!")
            await agent.chat([{"role": "user", "content": "hello"}])

        call_kwargs = gen.call_args
        model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        assert model == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# profile_to_risk_modifiers
# ---------------------------------------------------------------------------

class TestRiskModifiers:
    """Test injury profile to risk modifier mapping."""

    def test_no_profile_returns_defaults(self):
        mods = profile_to_risk_modifiers(None)
        assert mods["fppa_scale"] == 1.0
        assert mods["hip_drop_scale"] == 1.0
        assert mods["trunk_lean_scale"] == 1.0
        assert mods["asymmetry_scale"] == 1.0
        assert mods["angular_velocity_scale"] == 1.0
        assert mods["monitor_joints"] == []

    def test_empty_injuries_returns_defaults(self):
        mods = profile_to_risk_modifiers({"injuries": []})
        assert mods["fppa_scale"] == 1.0
        assert mods["monitor_joints"] == []

    def test_acl_lowers_fppa_scale(self):
        profile = _profile([_injury("acl", severity="severe")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["fppa_scale"] == 0.65
        assert "left_knee" in mods["monitor_joints"]
        assert "right_knee" in mods["monitor_joints"]

    def test_acl_moderate_severity(self):
        profile = _profile([_injury("acl", severity="moderate")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["fppa_scale"] == 0.8

    def test_acl_mild_severity(self):
        profile = _profile([_injury("acl", severity="mild")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["fppa_scale"] == 0.9

    def test_shoulder_lowers_angular_velocity(self):
        profile = _profile([_injury("shoulder", side="right", severity="severe")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["angular_velocity_scale"] == 0.65
        assert "right_elbow" in mods["monitor_joints"]
        # Left should NOT be in monitor_joints for right-side injury
        assert "left_elbow" not in mods["monitor_joints"]

    def test_shoulder_bilateral_monitors_both(self):
        profile = _profile([_injury("shoulder", side="bilateral", severity="mild")])
        mods = profile_to_risk_modifiers(profile)
        assert "left_elbow" in mods["monitor_joints"]
        assert "right_elbow" in mods["monitor_joints"]

    def test_shoulder_left_monitors_left(self):
        profile = _profile([_injury("shoulder", side="left", severity="mild")])
        mods = profile_to_risk_modifiers(profile)
        assert "left_elbow" in mods["monitor_joints"]
        assert "right_elbow" not in mods["monitor_joints"]

    def test_lower_back_lowers_trunk_lean_and_hip_drop(self):
        profile = _profile([_injury("lower_back", severity="severe")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["trunk_lean_scale"] == 0.65
        assert mods["hip_drop_scale"] == 0.65

    def test_ankle_adds_monitoring(self):
        profile = _profile([_injury("ankle", severity="moderate")])
        mods = profile_to_risk_modifiers(profile)
        assert "left_ankle" in mods["monitor_joints"]
        assert "right_ankle" in mods["monitor_joints"]
        # Ankle should not change any scale factors
        assert mods["fppa_scale"] == 1.0
        assert mods["trunk_lean_scale"] == 1.0

    def test_hip_lowers_hip_drop(self):
        profile = _profile([_injury("hip", severity="moderate")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["hip_drop_scale"] == 0.8

    def test_hamstring_lowers_asymmetry(self):
        profile = _profile([_injury("hamstring", severity="severe")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["asymmetry_scale"] == 0.65

    def test_knee_general_lowers_fppa(self):
        profile = _profile([_injury("knee_general", severity="moderate")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["fppa_scale"] == 0.8

    def test_multiple_injuries_stack_multiplicatively(self):
        """ACL severe (0.65) + knee_general moderate (0.8) -> 0.65 * 0.8 = 0.52"""
        profile = _profile([
            _injury("acl", severity="severe"),
            _injury("knee_general", severity="moderate"),
        ])
        mods = profile_to_risk_modifiers(profile)
        assert abs(mods["fppa_scale"] - 0.65 * 0.8) < 1e-9

    def test_multiple_different_injuries(self):
        """ACL + lower back should affect different scales independently."""
        profile = _profile([
            _injury("acl", severity="severe"),
            _injury("lower_back", severity="moderate"),
        ])
        mods = profile_to_risk_modifiers(profile)
        assert mods["fppa_scale"] == 0.65  # ACL severe
        assert mods["trunk_lean_scale"] == 0.8  # lower back moderate
        assert mods["hip_drop_scale"] == 0.8  # lower back moderate

    def test_no_injuries_key_returns_defaults(self):
        mods = profile_to_risk_modifiers({"something": "else"})
        assert mods["fppa_scale"] == 1.0

    def test_unknown_severity_defaults_to_mild(self):
        profile = _profile([_injury("acl", severity="unknown_level")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["fppa_scale"] == 0.9  # mild default

    def test_unknown_injury_type_no_change(self):
        profile = _profile([_injury("elbow_tendinitis", severity="severe")])
        mods = profile_to_risk_modifiers(profile)
        assert mods["fppa_scale"] == 1.0
        assert mods["hip_drop_scale"] == 1.0
        assert mods["trunk_lean_scale"] == 1.0


# ---------------------------------------------------------------------------
# InjuryModifiers dataclass
# ---------------------------------------------------------------------------

class TestInjuryModifiersDataclass:
    def test_defaults(self):
        mods = InjuryModifiers()
        assert mods.fppa_scale == 1.0
        assert mods.hip_drop_scale == 1.0
        assert mods.trunk_lean_scale == 1.0
        assert mods.asymmetry_scale == 1.0
        assert mods.angular_velocity_scale == 1.0
        assert mods.monitor_joints == []

    def test_custom_values(self):
        mods = InjuryModifiers(fppa_scale=0.5, monitor_joints=["left_knee"])
        assert mods.fppa_scale == 0.5
        assert mods.monitor_joints == ["left_knee"]


# ---------------------------------------------------------------------------
# FastAPI endpoint tests (using TestClient)
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    """Test the POST /api/chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        from chat_agent import router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        test_app = FastAPI()
        test_app.include_router(router)

        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response("Tell me about your injury history.")
            )
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as ac:
                resp = await ac.post("/api/chat", json={
                    "messages": [{"role": "user", "content": "Hi there"}]
                })

        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["profile_complete"] is False

    @pytest.mark.asyncio
    async def test_chat_empty_messages_422(self):
        from chat_agent import router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        test_app = FastAPI()
        test_app.include_router(router)

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/api/chat", json={"messages": []})

        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_with_extracted_profile(self):
        from chat_agent import router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        test_app = FastAPI()
        test_app.include_router(router)

        response_text = (
            "Summary:\n"
            '```json\n'
            '{"injuries": [{"type": "acl", "side": "left", '
            '"severity": "severe", "timeframe": "chronic"}], "complete": true}\n'
            '```'
        )
        with patch("chat_agent._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = (
                _make_gemini_response(response_text)
            )
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as ac:
                resp = await ac.post("/api/chat", json={
                    "messages": [{"role": "user", "content": "ACL tear left knee"}]
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["profile_complete"] is True
        assert data["extracted_profile"] is not None
        assert len(data["extracted_profile"]["injuries"]) == 1


class TestConfirmProfileEndpoint:
    """Test the POST /api/chat/confirm-profile endpoint."""

    @pytest.mark.asyncio
    async def test_confirm_profile_anonymous(self):
        from chat_agent import router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        test_app = FastAPI()
        test_app.include_router(router)

        profile = _profile([_injury("acl", severity="severe")])

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/api/chat/confirm-profile", json={
                "injury_profile": profile,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "risk_modifiers" in data
        assert data["risk_modifiers"]["fppa_scale"] == 0.65

    @pytest.mark.asyncio
    async def test_confirm_profile_with_user_id(self):
        from chat_agent import router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        test_app = FastAPI()
        test_app.include_router(router)

        profile = _profile([_injury("lower_back", severity="moderate")])

        with patch("chat_agent.get_collection") as mock_coll:
            mock_update = AsyncMock()
            mock_coll.return_value.update_one = mock_update

            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as ac:
                resp = await ac.post("/api/chat/confirm-profile", json={
                    "user_id": "507f1f77bcf86cd799439011",
                    "injury_profile": profile,
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        # Verify MongoDB was called
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        update_doc = call_args[0][1]["$set"]
        assert "injury_profile" in update_doc
        assert "risk_modifiers" in update_doc
        assert update_doc["risk_modifiers"]["trunk_lean_scale"] == 0.8

    @pytest.mark.asyncio
    async def test_confirm_profile_returns_modifiers(self):
        from chat_agent import router
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        test_app = FastAPI()
        test_app.include_router(router)

        profile = _profile([
            _injury("acl", severity="severe"),
            _injury("shoulder", side="left", severity="moderate"),
        ])

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/api/chat/confirm-profile", json={
                "injury_profile": profile,
            })

        data = resp.json()
        mods = data["risk_modifiers"]
        assert mods["fppa_scale"] == 0.65
        assert mods["angular_velocity_scale"] == 0.8
        assert "left_elbow" in mods["monitor_joints"]


# ---------------------------------------------------------------------------
# System prompt content
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_mentions_brace(self):
        assert "BRACE" in INTAKE_SYSTEM_PROMPT

    def test_system_prompt_lists_injury_types(self):
        for injury_type in ["acl", "shoulder", "ankle", "lower_back", "hip", "hamstring"]:
            assert injury_type in INTAKE_SYSTEM_PROMPT

    def test_system_prompt_lists_severities(self):
        for sev in ["mild", "moderate", "severe"]:
            assert sev in INTAKE_SYSTEM_PROMPT

    def test_system_prompt_lists_timeframes(self):
        for tf in ["acute", "chronic", "recovered"]:
            assert tf in INTAKE_SYSTEM_PROMPT
