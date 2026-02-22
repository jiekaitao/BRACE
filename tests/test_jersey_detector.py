"""Tests for jersey detection via Gemini Vision API."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from jersey_detector import (
    JerseyDetector,
    JerseyInfo,
    parse_jersey_response,
    cluster_teams,
    _encode_crop_jpeg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_crop(h: int = 200, w: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# parse_jersey_response tests
# ---------------------------------------------------------------------------

class TestParseJerseyResponse:
    def test_valid_json(self):
        text = '{"number": 23, "color": "red"}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number == 23
        assert info.color == "red"

    def test_number_as_string(self):
        text = '{"number": "7", "color": "blue"}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number == 7
        assert info.color == "blue"

    def test_missing_number(self):
        text = '{"color": "white"}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number is None
        assert info.color == "white"

    def test_missing_color(self):
        text = '{"number": 10}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number == 10
        assert info.color is None

    def test_null_values(self):
        text = '{"number": null, "color": null}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number is None
        assert info.color is None

    def test_empty_json(self):
        text = '{}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number is None
        assert info.color is None

    def test_invalid_json(self):
        text = 'not json at all'
        info = parse_jersey_response(text)
        assert info is None

    def test_number_zero(self):
        text = '{"number": 0, "color": "green"}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number == 0

    def test_extra_fields_ignored(self):
        text = '{"number": 5, "color": "yellow", "team": "Lakers"}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number == 5
        assert info.color == "yellow"

    def test_color_normalized_lowercase(self):
        text = '{"number": 12, "color": "RED"}'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.color == "red"

    def test_markdown_wrapped_json(self):
        """Gemini sometimes wraps JSON in markdown code fences."""
        text = '```json\n{"number": 23, "color": "white"}\n```'
        info = parse_jersey_response(text)
        assert info is not None
        assert info.number == 23
        assert info.color == "white"


# ---------------------------------------------------------------------------
# _encode_crop_jpeg tests
# ---------------------------------------------------------------------------

class TestEncodeCropJpeg:
    def test_produces_jpeg_bytes(self):
        crop = _make_crop()
        data = _encode_crop_jpeg(crop)
        assert data[:2] == b"\xff\xd8"

    def test_respects_max_dim(self):
        crop = _make_crop(h=1000, w=500)
        data = _encode_crop_jpeg(crop, max_dim=128)
        assert len(data) < 500 * 1000 * 3


# ---------------------------------------------------------------------------
# cluster_teams tests
# ---------------------------------------------------------------------------

class TestClusterTeams:
    def test_two_distinct_colors(self):
        jerseys = {
            1: JerseyInfo(number=23, color="red"),
            2: JerseyInfo(number=7, color="red"),
            3: JerseyInfo(number=30, color="blue"),
            4: JerseyInfo(number=11, color="blue"),
        }
        teams = cluster_teams(jerseys)
        assert len(teams) == 2
        # Each team should have 2 players
        team_sizes = sorted([len(t) for t in teams.values()])
        assert team_sizes == [2, 2]

    def test_single_color(self):
        jerseys = {
            1: JerseyInfo(number=1, color="white"),
            2: JerseyInfo(number=2, color="white"),
        }
        teams = cluster_teams(jerseys)
        assert len(teams) == 1

    def test_empty_input(self):
        teams = cluster_teams({})
        assert len(teams) == 0

    def test_none_colors_excluded(self):
        jerseys = {
            1: JerseyInfo(number=10, color="red"),
            2: JerseyInfo(number=None, color=None),
            3: JerseyInfo(number=5, color="red"),
        }
        teams = cluster_teams(jerseys)
        # Subject 2 has no color, should not appear in any team
        all_subjects = set()
        for members in teams.values():
            all_subjects.update(members)
        assert 2 not in all_subjects

    def test_similar_colors_grouped(self):
        """Slightly different color strings that are the same base color."""
        jerseys = {
            1: JerseyInfo(number=1, color="dark red"),
            2: JerseyInfo(number=2, color="red"),
            3: JerseyInfo(number=3, color="blue"),
        }
        teams = cluster_teams(jerseys)
        # "dark red" and "red" both contain "red" — implementation groups by base color
        assert len(teams) >= 2


# ---------------------------------------------------------------------------
# JerseyDetector tests
# ---------------------------------------------------------------------------

class TestJerseyDetector:
    @patch("jersey_detector.JerseyDetector._ensure_model")
    def test_detect_jersey_mocked(self, mock_ensure):
        """Mock Gemini to return known jersey info."""
        detector = JerseyDetector(api_key="test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"number": 23, "color": "red"}'
        mock_client.models.generate_content.return_value = mock_response
        detector._model = mock_client
        detector._init_error = None

        crop = _make_crop()
        result = detector.detect_jersey(crop)
        assert result is not None
        assert result.number == 23
        assert result.color == "red"

    def test_detect_jersey_no_model(self):
        """Returns None when model is unavailable."""
        detector = JerseyDetector(api_key="test-key")
        detector._init_error = "no model"
        detector._model = None
        result = detector.detect_jersey(_make_crop())
        assert result is None

    @patch("jersey_detector.JerseyDetector._ensure_model")
    def test_detect_jersey_api_exception(self, mock_ensure):
        """API exception returns None."""
        detector = JerseyDetector(api_key="test-key")
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota")
        detector._model = mock_client
        detector._init_error = None

        result = detector.detect_jersey(_make_crop())
        assert result is None

    @patch("jersey_detector.JerseyDetector._ensure_model")
    def test_detect_batch(self, mock_ensure):
        """Batch detection processes multiple crops."""
        detector = JerseyDetector(api_key="test-key")
        mock_client = MagicMock()

        responses = [
            MagicMock(text='{"number": 23, "color": "red"}'),
            MagicMock(text='{"number": 7, "color": "blue"}'),
        ]
        mock_client.models.generate_content.side_effect = responses
        detector._model = mock_client
        detector._init_error = None

        crops = {1: _make_crop(seed=1), 2: _make_crop(seed=2)}
        results = detector.detect_batch(crops)
        assert len(results) == 2
        assert results[1].number == 23
        assert results[2].number == 7

    @patch("jersey_detector.JerseyDetector._ensure_model")
    def test_caching(self, mock_ensure):
        """Second call with same subject_id returns cached result."""
        detector = JerseyDetector(api_key="test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"number": 10, "color": "white"}'
        mock_client.models.generate_content.return_value = mock_response
        detector._model = mock_client
        detector._init_error = None

        crop = _make_crop()
        r1 = detector.detect_jersey(crop, subject_id=5)
        assert r1 is not None
        assert r1.number == 10

        # Second call should use cache
        r2 = detector.detect_jersey(crop, subject_id=5)
        assert r2 is not None
        assert r2.number == 10
        # API should only have been called once
        assert mock_client.models.generate_content.call_count == 1

    def test_available_false_without_sdk(self):
        detector = JerseyDetector(api_key="test-key")
        detector._init_error = "no sdk"
        detector._model = None
        assert not detector.available
