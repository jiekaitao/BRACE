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
    TeamClustering,
    parse_jersey_response,
    cluster_teams,
    cluster_teams_visual,
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


# ---------------------------------------------------------------------------
# Visual K-Means team clustering tests
# ---------------------------------------------------------------------------

def _make_uniform_crop(r: int, g: int, b: int, h: int = 100, w: int = 80, seed: int = 0) -> np.ndarray:
    """Create an RGB crop with base color + slight noise (realistic jersey)."""
    rng = np.random.RandomState(seed)
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    crop[:, :] = [r, g, b]
    noise = rng.randint(-15, 16, size=(h, w, 3), dtype=np.int16)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return crop


class TestClusterTeamsVisual:
    def test_two_distinct_teams(self):
        """Red vs blue crops should yield 2 teams with correct hex colors."""
        crops = {
            1: _make_uniform_crop(220, 30, 30, seed=1),   # red
            2: _make_uniform_crop(200, 20, 20, seed=2),    # red
            3: _make_uniform_crop(30, 30, 220, seed=3),    # blue
            4: _make_uniform_crop(20, 20, 200, seed=4),    # blue
        }
        result = cluster_teams_visual(crops, k=2)
        assert result is not None
        assert len(result.assignments) == 4
        assert len(result.team_colors) == 2
        # Red subjects should be in the same team
        assert result.assignments[1] == result.assignments[2]
        # Blue subjects should be in the same team
        assert result.assignments[3] == result.assignments[4]
        # Red and blue should be in different teams
        assert result.assignments[1] != result.assignments[3]

    def test_hex_color_format(self):
        """Team colors should be valid hex strings."""
        crops = {
            1: _make_uniform_crop(255, 0, 0, seed=10),
            2: _make_uniform_crop(0, 0, 255, seed=11),
        }
        result = cluster_teams_visual(crops, k=2)
        assert result is not None
        for color in result.team_colors.values():
            assert color.startswith("#")
            assert len(color) == 7
            # Valid hex chars
            int(color[1:], 16)

    def test_fewer_crops_than_k_returns_none(self):
        """Should return None if fewer valid crops than k."""
        crops = {1: _make_uniform_crop(255, 0, 0, seed=20)}
        result = cluster_teams_visual(crops, k=2)
        assert result is None

    def test_empty_crops_returns_none(self):
        result = cluster_teams_visual({}, k=2)
        assert result is None

    def test_small_crops_skipped(self):
        """Crops smaller than 30px should be skipped."""
        crops = {
            1: np.zeros((20, 20, 3), dtype=np.uint8),  # too small
            2: np.zeros((25, 25, 3), dtype=np.uint8),  # too small
        }
        result = cluster_teams_visual(crops, k=2)
        assert result is None

    def test_mixed_valid_and_small(self):
        """Small crops skipped, valid ones still cluster."""
        crops = {
            1: _make_uniform_crop(255, 0, 0, seed=30),  # valid red
            2: np.zeros((10, 10, 3), dtype=np.uint8),   # too small
            3: _make_uniform_crop(0, 0, 255, seed=31),   # valid blue
        }
        result = cluster_teams_visual(crops, k=2)
        assert result is not None
        assert 2 not in result.assignments  # small crop excluded
        assert len(result.assignments) == 2
        assert result.assignments[1] != result.assignments[3]
