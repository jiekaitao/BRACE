"""Tests for backend/jersey_detector.py — jersey number and team color detection."""

import numpy as np
import pytest

from backend.jersey_detector import (
    parse_jersey_response,
    cluster_teams,
    JerseyDetector,
    _encode_crop_jpeg,
)


# ---------------------------------------------------------------------------
# parse_jersey_response
# ---------------------------------------------------------------------------

class TestParseJerseyResponse:
    def test_valid_json(self):
        text = '{"jersey_number": 23, "jersey_color_name": "red", "dominant_color": [255, 0, 0]}'
        result = parse_jersey_response(text)
        assert result["jersey_number"] == 23
        assert result["jersey_color_name"] == "red"
        assert result["dominant_color"] == [255, 0, 0]

    def test_null_number(self):
        text = '{"jersey_number": null, "jersey_color_name": "blue", "dominant_color": [0, 0, 255]}'
        result = parse_jersey_response(text)
        assert result["jersey_number"] is None
        assert result["jersey_color_name"] == "blue"

    def test_markdown_code_block(self):
        text = '```json\n{"jersey_number": 7, "jersey_color_name": "white", "dominant_color": [255, 255, 255]}\n```'
        result = parse_jersey_response(text)
        assert result["jersey_number"] == 7
        assert result["jersey_color_name"] == "white"

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"jersey_number": 11, "jersey_color_name": "green", "dominant_color": [0, 128, 0]} as requested.'
        result = parse_jersey_response(text)
        assert result["jersey_number"] == 11
        assert result["jersey_color_name"] == "green"

    def test_fallback_regex_extraction(self):
        text = "The player's jersey number: 5, color: blue"
        result = parse_jersey_response(text)
        assert result["jersey_number"] == 5
        assert result["jersey_color_name"] == "blue"

    def test_completely_unparseable(self):
        text = "I cannot determine any information from this image."
        result = parse_jersey_response(text)
        assert result["jersey_number"] is None
        assert result["jersey_color_name"] == "unknown"
        assert result["dominant_color"] == [128, 128, 128]

    def test_invalid_number_clamped(self):
        text = '{"jersey_number": 100, "jersey_color_name": "red", "dominant_color": [255, 0, 0]}'
        result = parse_jersey_response(text)
        # 100 is > 99, so should be set to None
        assert result["jersey_number"] is None

    def test_rgb_clamped_to_0_255(self):
        text = '{"jersey_number": 1, "jersey_color_name": "red", "dominant_color": [300, -10, 128]}'
        result = parse_jersey_response(text)
        assert result["dominant_color"] == [255, 0, 128]

    def test_missing_dominant_color(self):
        text = '{"jersey_number": 3, "jersey_color_name": "purple"}'
        result = parse_jersey_response(text)
        assert result["dominant_color"] == [128, 128, 128]


# ---------------------------------------------------------------------------
# _encode_crop_jpeg
# ---------------------------------------------------------------------------

class TestEncodeCropJpeg:
    def test_returns_bytes(self):
        crop = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = _encode_crop_jpeg(crop)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_downscales_large_image(self):
        crop = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
        result = _encode_crop_jpeg(crop, max_dim=256)
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# cluster_teams
# ---------------------------------------------------------------------------

class TestClusterTeams:
    def test_two_teams_distinct_colors(self):
        jersey_results = {
            1: {"dominant_color": [255, 0, 0]},    # red team
            2: {"dominant_color": [250, 10, 5]},    # red team
            3: {"dominant_color": [0, 0, 255]},     # blue team
            4: {"dominant_color": [5, 10, 250]},    # blue team
        }
        assignments = cluster_teams(jersey_results)
        assert len(assignments) == 4
        # Players 1 and 2 should be same team, players 3 and 4 same team
        assert assignments[1] == assignments[2]
        assert assignments[3] == assignments[4]
        assert assignments[1] != assignments[3]

    def test_single_player(self):
        jersey_results = {1: {"dominant_color": [255, 0, 0]}}
        assignments = cluster_teams(jersey_results)
        assert assignments == {1: 0}

    def test_missing_color_uses_default(self):
        jersey_results = {
            1: {},
            2: {"dominant_color": [255, 0, 0]},
        }
        assignments = cluster_teams(jersey_results)
        assert len(assignments) == 2


# ---------------------------------------------------------------------------
# JerseyDetector class
# ---------------------------------------------------------------------------

class TestJerseyDetector:
    def test_store_and_retrieve_result(self):
        det = JerseyDetector(api_key="test")
        result = {"jersey_number": 23, "jersey_color_name": "red", "dominant_color": [255, 0, 0]}
        det.store_result(1, result)
        assert det.has_result(1)
        assert det.get_result(1) == result
        assert not det.has_result(2)

    def test_get_all_results(self):
        det = JerseyDetector(api_key="test")
        det.store_result(1, {"jersey_number": 23})
        det.store_result(2, {"jersey_number": 7})
        all_results = det.get_all_results()
        assert len(all_results) == 2

    def test_get_stats(self):
        det = JerseyDetector(api_key="test")
        stats = det.get_stats()
        assert stats["api_calls"] == 0
        assert stats["players_detected"] == 0

    def test_detect_empty_crop_returns_unknown(self):
        det = JerseyDetector(api_key="test")
        result = det.detect(np.empty(0))
        assert result["jersey_number"] is None
        assert result["jersey_color_name"] == "unknown"

    def test_detect_none_crop_returns_unknown(self):
        det = JerseyDetector(api_key="test")
        result = det.detect(None)
        assert result["jersey_number"] is None
