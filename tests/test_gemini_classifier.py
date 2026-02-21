"""Tests for Gemini activity classifier with mocked API calls."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from gemini_classifier import (
    GeminiActivityClassifier,
    _crop_frame,
    _encode_frame_jpeg,
    _frames_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Create a deterministic synthetic RGB frame."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _crop_frame tests
# ---------------------------------------------------------------------------

class TestCropFrame:
    def test_basic_crop(self):
        frame = _make_rgb_frame(100, 200)
        crop = _crop_frame(frame, (0.25, 0.25, 0.75, 0.75), padding=0.0)
        assert crop.shape[0] == 50  # 100 * 0.5
        assert crop.shape[1] == 100  # 200 * 0.5
        assert crop.shape[2] == 3

    def test_full_frame_bbox(self):
        frame = _make_rgb_frame(100, 200)
        crop = _crop_frame(frame, (0.0, 0.0, 1.0, 1.0), padding=0.0)
        assert crop.shape == frame.shape

    def test_padding_clamps_to_image(self):
        frame = _make_rgb_frame(100, 200)
        crop = _crop_frame(frame, (0.0, 0.0, 1.0, 1.0), padding=0.5)
        # Padding should not go outside the image
        assert crop.shape == frame.shape

    def test_small_bbox_with_padding(self):
        frame = _make_rgb_frame(100, 200)
        crop = _crop_frame(frame, (0.4, 0.4, 0.6, 0.6), padding=0.15)
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0
        assert crop.ndim == 3

    def test_edge_bbox(self):
        frame = _make_rgb_frame(100, 200)
        crop = _crop_frame(frame, (0.9, 0.9, 1.0, 1.0), padding=0.1)
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0


# ---------------------------------------------------------------------------
# _encode_frame_jpeg tests
# ---------------------------------------------------------------------------

class TestEncodeFrameJpeg:
    def test_produces_jpeg_bytes(self):
        frame = _make_rgb_frame(100, 200)
        data = _encode_frame_jpeg(frame, max_dim=100)
        # JPEG starts with \xff\xd8
        assert data[:2] == b"\xff\xd8"

    def test_respects_max_dim(self):
        frame = _make_rgb_frame(1000, 2000)
        data = _encode_frame_jpeg(frame, max_dim=200)
        # Should be significantly smaller than raw
        assert len(data) < 1000 * 2000 * 3


# ---------------------------------------------------------------------------
# _frames_hash tests
# ---------------------------------------------------------------------------

class TestFramesHash:
    def test_deterministic(self):
        frames = [_make_rgb_frame(100, 100, seed=i) for i in range(3)]
        h1 = _frames_hash(frames)
        h2 = _frames_hash(frames)
        assert h1 == h2

    def test_different_frames_different_hash(self):
        frames_a = [_make_rgb_frame(100, 100, seed=0)]
        frames_b = [_make_rgb_frame(100, 100, seed=99)]
        assert _frames_hash(frames_a) != _frames_hash(frames_b)


# ---------------------------------------------------------------------------
# GeminiActivityClassifier tests
# ---------------------------------------------------------------------------

class TestGeminiActivityClassifier:
    def test_classify_returns_unknown_on_empty_frames(self):
        clf = GeminiActivityClassifier(api_key="test-key")
        result = clf.classify_activity([], (0.0, 0.0, 1.0, 1.0))
        assert result == "unknown"

    def test_classify_returns_unknown_when_model_unavailable(self):
        """When google-generativeai fails to init, classify returns 'unknown'."""
        clf = GeminiActivityClassifier(api_key="test-key")
        clf._init_error = "mock error"
        clf._model = None
        frames = [_make_rgb_frame()]
        result = clf.classify_activity(frames, (0.2, 0.2, 0.8, 0.8))
        assert result == "unknown"

    @patch("gemini_classifier.GeminiActivityClassifier._ensure_model")
    def test_classify_with_mocked_response(self, mock_ensure):
        """Mock the Gemini client to return a known activity label."""
        clf = GeminiActivityClassifier(api_key="test-key")

        # Create a mock client with client.models.generate_content()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "running"
        mock_client.models.generate_content.return_value = mock_response
        clf._model = mock_client
        clf._init_error = None

        frames = [_make_rgb_frame(seed=i) for i in range(3)]
        result = clf.classify_activity(frames, (0.2, 0.2, 0.8, 0.8))
        assert result == "running"
        mock_client.models.generate_content.assert_called_once()

    @patch("gemini_classifier.GeminiActivityClassifier._ensure_model")
    def test_caching(self, mock_ensure):
        """Second call with same frames should use cache, not call API."""
        clf = GeminiActivityClassifier(api_key="test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "squatting"
        mock_client.models.generate_content.return_value = mock_response
        clf._model = mock_client
        clf._init_error = None

        frames = [_make_rgb_frame(seed=42)]
        bbox = (0.1, 0.1, 0.9, 0.9)

        # First call hits the API
        r1 = clf.classify_activity(frames, bbox)
        assert r1 == "squatting"
        assert mock_client.models.generate_content.call_count == 1

        # Second call with same frames should use cache
        r2 = clf.classify_activity(frames, bbox)
        assert r2 == "squatting"
        assert mock_client.models.generate_content.call_count == 1  # NOT called again

    @patch("gemini_classifier.GeminiActivityClassifier._ensure_model")
    def test_classify_cleans_multiline_response(self, mock_ensure):
        """Multi-line response should be cleaned to first line."""
        clf = GeminiActivityClassifier(api_key="test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Running\nThe person appears to be jogging."
        mock_client.models.generate_content.return_value = mock_response
        clf._model = mock_client
        clf._init_error = None

        frames = [_make_rgb_frame(seed=7)]
        result = clf.classify_activity(frames, (0.2, 0.2, 0.8, 0.8))
        assert result == "running"

    @patch("gemini_classifier.GeminiActivityClassifier._ensure_model")
    def test_classify_truncates_long_response(self, mock_ensure):
        """Long response should be truncated to 3 words max."""
        clf = GeminiActivityClassifier(api_key="test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "doing some kind of complicated exercise movement"
        mock_client.models.generate_content.return_value = mock_response
        clf._model = mock_client
        clf._init_error = None

        frames = [_make_rgb_frame(seed=8)]
        result = clf.classify_activity(frames, (0.2, 0.2, 0.8, 0.8))
        words = result.split()
        assert len(words) <= 3

    @patch("gemini_classifier.GeminiActivityClassifier._ensure_model")
    def test_classify_handles_api_exception(self, mock_ensure):
        """API exception should result in 'unknown'."""
        clf = GeminiActivityClassifier(api_key="test-key")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API quota exceeded")
        clf._model = mock_client
        clf._init_error = None

        frames = [_make_rgb_frame(seed=9)]
        result = clf.classify_activity(frames, (0.2, 0.2, 0.8, 0.8))
        assert result == "unknown"

    def test_get_representative_frames(self):
        clf = GeminiActivityClassifier(api_key="test-key")

        all_frames = {i: _make_rgb_frame(seed=i) for i in range(20)}
        indices = list(range(20))

        def getter(idx):
            return all_frames.get(idx)

        selected = clf.get_representative_frames(indices, getter, count=4)
        assert len(selected) == 4
        assert all(isinstance(f, np.ndarray) for f in selected)

    def test_get_representative_frames_fewer_than_count(self):
        clf = GeminiActivityClassifier(api_key="test-key")
        frames = {0: _make_rgb_frame(seed=0), 1: _make_rgb_frame(seed=1)}
        selected = clf.get_representative_frames(
            [0, 1], lambda idx: frames.get(idx), count=4
        )
        assert len(selected) == 2

    def test_get_representative_frames_empty(self):
        clf = GeminiActivityClassifier(api_key="test-key")
        selected = clf.get_representative_frames([], lambda idx: None, count=4)
        assert selected == []

    def test_cache_result_and_lookup(self):
        clf = GeminiActivityClassifier(api_key="test-key")
        assert clf.classify_cached("cluster_0") is None
        clf.cache_result("cluster_0", "walking")
        assert clf.classify_cached("cluster_0") == "walking"

    def test_clear_cache(self):
        clf = GeminiActivityClassifier(api_key="test-key")
        clf.cache_result("k1", "running")
        clf.clear_cache()
        assert clf.classify_cached("k1") is None

    def test_available_false_without_sdk(self):
        """If model init fails, available should be False."""
        clf = GeminiActivityClassifier(api_key="test-key")
        clf._init_error = "no sdk"
        clf._model = None
        assert not clf.available
