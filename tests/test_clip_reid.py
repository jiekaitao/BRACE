"""Tests for CLIPReIDExtractor.

Tests use a mock approach since CLIP models may not be available in test
environments. The fallback (zero-embedding) behavior is always testable.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from clip_reid_extractor import CLIPReIDExtractor


class TestCLIPReIDFallback:
    """Test graceful fallback when CLIP is unavailable."""

    def test_returns_correct_dimension_when_unavailable(self):
        """When CLIP is not installed, extractor returns 768D zero vectors."""
        # Patch all CLIP imports to fail
        with patch.dict("sys.modules", {"clip": None, "transformers": None}):
            ext = CLIPReIDExtractor.__new__(CLIPReIDExtractor)
            ext._available = False
            ext._model = None
            ext._preprocess = None
            ext._device = "cpu"
            ext._model_name = "ViT-B/16"

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            bboxes = [(10, 10, 100, 200), (200, 50, 350, 300)]

            embeddings = ext.extract(frame, bboxes)

            assert len(embeddings) == 2
            for emb in embeddings:
                assert emb.shape == (768,)
                assert emb.dtype == np.float32
                assert np.allclose(emb, 0.0)

    def test_empty_bboxes_returns_empty(self):
        """Empty bbox list returns empty result."""
        ext = CLIPReIDExtractor.__new__(CLIPReIDExtractor)
        ext._available = False
        ext._model = None
        ext._preprocess = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        embeddings = ext.extract(frame, [])
        assert embeddings == []

    def test_available_property_false_when_no_model(self):
        """available property should be False when no CLIP model loaded."""
        ext = CLIPReIDExtractor.__new__(CLIPReIDExtractor)
        ext._available = False
        assert ext.available is False

    def test_available_property_true_when_loaded(self):
        """available property should be True when a model was loaded."""
        ext = CLIPReIDExtractor.__new__(CLIPReIDExtractor)
        ext._available = True
        assert ext.available is True


class TestCLIPReIDNormalization:
    """Test embedding normalization logic."""

    def test_normalize_embeddings_unit_norm(self):
        """Normalized embeddings should have unit L2 norm."""
        features = np.random.randn(3, 768).astype(np.float32)
        embeddings = CLIPReIDExtractor._normalize_embeddings(features)

        assert len(embeddings) == 3
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_normalize_embeddings_zero_vector(self):
        """Zero vectors should remain zero after normalization."""
        features = np.zeros((2, 768), dtype=np.float32)
        embeddings = CLIPReIDExtractor._normalize_embeddings(features)

        for emb in embeddings:
            assert np.allclose(emb, 0.0)

    def test_normalize_preserves_direction(self):
        """Normalization should preserve the direction of the vector."""
        v = np.array([[3.0, 4.0] + [0.0] * 766], dtype=np.float32)
        embeddings = CLIPReIDExtractor._normalize_embeddings(v)
        emb = embeddings[0]

        # Direction should be [0.6, 0.8, 0, 0, ...]
        assert abs(emb[0] - 0.6) < 1e-5
        assert abs(emb[1] - 0.8) < 1e-5


class TestCLIPReIDCropExtraction:
    """Test the crop extraction helper."""

    def test_crop_extraction_valid_bboxes(self):
        """Valid bboxes should produce 224x224 crops."""
        ext = CLIPReIDExtractor.__new__(CLIPReIDExtractor)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bboxes = [(10, 10, 100, 200), (200, 50, 350, 300)]

        crops = ext._extract_crops(frame, bboxes)

        assert len(crops) == 2
        for crop in crops:
            assert crop.shape == (224, 224, 3)

    def test_crop_extraction_degenerate_bbox(self):
        """Degenerate (zero-area) bboxes should produce a black 224x224 crop."""
        ext = CLIPReIDExtractor.__new__(CLIPReIDExtractor)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bboxes = [(100, 100, 100, 100)]  # zero area

        crops = ext._extract_crops(frame, bboxes)

        assert len(crops) == 1
        assert crops[0].shape == (224, 224, 3)
        assert np.allclose(crops[0], 0)

    def test_crop_extraction_out_of_bounds(self):
        """Bboxes that extend past frame edges should be clamped."""
        ext = CLIPReIDExtractor.__new__(CLIPReIDExtractor)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = [(-10, -10, 50, 50)]

        crops = ext._extract_crops(frame, bboxes)

        assert len(crops) == 1
        assert crops[0].shape == (224, 224, 3)
