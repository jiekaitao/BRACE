"""Tests for HybrIK estimator and UV texture projector."""

import numpy as np
import pytest
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from hybrik_estimator import HybrIKEstimator
from uv_texture_projector import UVTextureProjector


class TestHybrIKEstimator:
    """Tests for HybrIKEstimator graceful fallback."""

    def test_unavailable_when_hybrik_not_installed(self):
        """HybrIK should report unavailable when package is missing."""
        estimator = HybrIKEstimator(device="cpu")
        assert estimator.available is False

    def test_estimate_returns_none_list_when_unavailable(self):
        """estimate() should return [None, ...] when model is unavailable."""
        estimator = HybrIKEstimator(device="cpu")
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [(10, 20, 100, 200), (150, 50, 300, 400)]
        results = estimator.estimate(rgb, bboxes)
        assert len(results) == len(bboxes)
        assert all(r is None for r in results)

    def test_estimate_empty_bboxes(self):
        """estimate() with empty bboxes returns empty list."""
        estimator = HybrIKEstimator(device="cpu")
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        results = estimator.estimate(rgb, [])
        assert results == []

    def test_estimate_single_bbox_returns_none(self):
        """estimate() with single bbox returns [None] when model unavailable."""
        estimator = HybrIKEstimator(device="cpu")
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        results = estimator.estimate(rgb, [(0, 0, 100, 100)])
        assert len(results) == 1
        assert results[0] is None


class TestUVTextureProjector:
    """Tests for UVTextureProjector."""

    def test_init_default_size(self):
        """Default texture size should be 256x256."""
        proj = UVTextureProjector()
        tex = proj.get_texture()
        assert tex.shape == (256, 256, 3)
        assert tex.dtype == np.uint8

    def test_init_custom_size(self):
        """Custom texture size should be respected."""
        proj = UVTextureProjector(texture_size=128)
        tex = proj.get_texture()
        assert tex.shape == (128, 128, 3)

    def test_init_neutral_color(self):
        """Initial texture should be filled with neutral mannequin color (180)."""
        proj = UVTextureProjector()
        tex = proj.get_texture()
        assert np.all(tex == 180)

    def test_get_texture_returns_copy(self):
        """get_texture() should return a copy, not a reference."""
        proj = UVTextureProjector()
        tex1 = proj.get_texture()
        tex1[:] = 0  # modify the copy
        tex2 = proj.get_texture()
        assert np.all(tex2 == 180)  # original unchanged

    def test_reset_clears_to_neutral(self):
        """reset() should restore texture to neutral color."""
        proj = UVTextureProjector(texture_size=64)
        # Project something to change texture
        crop = np.full((100, 80, 3), 50, dtype=np.uint8)
        proj.project(crop)
        # Verify texture changed
        tex_after_project = proj.get_texture()
        assert not np.all(tex_after_project == 180)
        # Reset and verify back to neutral
        proj.reset()
        tex_after_reset = proj.get_texture()
        assert np.all(tex_after_reset == 180)

    def test_reset_clears_coverage(self):
        """reset() should clear internal coverage map."""
        proj = UVTextureProjector(texture_size=64)
        crop = np.full((100, 80, 3), 100, dtype=np.uint8)
        proj.project(crop)
        assert np.any(proj._coverage > 0)
        proj.reset()
        assert np.all(proj._coverage == 0)

    def test_project_simplified_changes_texture(self):
        """Projecting a crop should modify the texture."""
        proj = UVTextureProjector(texture_size=64)
        crop = np.full((100, 80, 3), 50, dtype=np.uint8)
        result = proj.project(crop)
        assert result.shape == (64, 64, 3)
        # After projection, texture should no longer be all neutral
        assert not np.all(result == 180)

    def test_project_returns_correct_shape(self):
        """project() should always return (texture_size, texture_size, 3)."""
        proj = UVTextureProjector(texture_size=128)
        crop = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        result = proj.project(crop)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8

    def test_project_empty_crop_returns_current(self):
        """project() with empty crop should return current texture unchanged."""
        proj = UVTextureProjector(texture_size=64)
        empty = np.empty(0)
        result = proj.project(empty)
        assert np.all(result == 180)

    def test_project_none_crop_returns_current(self):
        """project() with None crop should return current texture unchanged."""
        proj = UVTextureProjector(texture_size=64)
        result = proj.project(None)
        assert np.all(result == 180)

    def test_accumulation_over_frames(self):
        """Multiple project() calls should accumulate coverage."""
        proj = UVTextureProjector(texture_size=64)
        crop1 = np.full((100, 80, 3), 100, dtype=np.uint8)
        crop2 = np.full((100, 80, 3), 200, dtype=np.uint8)
        proj.project(crop1)
        cov1 = proj._coverage.copy()
        proj.project(crop2)
        cov2 = proj._coverage.copy()
        # Coverage should increase (or stay clamped at 1.0)
        assert np.all(cov2 >= cov1)
