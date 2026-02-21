"""Tests for PipelineResult, PoseBackend ABC, LegacyPoseBackend, and resolve_pipeline_results."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from pipeline_interface import PipelineResult, PoseBackend
from identity_resolver import IdentityResolver, ResolvedPipelineResult
from embedding_extractor import DummyExtractor


def _make_valid_mp_landmarks(
    hip_width: float = 60.0,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Create a synthetic (33, 4) landmarks array with valid hips/shoulders."""
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 0.9

    cx, cy = 300 + offset_x, 400 + offset_y
    hw = hip_width * scale / 2

    lm[23] = [cx - hw, cy, 0, 0.9]
    lm[24] = [cx + hw, cy, 0, 0.9]
    lm[11] = [cx - hw * 1.3, cy - hw * 3.3, 0, 0.9]
    lm[12] = [cx + hw * 1.3, cy - hw * 3.3, 0, 0.9]
    lm[13] = [cx - hw * 1.5, cy - hw * 1.7, 0, 0.9]
    lm[14] = [cx + hw * 1.5, cy - hw * 1.7, 0, 0.9]
    lm[15] = [cx - hw * 2.0, cy, 0, 0.9]
    lm[16] = [cx + hw * 2.0, cy, 0, 0.9]
    lm[25] = [cx - hw * 0.8, cy + hw * 3.3, 0, 0.9]
    lm[26] = [cx + hw * 0.8, cy + hw * 3.3, 0, 0.9]
    lm[27] = [cx - hw * 0.8, cy + hw * 6.0, 0, 0.9]
    lm[28] = [cx + hw * 0.8, cy + hw * 6.0, 0, 0.9]
    lm[31] = [cx - hw * 0.8, cy + hw * 6.7, 0, 0.9]
    lm[32] = [cx + hw * 0.8, cy + hw * 6.7, 0, 0.9]

    return lm


def _make_pipeline_result(track_id: int = 1, w: int = 640, h: int = 480) -> PipelineResult:
    """Create a PipelineResult with valid MediaPipe landmarks."""
    landmarks = _make_valid_mp_landmarks()
    return PipelineResult(
        track_id=track_id,
        bbox_pixel=(160, 100, 340, 450),
        bbox_normalized=(160 / w, 100 / h, 340 / w, 450 / h),
        landmarks_mp=landmarks,
    )


# =============================================================================
# PipelineResult dataclass tests
# =============================================================================

class TestPipelineResult:
    def test_required_fields(self):
        """PipelineResult stores track_id, bboxes, and landmarks_mp."""
        lm = np.zeros((33, 4), dtype=np.float32)
        pr = PipelineResult(
            track_id=7,
            bbox_pixel=(10, 20, 100, 200),
            bbox_normalized=(0.1, 0.2, 0.5, 0.8),
            landmarks_mp=lm,
        )
        assert pr.track_id == 7
        assert pr.bbox_pixel == (10, 20, 100, 200)
        assert pr.bbox_normalized == (0.1, 0.2, 0.5, 0.8)
        assert pr.landmarks_mp.shape == (33, 4)

    def test_optional_fields_default_none(self):
        """Optional fields default to None or empty."""
        lm = np.zeros((33, 4), dtype=np.float32)
        pr = PipelineResult(
            track_id=1,
            bbox_pixel=(0, 0, 10, 10),
            bbox_normalized=(0, 0, 0.5, 0.5),
            landmarks_mp=lm,
        )
        assert pr.landmarks_3d is None
        assert pr.smpl_params is None
        assert pr.smpl_texture_uv is None
        assert pr.reid_embedding is None

    def test_optional_fields_set(self):
        """Optional fields can be set explicitly."""
        lm = np.zeros((33, 4), dtype=np.float32)
        lm3d = np.zeros((133, 4), dtype=np.float32)
        emb = np.ones(512, dtype=np.float32)
        pr = PipelineResult(
            track_id=2,
            bbox_pixel=(0, 0, 10, 10),
            bbox_normalized=(0, 0, 0.5, 0.5),
            landmarks_mp=lm,
            landmarks_3d=lm3d,
            smpl_params={"betas": [0.0]},
            reid_embedding=emb,
        )
        assert pr.landmarks_3d is not None
        assert pr.landmarks_3d.shape == (133, 4)
        assert pr.smpl_params == {"betas": [0.0]}
        assert pr.reid_embedding is not None


# =============================================================================
# PoseBackend ABC tests
# =============================================================================

class TestPoseBackendABC:
    def test_cannot_instantiate_abc(self):
        """PoseBackend is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PoseBackend()

    def test_subclass_must_implement_process_frame_and_reset(self):
        """Subclass missing abstract methods raises TypeError."""
        class IncompleteBacked(PoseBackend):
            pass

        with pytest.raises(TypeError):
            IncompleteBacked()

    def test_concrete_subclass_works(self):
        """A fully implemented subclass can be instantiated and called."""
        class StubBackend(PoseBackend):
            def process_frame(self, rgb):
                return []

            def reset(self):
                pass

        backend = StubBackend()
        assert backend.process_frame(np.zeros((10, 10, 3))) == []
        backend.reset()

    def test_on_scene_cut_default_noop(self):
        """on_scene_cut has a default no-op implementation."""
        class StubBackend(PoseBackend):
            def process_frame(self, rgb):
                return []

            def reset(self):
                pass

        backend = StubBackend()
        backend.on_scene_cut()  # should not raise


# =============================================================================
# LegacyPoseBackend tests (mocked YOLO)
# =============================================================================

class TestLegacyPoseBackend:
    def test_process_frame_returns_pipeline_results(self):
        """LegacyPoseBackend.process_frame returns list of PipelineResult."""
        from unittest.mock import MagicMock, patch
        from multi_person_tracker import TrackedPerson

        # Create a mock TrackedPerson that MultiPersonTracker would return
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[5] = [200, 150, 0.9]   # left_shoulder
        kp[6] = [280, 150, 0.9]   # right_shoulder
        kp[11] = [220, 280, 0.9]  # left_hip
        kp[12] = [260, 280, 0.9]  # right_hip

        mock_person = TrackedPerson(
            track_id=1,
            bbox_pixel=(160, 100, 340, 450),
            bbox_normalized=(0.25, 0.208, 0.531, 0.938),
            keypoints=kp,
        )

        with patch("legacy_backend.MultiPersonTracker") as MockTracker:
            mock_tracker = MagicMock()
            mock_tracker.process_frame.return_value = [mock_person]
            MockTracker.return_value = mock_tracker

            from legacy_backend import LegacyPoseBackend
            backend = LegacyPoseBackend.__new__(LegacyPoseBackend)
            backend._tracker = mock_tracker

            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            results = backend.process_frame(rgb)

            assert len(results) == 1
            pr = results[0]
            assert isinstance(pr, PipelineResult)
            assert pr.track_id == 1
            assert pr.landmarks_mp.shape == (33, 4)
            assert pr.landmarks_3d is None
            assert pr.smpl_params is None
            assert pr.bbox_pixel == (160, 100, 340, 450)


# =============================================================================
# resolve_pipeline_results tests
# =============================================================================

class TestResolvePipelineResults:
    def test_basic_resolve(self):
        """resolve_pipeline_results returns ResolvedPipelineResult with correct fields."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        results = resolver.resolve_pipeline_results([pr], rgb, 640, 480)

        assert len(results) == 1
        rp = results[0]
        assert isinstance(rp, ResolvedPipelineResult)
        assert rp.pipeline_result is pr
        assert rp.subject_id >= 1
        assert rp.identity_status == "unknown"
        assert rp.label.startswith("S")
        assert isinstance(rp.identity_confidence, float)

    def test_same_track_id_keeps_same_subject(self):
        """Persistent track_id maps to the same subject_id across frames."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr = _make_pipeline_result(track_id=5)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        r1 = resolver.resolve_pipeline_results([pr], rgb, 640, 480)
        r2 = resolver.resolve_pipeline_results([pr], rgb, 640, 480)

        assert r1[0].subject_id == r2[0].subject_id

    def test_two_pipeline_results_get_different_ids(self):
        """Two simultaneous PipelineResults get different subject_ids."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr1 = _make_pipeline_result(track_id=1)
        pr2 = _make_pipeline_result(track_id=2)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        results = resolver.resolve_pipeline_results([pr1, pr2], rgb, 640, 480)
        assert len(results) == 2
        assert results[0].subject_id != results[1].subject_id

    def test_reappearing_person_matched_by_embedding(self):
        """Person who leaves and returns with new track_id gets same subject_id."""
        emb_a = np.random.RandomState(42).randn(512).astype(np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)

        extractor = DummyExtractor(embeddings={
            0: emb_a.copy(),
            1: emb_a.copy(),
            2: emb_a.copy(),
        })
        resolver = IdentityResolver(
            extractor,
            extraction_interval=1,
            match_threshold=0.5,
        )

        pr = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        r1 = resolver.resolve_pipeline_results([pr], rgb, 640, 480)
        original_sid = r1[0].subject_id

        r2 = resolver.resolve_pipeline_results([pr], rgb, 640, 480)
        assert r2[0].subject_id == original_sid

        resolver.cleanup_stale_tracks(set())

        pr_back = _make_pipeline_result(track_id=99)
        r3 = resolver.resolve_pipeline_results([pr_back], rgb, 640, 480)
        assert r3[0].subject_id == original_sid

    def test_empty_results_list(self):
        """Empty input returns empty output."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        results = resolver.resolve_pipeline_results([], rgb, 640, 480)
        assert results == []

    def test_pipeline_result_preserved_in_resolved(self):
        """The original PipelineResult is accessible via .pipeline_result."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr = _make_pipeline_result(track_id=42)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        results = resolver.resolve_pipeline_results([pr], rgb, 640, 480)
        assert results[0].pipeline_result.track_id == 42
        assert results[0].pipeline_result.bbox_pixel == pr.bbox_pixel
        np.testing.assert_array_equal(results[0].pipeline_result.landmarks_mp, pr.landmarks_mp)
