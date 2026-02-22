"""Tests for jersey-based identity resolution extensions."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from embedding_extractor import DummyExtractor
from identity_resolver import IdentityResolver, _IdentityGallery
from pipeline_interface import PipelineResult
from subject_manager import SubjectManager
from streaming_analyzer import StreamingAnalyzer


def _make_valid_mp_landmarks(
    hip_width: float = 60.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Create a synthetic (33, 4) landmarks array."""
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 0.9
    cx, cy = 300 + offset_x, 400 + offset_y
    hw = hip_width / 2
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


def _make_pipeline_result(track_id: int, offset_x: float = 0.0) -> PipelineResult:
    lm = _make_valid_mp_landmarks(offset_x=offset_x)
    x1 = (250 + offset_x) / 640
    return PipelineResult(
        track_id=track_id,
        bbox_pixel=(int(250 + offset_x), 200, int(350 + offset_x), 450),
        bbox_normalized=(x1, 200 / 480, x1 + 100 / 640, 450 / 480),
        landmarks_mp=lm,
    )


# ---------------------------------------------------------------------------
# _IdentityGallery jersey field tests
# ---------------------------------------------------------------------------

class TestGalleryJerseyFields:
    def test_gallery_has_jersey_fields(self):
        """Gallery should have jersey_number, jersey_color, first_seen_frame, total_frames."""
        g = _IdentityGallery(subject_id=1)
        assert g.jersey_number is None
        assert g.jersey_color is None
        assert g.first_seen_frame == 0
        assert g.total_frames == 0

    def test_gallery_set_jersey_fields(self):
        g = _IdentityGallery(subject_id=1)
        g.jersey_number = 23
        g.jersey_color = "red"
        assert g.jersey_number == 23
        assert g.jersey_color == "red"


# ---------------------------------------------------------------------------
# IdentityResolver.set_jersey tests
# ---------------------------------------------------------------------------

class TestSetJersey:
    def test_set_jersey_on_existing_subject(self):
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        resolved = resolver.resolve_pipeline_results([pr], rgb, 640, 480)
        sid = resolved[0].subject_id

        resolver.set_jersey(sid, number=23, color="red")
        gallery = resolver._galleries[sid]
        assert gallery.jersey_number == 23
        assert gallery.jersey_color == "red"

    def test_set_jersey_nonexistent_subject(self):
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)
        # Should not raise
        resolver.set_jersey(999, number=10, color="blue")

    def test_set_jersey_partial(self):
        """Can set just number or just color."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        resolved = resolver.resolve_pipeline_results([pr], rgb, 640, 480)
        sid = resolved[0].subject_id

        resolver.set_jersey(sid, number=7)
        assert resolver._galleries[sid].jersey_number == 7
        assert resolver._galleries[sid].jersey_color is None


# ---------------------------------------------------------------------------
# IdentityResolver._match_by_jersey tests
# ---------------------------------------------------------------------------

class TestMatchByJersey:
    def test_match_by_jersey_number(self):
        """Subject with same jersey number should be matched."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        # Create subject with jersey
        pr1 = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = resolver.resolve_pipeline_results([pr1], rgb, 640, 480)
        sid1 = r1[0].subject_id
        resolver.set_jersey(sid1, number=23, color="red")

        # Now test _match_by_jersey
        match = resolver._match_by_jersey(23, "red", skip=set())
        assert match == sid1

    def test_no_match_different_number(self):
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr1 = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = resolver.resolve_pipeline_results([pr1], rgb, 640, 480)
        sid1 = r1[0].subject_id
        resolver.set_jersey(sid1, number=23, color="red")

        match = resolver._match_by_jersey(99, "red", skip=set())
        assert match is None

    def test_match_skips_active(self):
        """Subjects in skip set should not be matched."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr1 = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = resolver.resolve_pipeline_results([pr1], rgb, 640, 480)
        sid1 = r1[0].subject_id
        resolver.set_jersey(sid1, number=23, color="red")

        match = resolver._match_by_jersey(23, "red", skip={sid1})
        assert match is None

    def test_match_by_number_only(self):
        """Match by number even if color is None."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr1 = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = resolver.resolve_pipeline_results([pr1], rgb, 640, 480)
        sid1 = r1[0].subject_id
        resolver.set_jersey(sid1, number=23, color="red")

        # Match with number only (no color)
        match = resolver._match_by_jersey(23, None, skip=set())
        assert match == sid1


# ---------------------------------------------------------------------------
# IdentityResolver.merge_by_jersey tests
# ---------------------------------------------------------------------------

class TestMergeByJersey:
    def test_merge_two_subjects_same_jersey(self):
        """Two subjects with same jersey should be merged into one."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr1 = _make_pipeline_result(track_id=1)
        pr2 = _make_pipeline_result(track_id=2, offset_x=100)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        r1 = resolver.resolve_pipeline_results([pr1], rgb, 640, 480)
        sid1 = r1[0].subject_id

        # Make track 1 disappear, track 2 appear
        resolver.cleanup_stale_tracks(set())
        r2 = resolver.resolve_pipeline_results([pr2], rgb, 640, 480)
        sid2 = r2[0].subject_id

        assert sid1 != sid2

        # Set same jersey on both
        resolver.set_jersey(sid1, number=23, color="red")
        resolver.set_jersey(sid2, number=23, color="red")

        merged = resolver.merge_by_jersey()
        # Should have merged
        assert len(merged) > 0 or sid1 == sid2  # merged pairs returned
        # After merge, only one gallery should have jersey #23
        jerseys_23 = [
            sid for sid, g in resolver._galleries.items()
            if g.jersey_number == 23
        ]
        assert len(jerseys_23) == 1

    def test_no_merge_different_jersey(self):
        """Subjects with different jerseys should not be merged."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr1 = _make_pipeline_result(track_id=1)
        pr2 = _make_pipeline_result(track_id=2, offset_x=100)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        r1 = resolver.resolve_pipeline_results([pr1], rgb, 640, 480)
        sid1 = r1[0].subject_id
        resolver.cleanup_stale_tracks(set())

        r2 = resolver.resolve_pipeline_results([pr2], rgb, 640, 480)
        sid2 = r2[0].subject_id

        resolver.set_jersey(sid1, number=23, color="red")
        resolver.set_jersey(sid2, number=7, color="blue")

        merged = resolver.merge_by_jersey()
        assert len(merged) == 0


# ---------------------------------------------------------------------------
# IdentityResolver.get_confirmed_subjects tests
# ---------------------------------------------------------------------------

class TestGetConfirmedSubjects:
    def test_returns_dict(self):
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)
        result = resolver.get_confirmed_subjects()
        assert isinstance(result, dict)

    def test_includes_jersey_info(self):
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        resolved = resolver.resolve_pipeline_results([pr], rgb, 640, 480)
        sid = resolved[0].subject_id
        resolver.set_jersey(sid, number=23, color="red")

        subjects = resolver.get_confirmed_subjects()
        assert sid in subjects
        assert subjects[sid]["jersey_number"] == 23
        assert subjects[sid]["jersey_color"] == "red"


# ---------------------------------------------------------------------------
# IdentityResolver.total_frames increment tests
# ---------------------------------------------------------------------------

class TestTotalFramesIncrement:
    def test_total_frames_incremented(self):
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        pr = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(5):
            resolver.resolve_pipeline_results([pr], rgb, 640, 480)

        sid = resolver._track_to_subject[1]
        assert resolver._galleries[sid].total_frames == 5


# ---------------------------------------------------------------------------
# Fragment filtering tests
# ---------------------------------------------------------------------------

class TestFragmentFiltering:
    def test_get_confirmed_subjects_excludes_short_lived(self):
        """Subjects with very few frames should be filtered as fragments."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        # Subject with many frames
        pr1 = _make_pipeline_result(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(30):
            resolver.resolve_pipeline_results([pr1], rgb, 640, 480)
        sid1 = resolver._track_to_subject[1]
        resolver.set_jersey(sid1, number=23, color="red")

        # Subject with very few frames
        resolver.cleanup_stale_tracks(set())
        pr2 = _make_pipeline_result(track_id=2, offset_x=200)
        resolver.resolve_pipeline_results([pr2], rgb, 640, 480)
        sid2 = resolver._track_to_subject[2]

        subjects = resolver.get_confirmed_subjects(min_frames=10)
        assert sid1 in subjects
        assert sid2 not in subjects


# ---------------------------------------------------------------------------
# SubjectManager.merge_subject tests
# ---------------------------------------------------------------------------

class TestSubjectManagerMerge:
    def test_merge_subject(self):
        """merge_subject transfers analyzer from source to target."""
        manager = SubjectManager(fps=30.0)
        a1 = manager.get_or_create_analyzer(1)
        a2 = manager.get_or_create_analyzer(2)

        lm = _make_valid_mp_landmarks()
        for _ in range(5):
            a1.process_frame(lm)
        for _ in range(3):
            a2.process_frame(lm)

        n1_before = len(a1.features_list)
        n2_before = len(a2.features_list)

        manager.merge_subject(from_id=2, to_id=1)

        # Source should be removed
        assert 2 not in manager.analyzers
        # Target should have absorbed source features
        assert len(manager.analyzers[1].features_list) == n1_before + n2_before

    def test_merge_nonexistent_source(self):
        """Merging nonexistent source should not raise."""
        manager = SubjectManager(fps=30.0)
        manager.get_or_create_analyzer(1)
        manager.merge_subject(from_id=999, to_id=1)  # should not raise

    def test_merge_nonexistent_target(self):
        """Merging to nonexistent target should not raise."""
        manager = SubjectManager(fps=30.0)
        manager.get_or_create_analyzer(1)
        manager.merge_subject(from_id=1, to_id=999)  # should not raise
