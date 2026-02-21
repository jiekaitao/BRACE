"""Tests for identity resolution: body proportions, IdentityResolver, and StreamingAnalyzer.absorb()."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from body_proportions import compute_limb_lengths, ProportionAccumulator
from embedding_extractor import DummyExtractor
from identity_resolver import IdentityResolver, ResolvedPerson
from multi_person_tracker import TrackedPerson
from streaming_analyzer import StreamingAnalyzer


def _make_valid_mp_landmarks(
    hip_width: float = 60.0,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Create a synthetic (33, 4) landmarks array with valid hips/shoulders.

    All joint positions are defined relative to hip width so that scaling
    hip_width uniformly scales the entire skeleton (for scale-invariance tests).

    Args:
        hip_width: distance between hips in pixels.
        scale: multiplier for the base hip_width.
        offset_x, offset_y: translation offsets.
    """
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 0.9  # all visible

    cx, cy = 300 + offset_x, 400 + offset_y
    hw = hip_width * scale / 2  # half hip width

    # All positions defined as multiples of hw for scale-invariance
    # Hips
    lm[23] = [cx - hw, cy, 0, 0.9]
    lm[24] = [cx + hw, cy, 0, 0.9]

    # Shoulders (wider, above hips)
    lm[11] = [cx - hw * 1.3, cy - hw * 3.3, 0, 0.9]
    lm[12] = [cx + hw * 1.3, cy - hw * 3.3, 0, 0.9]

    # Elbows
    lm[13] = [cx - hw * 1.5, cy - hw * 1.7, 0, 0.9]
    lm[14] = [cx + hw * 1.5, cy - hw * 1.7, 0, 0.9]

    # Wrists
    lm[15] = [cx - hw * 2.0, cy, 0, 0.9]
    lm[16] = [cx + hw * 2.0, cy, 0, 0.9]

    # Knees
    lm[25] = [cx - hw * 0.8, cy + hw * 3.3, 0, 0.9]
    lm[26] = [cx + hw * 0.8, cy + hw * 3.3, 0, 0.9]

    # Ankles
    lm[27] = [cx - hw * 0.8, cy + hw * 6.0, 0, 0.9]
    lm[28] = [cx + hw * 0.8, cy + hw * 6.0, 0, 0.9]

    # Feet
    lm[31] = [cx - hw * 0.8, cy + hw * 6.7, 0, 0.9]
    lm[32] = [cx + hw * 0.8, cy + hw * 6.7, 0, 0.9]

    return lm


def _make_tracked_person(track_id: int, w: int = 640, h: int = 480) -> TrackedPerson:
    """Create a TrackedPerson with valid COCO keypoints."""
    kp = np.zeros((17, 3), dtype=np.float32)
    # COCO: nose, leye, reye, lear, rear, lsho, rsho, lelb, relb, lwri, rwri, lhip, rhip, lknee, rknee, lank, rank
    kp[5] = [200, 150, 0.9]   # left_shoulder
    kp[6] = [280, 150, 0.9]   # right_shoulder
    kp[7] = [180, 200, 0.9]   # left_elbow
    kp[8] = [300, 200, 0.9]   # right_elbow
    kp[9] = [170, 260, 0.9]   # left_wrist
    kp[10] = [310, 260, 0.9]  # right_wrist
    kp[11] = [220, 280, 0.9]  # left_hip
    kp[12] = [260, 280, 0.9]  # right_hip
    kp[13] = [215, 360, 0.9]  # left_knee
    kp[14] = [265, 360, 0.9]  # right_knee
    kp[15] = [215, 430, 0.9]  # left_ankle
    kp[16] = [265, 430, 0.9]  # right_ankle

    return TrackedPerson(
        track_id=track_id,
        bbox_pixel=(160, 100, 340, 450),
        bbox_normalized=(160/w, 100/h, 340/w, 450/h),
        keypoints=kp,
    )


# =============================================================================
# BodyProportions tests
# =============================================================================

class TestBodyProportions:
    def test_compute_limb_lengths_valid(self):
        """Returns (10,) vector for a valid skeleton."""
        lm = _make_valid_mp_landmarks()
        result = compute_limb_lengths(lm)
        assert result is not None
        assert result.shape == (10,)
        assert np.all(result >= 0)

    def test_compute_limb_lengths_missing_hips(self):
        """Returns None if hips are not visible."""
        lm = _make_valid_mp_landmarks()
        lm[23, 3] = 0.0  # hide left hip
        result = compute_limb_lengths(lm)
        assert result is None

    def test_limb_lengths_scale_invariant(self):
        """Hip-width normalization makes limb lengths scale-invariant."""
        lm1 = _make_valid_mp_landmarks(hip_width=60.0, scale=1.0)
        lm2 = _make_valid_mp_landmarks(hip_width=120.0, scale=2.0)

        r1 = compute_limb_lengths(lm1)
        r2 = compute_limb_lengths(lm2)

        assert r1 is not None
        assert r2 is not None
        np.testing.assert_allclose(r1, r2, atol=0.01)

    def test_proportion_accumulator_needs_min_samples(self):
        """No proportion vector returned until min_samples reached."""
        acc = ProportionAccumulator(min_samples=3)
        lm = _make_valid_mp_landmarks()
        limbs = compute_limb_lengths(lm)
        assert limbs is not None

        acc.add(limbs)
        acc.add(limbs)
        assert acc.get_proportion_vector() is None

        acc.add(limbs)
        assert acc.get_proportion_vector() is not None

    def test_proportion_similarity_same_person(self):
        """Same skeleton proportions should have similarity > 0.95."""
        acc1 = ProportionAccumulator(min_samples=3)
        acc2 = ProportionAccumulator(min_samples=3)

        lm = _make_valid_mp_landmarks()
        rng = np.random.RandomState(42)
        for _ in range(5):
            noisy = lm.copy()
            noisy[:, :2] += rng.randn(33, 2) * 2  # small noise
            limbs = compute_limb_lengths(noisy)
            if limbs is not None:
                acc1.add(limbs)
                acc2.add(limbs)

        sim = acc1.similarity(acc2)
        assert sim is not None
        assert sim > 0.95

    def test_proportion_similarity_different_person(self):
        """Different body proportions should have lower similarity."""
        acc1 = ProportionAccumulator(min_samples=3)
        acc2 = ProportionAccumulator(min_samples=3)

        # Person 1: normal proportions
        lm1 = _make_valid_mp_landmarks(hip_width=60)
        # Person 2: very different body shape — long torso, short legs
        lm2 = _make_valid_mp_landmarks(hip_width=60)
        # Make legs much shorter for person 2
        lm2[25, 1] = lm2[23, 1] + 30   # knees close to hips
        lm2[26, 1] = lm2[24, 1] + 30
        lm2[27, 1] = lm2[23, 1] + 50   # ankles close
        lm2[28, 1] = lm2[24, 1] + 50
        # Make torso much longer
        lm2[11, 1] = lm2[23, 1] - 200
        lm2[12, 1] = lm2[24, 1] - 200

        for _ in range(5):
            l1 = compute_limb_lengths(lm1)
            l2 = compute_limb_lengths(lm2)
            if l1 is not None:
                acc1.add(l1)
            if l2 is not None:
                acc2.add(l2)

        sim = acc1.similarity(acc2)
        assert sim is not None
        assert sim < 0.90


# =============================================================================
# IdentityResolver tests
# =============================================================================

class TestIdentityResolver:
    def test_new_person_gets_unknown_status_with_s_label(self):
        """First appearance produces an 'unknown' status but always S# label."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        person = _make_tracked_person(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        results = resolver.resolve([person], rgb, 640, 480)

        assert len(results) == 1
        rp = results[0]
        assert rp.identity_status == "unknown"
        assert rp.label == "S1"

    def test_same_track_id_keeps_same_subject(self):
        """Persistent track_id maps to the same subject_id across frames."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        person = _make_tracked_person(track_id=5)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        r1 = resolver.resolve([person], rgb, 640, 480)
        r2 = resolver.resolve([person], rgb, 640, 480)

        assert r1[0].subject_id == r2[0].subject_id

    def test_two_persons_get_different_ids(self):
        """Two simultaneous persons get different subject_ids."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        p1 = _make_tracked_person(track_id=1)
        p2 = _make_tracked_person(track_id=2)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        results = resolver.resolve([p1, p2], rgb, 640, 480)
        assert len(results) == 2
        assert results[0].subject_id != results[1].subject_id

    def test_reappearing_person_matched_by_embedding(self):
        """Person who leaves and returns with new track_id gets same subject_id."""
        # Create distinctive embeddings for person A
        emb_a = np.random.RandomState(42).randn(512).astype(np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)

        # DummyExtractor: index 0 = first extraction for track 1,
        # index 1 = second frame extraction for track 1,
        # index 2 = extraction for track 99 (re-appeared person A)
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

        person_a = _make_tracked_person(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: person A appears
        r1 = resolver.resolve([person_a], rgb, 640, 480)
        original_sid = r1[0].subject_id

        # Frame 2: person A still here (build gallery)
        r2 = resolver.resolve([person_a], rgb, 640, 480)
        assert r2[0].subject_id == original_sid

        # Person A disappears — cleanup stale track
        resolver.cleanup_stale_tracks(set())

        # Frame 3: person A re-appears with new track_id 99
        person_a_back = _make_tracked_person(track_id=99)
        r3 = resolver.resolve([person_a_back], rgb, 640, 480)

        # Should be matched to original subject
        assert r3[0].subject_id == original_sid

    def test_cleanup_removes_track_mapping_keeps_identity(self):
        """Stale track mapping is removed but gallery is preserved."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        person = _make_tracked_person(track_id=10)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        r = resolver.resolve([person], rgb, 640, 480)
        sid = r[0].subject_id

        # Track 10 disappears
        resolver.cleanup_stale_tracks(set())

        # Track mapping should be gone
        assert 10 not in resolver._track_to_subject

        # But gallery should still exist
        assert sid in resolver._galleries

    def test_status_promotion(self):
        """Status progresses: unknown -> tentative -> confirmed as gallery builds."""
        # Use consistent embeddings for high gallery_consistency
        emb = np.ones(512, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)

        # Provide enough embeddings for promotion
        embs = {i: emb.copy() for i in range(20)}
        extractor = DummyExtractor(embeddings=embs)

        resolver = IdentityResolver(
            extractor,
            extraction_interval=1,
            match_threshold=0.5,
            confirm_threshold=0.8,
        )

        person = _make_tracked_person(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        # First frame: unknown
        r = resolver.resolve([person], rgb, 640, 480)
        assert r[0].identity_status == "unknown"

        # Feed more frames until gallery builds consistency
        statuses = set()
        for _ in range(15):
            r = resolver.resolve([person], rgb, 640, 480)
            statuses.add(r[0].identity_status)

        # Should have progressed through at least tentative
        assert "tentative" in statuses or "confirmed" in statuses


# =============================================================================
# Cross-cut re-identification tests
# =============================================================================

class TestIdentityResolverCrossCut:
    def test_on_scene_cut_clears_track_mappings(self):
        """on_scene_cut() clears all track->subject mappings."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        person = _make_tracked_person(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        resolver.resolve([person], rgb, 640, 480)

        assert 1 in resolver._track_to_subject

        resolver.on_scene_cut()

        assert len(resolver._track_to_subject) == 0

    def test_on_scene_cut_preserves_galleries(self):
        """on_scene_cut() preserves all galleries for future matching."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        person = _make_tracked_person(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        r = resolver.resolve([person], rgb, 640, 480)
        sid = r[0].subject_id

        resolver.on_scene_cut()

        assert sid in resolver._galleries

    def test_on_scene_cut_clears_active_subjects(self):
        """After scene cut, no subjects should be considered active."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        person = _make_tracked_person(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        resolver.resolve([person], rgb, 640, 480)

        assert len(resolver._active_subjects) > 0

        resolver.on_scene_cut()

        assert len(resolver._active_subjects) == 0

    def test_post_cut_mode_activates_with_cross_cut_extractor(self):
        """Post-cut mode is activated only when cross_cut_extractor is set."""
        extractor = DummyExtractor()
        cross_cut = DummyExtractor()
        resolver = IdentityResolver(
            extractor, cross_cut_extractor=cross_cut, extraction_interval=1
        )

        assert resolver.in_post_cut_mode is False

        resolver.on_scene_cut()

        assert resolver.in_post_cut_mode is True

    def test_post_cut_mode_not_activated_without_cross_cut_extractor(self):
        """Without cross_cut_extractor, on_scene_cut does not enter post-cut mode."""
        extractor = DummyExtractor()
        resolver = IdentityResolver(extractor, extraction_interval=1)

        resolver.on_scene_cut()

        assert resolver.in_post_cut_mode is False

    def test_post_cut_mode_expires_after_n_frames(self):
        """Post-cut mode deactivates after post_cut_frames frames."""
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        embs = {i: emb.copy() for i in range(20)}
        extractor = DummyExtractor(embeddings=embs)
        cross_cut = DummyExtractor(embeddings={i: emb.copy() for i in range(20)})

        resolver = IdentityResolver(
            extractor,
            cross_cut_extractor=cross_cut,
            extraction_interval=1,
            post_cut_frames=3,
        )

        person = _make_tracked_person(track_id=1)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        # Build gallery
        resolver.resolve([person], rgb, 640, 480)
        resolver.on_scene_cut()

        assert resolver.in_post_cut_mode is True

        # Process 3 frames in post-cut mode (using resolve_pipeline_results)
        from pipeline_interface import PipelineResult
        lm = _make_valid_mp_landmarks()
        pr = PipelineResult(
            track_id=50,
            bbox_pixel=(160, 100, 340, 450),
            bbox_normalized=(0.25, 0.2, 0.53, 0.94),
            landmarks_mp=lm,
        )

        for _ in range(3):
            resolver.resolve_pipeline_results([pr], rgb, 640, 480)

        assert resolver.in_post_cut_mode is False


# =============================================================================
# StreamingAnalyzer.absorb() tests
# =============================================================================

class TestStreamingAnalyzerAbsorb:
    def test_absorb_concatenates_features(self):
        """Features lists are concatenated after absorb."""
        a1 = StreamingAnalyzer()
        a2 = StreamingAnalyzer()

        lm = _make_valid_mp_landmarks()
        for _ in range(5):
            a1.process_frame(lm)
        for _ in range(3):
            a2.process_frame(lm)

        n1 = len(a1.features_list)
        n2 = len(a2.features_list)

        a1.absorb(a2)
        assert len(a1.features_list) == n1 + n2

    def test_absorb_resets_umap(self):
        """UMAP state is cleared after absorb for full refit."""
        a1 = StreamingAnalyzer()
        a2 = StreamingAnalyzer()

        # Simulate a1 having a UMAP mapper
        a1._umap_mapper = "fake_mapper"
        a1._umap_embeddings = [[0, 1], [2, 3]]
        a1._umap_last_fit_count = 10

        lm = _make_valid_mp_landmarks()
        a2.process_frame(lm)

        a1.absorb(a2)

        assert a1._umap_mapper is None
        assert a1._umap_embeddings == []
        assert a1._umap_last_fit_count == 0

    def test_absorb_forces_reanalysis(self):
        """_last_analysis_frame is reset to 0 after absorb."""
        a1 = StreamingAnalyzer()
        a1._last_analysis_frame = 100

        a2 = StreamingAnalyzer()
        lm = _make_valid_mp_landmarks()
        a2.process_frame(lm)

        a1.absorb(a2)
        assert a1._last_analysis_frame == 0
