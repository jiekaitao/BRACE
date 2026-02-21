"""End-to-end test for multi-person tracking + analysis pipeline.

Runs the full backend logic (YOLO-pose + StreamingAnalyzer) on a real
video to verify the pipeline produces valid multi-subject output.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add backend to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from multi_person_tracker import MultiPersonTracker, TrackedPerson, denormalize_landmarks
from subject_manager import SubjectManager
from streaming_analyzer import StreamingAnalyzer
from brace.core.pose import coco_keypoints_to_landmarks, NUM_MP_LANDMARKS

# Test video path (exercise video with a visible person)
EXERCISE_VIDEO = Path(__file__).parent.parent / "data" / "sports_videos" / "exercise.mp4"


class TestMultiPersonTracker:
    """Tests for YOLO-pose + ByteTrack tracker."""

    @pytest.fixture(scope="class")
    def tracker(self):
        return MultiPersonTracker(model_name="yolo11x-pose.pt", conf_threshold=0.3)

    def test_tracker_init(self, tracker):
        assert tracker.model is not None
        assert tracker.conf_threshold == 0.3

    def test_empty_frame(self, tracker):
        """All-black frame should return no or few detections."""
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        persons = tracker.process_frame(black)
        assert isinstance(persons, list)
        # May or may not detect anyone in a black frame

    @pytest.mark.skipif(not EXERCISE_VIDEO.exists(), reason="Exercise video not found")
    def test_real_video_detects_people(self, tracker):
        """Run on a real exercise video and verify people are detected with keypoints."""
        cap = cv2.VideoCapture(str(EXERCISE_VIDEO))
        assert cap.isOpened()

        detected_any = False
        track_ids_seen = set()
        frame_count = 0

        while frame_count < 30:  # Test first 30 frames
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            persons = tracker.process_frame(rgb)

            for p in persons:
                assert isinstance(p, TrackedPerson)
                assert isinstance(p.track_id, int)
                # Keypoints should be (17, 3)
                assert p.keypoints.shape == (17, 3)
                # Bbox normalized should be in [0, 1]
                assert 0 <= p.bbox_normalized[0] <= 1
                assert 0 <= p.bbox_normalized[1] <= 1
                assert 0 <= p.bbox_normalized[2] <= 1
                assert 0 <= p.bbox_normalized[3] <= 1
                track_ids_seen.add(p.track_id)
                detected_any = True

            frame_count += 1

        cap.release()

        assert detected_any, "Should detect at least one person in exercise video"
        print(f"  Detected {len(track_ids_seen)} unique track IDs in {frame_count} frames")


class TestDenormalizeLandmarks:
    def test_offset_shift(self):
        """Landmarks should be shifted by crop offset."""
        lm = np.zeros((33, 4), dtype=np.float32)
        lm[0] = [100, 200, 0, 0.9]
        result = denormalize_landmarks(lm, (50, 30), 200, 300)
        assert result[0, 0] == pytest.approx(150, abs=0.01)
        assert result[0, 1] == pytest.approx(230, abs=0.01)

    def test_preserves_visibility(self):
        """Visibility should not change."""
        lm = np.zeros((33, 4), dtype=np.float32)
        lm[5, 3] = 0.95
        result = denormalize_landmarks(lm, (10, 20), 100, 100)
        assert result[5, 3] == pytest.approx(0.95, abs=0.01)


class TestCocoToLandmarks:
    """Tests for coco_keypoints_to_landmarks mapping."""

    def test_output_shape(self):
        """Output should be (33, 4) MediaPipe format."""
        kp = np.zeros((17, 3), dtype=np.float32)
        result = coco_keypoints_to_landmarks(kp, 640, 480)
        assert result.shape == (NUM_MP_LANDMARKS, 4)

    def test_hip_mapping(self):
        """COCO left_hip (11) should map to MediaPipe left_hip (23)."""
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[11] = [100, 200, 0.9]  # left_hip
        kp[12] = [150, 200, 0.85]  # right_hip
        result = coco_keypoints_to_landmarks(kp, 640, 480)
        assert result[23, 0] == pytest.approx(100, abs=0.01)
        assert result[23, 1] == pytest.approx(200, abs=0.01)
        assert result[23, 3] == pytest.approx(0.9, abs=0.01)
        assert result[24, 0] == pytest.approx(150, abs=0.01)
        assert result[24, 3] == pytest.approx(0.85, abs=0.01)

    def test_shoulder_mapping(self):
        """COCO shoulders (5, 6) should map to MediaPipe shoulders (11, 12)."""
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[5] = [120, 100, 0.95]  # left_shoulder
        kp[6] = [180, 100, 0.92]  # right_shoulder
        result = coco_keypoints_to_landmarks(kp, 640, 480)
        assert result[11, 0] == pytest.approx(120, abs=0.01)
        assert result[12, 0] == pytest.approx(180, abs=0.01)

    def test_feet_approximate_ankles(self):
        """Feet (31, 32) should copy ankle values (27, 28)."""
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[15] = [100, 500, 0.8]  # left_ankle
        kp[16] = [200, 500, 0.7]  # right_ankle
        result = coco_keypoints_to_landmarks(kp, 640, 480)
        # left_ankle (27) -> left_foot (31)
        assert result[31, 0] == result[27, 0]
        assert result[31, 1] == result[27, 1]
        assert result[31, 3] == result[27, 3]
        # right_ankle (28) -> right_foot (32)
        assert result[32, 0] == result[28, 0]
        assert result[32, 1] == result[28, 1]

    def test_all_feature_joints_mapped(self):
        """All 14 feature joints should have nonzero values when COCO keypoints are set."""
        from brace.core.pose import FEATURE_INDICES
        kp = np.ones((17, 3), dtype=np.float32) * 100
        kp[:, 2] = 0.9
        result = coco_keypoints_to_landmarks(kp, 640, 480)
        for idx in FEATURE_INDICES:
            assert result[idx, 3] > 0, f"Feature joint {idx} should have visibility > 0"

    def test_unmapped_joints_zero(self):
        """Unmapped MediaPipe joints (face, fingers, heels) should be zero."""
        kp = np.ones((17, 3), dtype=np.float32) * 100
        kp[:, 2] = 0.9
        result = coco_keypoints_to_landmarks(kp, 640, 480)
        # Face keypoints are now skipped (nose=0, eyes=2,5, ears=7,8)
        # plus inner eye, mouth, pinky, index, thumb, heels
        unmapped = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30]
        for idx in unmapped:
            assert result[idx, 3] == 0, f"Joint {idx} should be unmapped (zero)"


class TestSubjectManager:
    def test_create_and_label(self):
        mgr = SubjectManager(fps=30.0)
        a1 = mgr.get_or_create_analyzer(1)
        a2 = mgr.get_or_create_analyzer(2)
        assert isinstance(a1, StreamingAnalyzer)
        assert mgr.get_label(1) == "S1"
        assert mgr.get_label(2) == "S2"
        # Same track_id returns same analyzer
        assert mgr.get_or_create_analyzer(1) is a1

    def test_cleanup_stale(self):
        mgr = SubjectManager()
        a1 = mgr.get_or_create_analyzer(1)
        a2 = mgr.get_or_create_analyzer(2)
        a1.last_seen_frame = 0
        a2.last_seen_frame = 100

        removed = mgr.cleanup_stale(current_frame=100, max_missing=90)
        assert 1 in removed
        assert 2 not in removed
        assert 1 not in mgr.analyzers
        assert 2 in mgr.analyzers

    def test_active_track_ids(self):
        mgr = SubjectManager()
        mgr.get_or_create_analyzer(5)
        mgr.get_or_create_analyzer(3)
        ids = mgr.get_active_track_ids()
        assert set(ids) == {5, 3}


class TestStreamingAnalyzerUMAP:
    """Test UMAP and SRP joint extensions to StreamingAnalyzer."""

    def test_last_seen_frame_default(self):
        a = StreamingAnalyzer()
        assert a.last_seen_frame == 0

    def test_srp_joints_none_when_no_landmarks(self):
        a = StreamingAnalyzer()
        a.process_frame(None)
        assert a.get_srp_joints() is None

    def test_srp_joints_returned_for_valid_frame(self):
        """Build a valid synthetic frame and check SRP joints are produced."""
        a = StreamingAnalyzer()
        lm = _make_valid_landmarks()
        a.process_frame(lm)
        joints = a.get_srp_joints()
        assert joints is not None
        assert len(joints) == 14  # 14 feature joints
        assert len(joints[0]) in (2, 3)  # [x, y] or [x, y, z]

    def test_umap_needs_refit_logic(self):
        a = StreamingAnalyzer()
        # With < 20 features, should not need refit
        assert not a.needs_umap_refit()

        # Add 25 features — not enough for first fit (needs 50)
        lm = _make_valid_landmarks()
        for _ in range(25):
            a.process_frame(lm)
        assert not a.needs_umap_refit()

        # Add more to reach 55 — first fit triggers at 50
        for _ in range(30):
            a.process_frame(lm)
        assert a.needs_umap_refit()

        # After first fit, needs 200 more for refit
        a.run_umap_fit()
        assert not a.needs_umap_refit()

        # Add 100 more — still not enough (need 200)
        for _ in range(100):
            a.process_frame(lm)
        assert not a.needs_umap_refit()

        # Add 100 more to reach 200 since last fit
        for _ in range(100):
            a.process_frame(lm)
        assert a.needs_umap_refit()

    def test_umap_fit_produces_valid_output(self):
        a = StreamingAnalyzer()
        lm = _make_valid_landmarks()
        rng = np.random.RandomState(42)
        # Need enough frames for UMAP
        for i in range(25):
            # Vary the landmarks slightly so UMAP has variance
            varied_lm = lm.copy()
            varied_lm[:, 0] += rng.randn(33) * 5
            varied_lm[:, 1] += rng.randn(33) * 5
            a.process_frame(varied_lm)

        result = a.run_umap_fit()
        assert result["type"] == "full"
        assert len(result["points"]) == len(a.features_list)
        assert len(result["cluster_ids"]) == len(a.features_list)
        assert result["current_idx"] == len(a.features_list) - 1

        # Each point should be [float, float, float] (3D UMAP)
        for pt in result["points"]:
            assert len(pt) == 3
            assert isinstance(pt[0], float)
            assert isinstance(pt[1], float)
            assert isinstance(pt[2], float)

    def test_umap_transform_after_fit(self):
        a = StreamingAnalyzer()
        lm = _make_valid_landmarks()
        rng = np.random.RandomState(123)
        for i in range(25):
            varied = lm.copy()
            varied[:, 0] += rng.randn(33) * 5
            varied[:, 1] += rng.randn(33) * 3
            a.process_frame(varied)

        a.run_umap_fit()

        # Now process one more frame and transform
        a.process_frame(lm)
        feat = a.features_list[-1]
        result = a.run_umap_transform(feat)
        assert result is not None
        assert result["type"] == "append"
        assert len(result["new_points"]) == 1
        assert len(result["new_cluster_ids"]) == 1


def _make_valid_landmarks() -> np.ndarray:
    """Create a synthetic (33, 4) landmarks array that passes normalize_frame.

    Places hips, shoulders, and other joints in a realistic configuration.
    """
    lm = np.zeros((33, 4), dtype=np.float32)

    # Set all visibility high
    lm[:, 3] = 0.9

    # Hip center at (300, 400), width 60
    lm[23] = [270, 400, 0, 0.9]  # left hip
    lm[24] = [330, 400, 0, 0.9]  # right hip

    # Shoulders above hips
    lm[11] = [260, 300, 0, 0.9]  # left shoulder
    lm[12] = [340, 300, 0, 0.9]  # right shoulder

    # Elbows
    lm[13] = [230, 350, 0, 0.9]  # left elbow
    lm[14] = [370, 350, 0, 0.9]  # right elbow

    # Wrists
    lm[15] = [210, 400, 0, 0.9]  # left wrist
    lm[16] = [390, 400, 0, 0.9]  # right wrist

    # Knees
    lm[25] = [275, 500, 0, 0.9]  # left knee
    lm[26] = [325, 500, 0, 0.9]  # right knee

    # Ankles
    lm[27] = [275, 580, 0, 0.9]  # left ankle
    lm[28] = [325, 580, 0, 0.9]  # right ankle

    # Feet
    lm[31] = [275, 600, 0, 0.9]  # left foot
    lm[32] = [325, 600, 0, 0.9]  # right foot

    return lm
