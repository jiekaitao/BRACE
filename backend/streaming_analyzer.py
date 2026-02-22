"""Stateful per-session analysis engine wrapping brace core functions.

Accumulates landmarks frame-by-frame and periodically re-runs the full
segmentation -> clustering -> consistency pipeline on the growing buffer.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from typing import Any

import numpy as np


class OneEuroFilter:
    """Casiez et al. 2012 — adaptive low-pass for real-time smoothing."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.5, beta: float = 0.01, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        edx = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = edx
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(cutoff)
        result = a * x + (1 - a) * self.x_prev
        self.x_prev = result
        return result

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0

from brace.core.motion_segments import (
    normalize_frame,
    normalize_frame_3d,
    normalize_frame_3d_real,
    feature_vector,
    segment_motions,
    cluster_segments,
    analyze_consistency,
    _resample_segment,
)
from brace.core.pose import (
    FEATURE_INDICES,
    NUM_MP_LANDMARKS,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
)
try:
    from backend.movement_quality import MovementQualityTracker
except ImportError:
    from movement_quality import MovementQualityTracker

# Head landmark indices in MediaPipe 33-point layout (for concussion tracking)
_HEAD_NOSE = 0
_HEAD_LEFT_EAR = 7
_HEAD_RIGHT_EAR = 8
_HEAD_VISIBILITY_THRESHOLD = 0.3



class StreamingAnalyzer:
    """Incremental motion analysis for a single WebSocket session."""

    MAX_ANALYSIS_WINDOW = 2000  # ~67 seconds at 30fps

    def __init__(self, fps: float = 30.0, cluster_threshold: float = 2.0, risk_modifiers: Any = None, vectorai_store: Any = None):
        self.fps = fps
        self.cluster_threshold = cluster_threshold

        # Accumulated buffers
        self.landmarks_list: list[np.ndarray | None] = []
        self.features_list: list[np.ndarray] = []
        self.raw_features_list: list[np.ndarray] = []  # hip-centered only (no SRP rotation/scale)
        self.valid_indices: list[int] = []
        self.frame_count = 0

        # Cached analysis results (swapped atomically)
        self._segments: list[dict] = []
        self._cluster_analysis: dict[int, dict] = {}
        self._n_clusters = 0

        # Lookup: valid_index -> segment info
        self._frame_to_segment: dict[int, dict] = {}

        # Reanalysis tracking
        self._last_analysis_frame = 0
        self._lock = threading.Lock()

        # Multi-person tracking support
        self.last_seen_frame: int = 0

        # UMAP embedding state
        self._umap_mapper: Any = None  # umap.UMAP instance
        self._umap_embeddings: list[list[float]] = []
        self._umap_cluster_ids: list[int | None] = []
        self._umap_last_fit_count: int = 0
        self._umap_refit_pending: bool = False  # True while background refit is running
        self._umap_refit_result: dict[str, Any] | None = None  # cached result from bg refit

        # Current SRP-normalized joints for skeleton graph
        self._current_srp_joints: list[list[float]] | None = None
        self._current_joint_vis: list[float] | None = None

        # Cluster representatives: cluster_id -> resampled mean trajectory (30, feat_dim)
        self._cluster_representatives: dict[int, np.ndarray] = {}
        self._last_sent_reps_version: int = 0  # tracks when reps were last serialized
        self._reps_version: int = 0  # incremented when reps change

        # Feature dimension lock: once set, all frames must match
        # None = not yet determined, 2 = 2D (28D features), 3 = 3D (42D features)
        self._norm_dims: int | None = None

        # One Euro temporal smoothing for skeleton display joints
        self._joint_filters: dict[tuple[int, int], OneEuroFilter] = {}

        # Gemini activity classification: cluster_id -> label
        self._activity_labels: dict[int, str] = {}
        # Clusters already sent for classification (avoid re-requesting)
        self._pending_classification: set[int] = set()

        # Video loop handling: freeze analysis after first loop
        self._frozen: bool = False  # when True, stop accumulating features

        # Image dimensions for normalizing landmark pixel coords to [0,1]
        self._img_w: int = 1
        self._img_h: int = 1

        # Velocity tracking
        self.velocity_list: list[float] = []
        self._rolling_velocity: float = 0.0
        self._peak_velocity: float = 0.0
        self._velocity_window: deque[float] = deque(maxlen=300)  # ~10s at 30fps
        self._fatigue_index: float = 0.0
        self._fatigue_ema: float = 0.0

        # Head landmark filters (tighter smoothing for derivative-based concussion tracking)
        self._head_filters: dict[tuple[int, int], OneEuroFilter] = {}

        # Movement quality tracker (biomechanics, form scoring, fatigue detection)
        self._quality_tracker = MovementQualityTracker(fps=fps, risk_modifiers=risk_modifiers)
        self._cluster_quality: dict[int, dict] = {}  # per-cluster quality analysis

        # VectorAI store for motion segment persistence
        self._vectorai_store = vectorai_store
        self._session_id: str = ""  # set externally by caller
        self._person_id: str = ""   # set externally by caller
        self._stored_clusters: set[int] = set()  # clusters already stored in VectorAI


    def process_frame(self, landmarks_xyzv: np.ndarray | None, img_wh: tuple[int, int] | None = None) -> dict[str, Any]:
        """Fast path: normalize + feature extract a single frame.

        Args:
            landmarks_xyzv: (33, 4) array [x_px, y_px, z, visibility] or None.
            img_wh: (width, height) of the source image for bbox normalization.

        Returns:
            Per-frame response dict for the WebSocket client.
        """
        frame_idx = self.frame_count
        self.frame_count += 1
        self.landmarks_list.append(landmarks_xyzv)

        if img_wh is not None:
            self._img_w, self._img_h = img_wh

        if landmarks_xyzv is None:
            self._current_srp_joints = None
            self._current_joint_vis = None
            return self._build_response(frame_idx, None, None)

        # Detect if we have real 3D depth: non-zero Z on hips and shoulders
        has_real_3d = self._has_real_depth(landmarks_xyzv)

        # Lock feature dimension on first successful frame
        if has_real_3d and self._norm_dims is None:
            self._norm_dims = 3
        elif not has_real_3d and self._norm_dims is None:
            self._norm_dims = 2

        # Choose normalization based on locked dimension
        if self._norm_dims == 3 and has_real_3d:
            norm = normalize_frame_3d_real(landmarks_xyzv)
        else:
            norm = normalize_frame(landmarks_xyzv)

        if norm is None:
            self._current_srp_joints = None
            self._current_joint_vis = None
            return self._build_response(frame_idx, landmarks_xyzv, None)

        # Store hip-centered raw joints for skeleton graph (no normalization)
        lhip = landmarks_xyzv[LEFT_HIP, :3].astype(np.float64)
        rhip = landmarks_xyzv[RIGHT_HIP, :3].astype(np.float64)
        pelvis = (lhip + rhip) * 0.5
        raw_joints = landmarks_xyzv[FEATURE_INDICES, :3].astype(np.float64) - pelvis

        # Apply One Euro temporal smoothing to skeleton display joints
        # Skip filter update for low-visibility joints (hold last good position)
        smoothed_joints = np.copy(raw_joints)
        joint_vis = []
        for j in range(raw_joints.shape[0]):
            mp_idx = FEATURE_INDICES[j]
            vis = float(landmarks_xyzv[mp_idx, 3])
            joint_vis.append(round(vis, 3))
            for c in range(raw_joints.shape[1]):
                key = (j, c)
                if key not in self._joint_filters:
                    self._joint_filters[key] = OneEuroFilter(freq=self.fps, min_cutoff=1.5, beta=0.01)
                if vis >= 0.3:
                    smoothed_joints[j, c] = self._joint_filters[key](float(raw_joints[j, c]))
                elif self._joint_filters[key].x_prev is not None:
                    # Low visibility — hold last good filtered position
                    smoothed_joints[j, c] = self._joint_filters[key].x_prev
        self._current_srp_joints = smoothed_joints.tolist()
        self._current_joint_vis = joint_vis

        feat = feature_vector(norm)
        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
            return self._build_response(frame_idx, landmarks_xyzv, None)

        # When frozen (video looped), don't accumulate more features —
        # just render the skeleton and reuse the existing analysis
        if self._frozen:
            return self._build_response(frame_idx, landmarks_xyzv, None)

        # Store raw hip-centered feature vector using smoothed joints
        raw_feat = smoothed_joints.flatten().astype(np.float32)
        # Trim to match feat dimension (2D features use xy only)
        if feat.shape[0] == len(FEATURE_INDICES) * 2:
            raw_feat = smoothed_joints[:, :2].flatten().astype(np.float32)

        valid_idx = len(self.features_list)
        self.features_list.append(feat)
        self.raw_features_list.append(raw_feat)
        self.valid_indices.append(frame_idx)

        # Compute per-frame velocity from feature space displacement
        if len(self.features_list) >= 2:
            vel = float(np.linalg.norm(self.features_list[-1] - self.features_list[-2]))
        else:
            vel = 0.0
        self.velocity_list.append(vel)

        # Rolling velocity (EMA, alpha=0.1 ≈ 10-frame smoothing)
        alpha_vel = 0.1
        self._rolling_velocity = alpha_vel * vel + (1.0 - alpha_vel) * self._rolling_velocity

        # Peak velocity (track max, slow decay to allow fatigue detection)
        if self._rolling_velocity > self._peak_velocity:
            self._peak_velocity = self._rolling_velocity
        else:
            # Slow decay: 0.999 per frame ≈ 3% decay per second at 30fps
            self._peak_velocity *= 0.999

        # Fatigue index: compare first half vs second half of velocity window
        self._velocity_window.append(vel)
        if len(self._velocity_window) >= 60:  # need at least 2s of data
            window = list(self._velocity_window)
            mid = len(window) // 2
            first_mean = np.mean(window[:mid])
            second_mean = np.mean(window[mid:])
            if first_mean > 1e-6:
                raw_fatigue = max(0.0, min(1.0, (first_mean - second_mean) / first_mean))
            else:
                raw_fatigue = 0.0
            # Smooth with EMA (alpha=0.05)
            self._fatigue_ema = 0.05 * raw_fatigue + 0.95 * self._fatigue_ema
            self._fatigue_index = self._fatigue_ema

        # Look up current frame in cached analysis
        seg_info = self._frame_to_segment.get(valid_idx)

        # Movement quality per-frame assessment
        srp_joints = norm[FEATURE_INDICES, :3] if self._norm_dims == 3 else norm[FEATURE_INDICES, :2]
        cluster_id = seg_info.get("cluster") if seg_info else None

        # Get representative joints for current frame's cluster/phase
        rep_joints = None
        rep_raw = self.get_representative_joints()
        if rep_raw is not None:
            rep_joints = np.array(rep_raw, dtype=np.float64)

        # Annotate seg_info with current valid index for phase detection
        if seg_info is not None:
            seg_info["_current_valid_idx"] = valid_idx

        # Update quality tracker with current cluster's activity label
        current_label = self._activity_labels.get(cluster_id) if cluster_id is not None else None
        self._quality_tracker.set_activity_label(current_label)

        # --- Extract + smooth head landmarks for concussion tracking ---
        head_landmarks = None
        shoulder_width_px = 0.0
        head_indices = [_HEAD_NOSE, _HEAD_LEFT_EAR, _HEAD_RIGHT_EAR]
        head_vis = [float(landmarks_xyzv[idx, 3]) for idx in head_indices]
        if all(v >= _HEAD_VISIBILITY_THRESHOLD for v in head_vis):
            # Hip-center the head landmarks (same origin as body joints)
            raw_head = np.array([
                landmarks_xyzv[idx, :2].astype(np.float64) - pelvis[:2]
                for idx in head_indices
            ])  # shape (3, 2): [nose, left_ear, right_ear]
            # Apply OneEuroFilter (tighter than body: min_cutoff=1.0, beta=0.005)
            smoothed_head = np.copy(raw_head)
            for h in range(raw_head.shape[0]):
                for c in range(raw_head.shape[1]):
                    key = (h, c)
                    if key not in self._head_filters:
                        self._head_filters[key] = OneEuroFilter(
                            freq=self.fps, min_cutoff=1.0, beta=0.005,
                        )
                    smoothed_head[h, c] = self._head_filters[key](float(raw_head[h, c]))
            head_landmarks = smoothed_head

        # Shoulder width in pixels for pixel→meter scaling
        ls = landmarks_xyzv[LEFT_SHOULDER, :2].astype(np.float64)
        rs = landmarks_xyzv[RIGHT_SHOULDER, :2].astype(np.float64)
        shoulder_width_px = float(np.linalg.norm(ls - rs))

        self._quality_tracker.process_frame(
            srp_joints=srp_joints,
            cluster_id=cluster_id,
            seg_info=seg_info,
            representative_joints=rep_joints,
            fatigue_index=self._fatigue_index,
            video_time=self.frame_count / self.fps,
            raw_joints=smoothed_joints,
            joint_vis=joint_vis,
            head_landmarks=head_landmarks,
            shoulder_width_px=shoulder_width_px,
        )

        return self._build_response(frame_idx, landmarks_xyzv, seg_info)

    def needs_reanalysis(self) -> bool:
        """Check if enough new frames have accumulated to warrant re-analysis."""
        if self._frozen:
            return False
        n = len(self.features_list)
        since = n - self._last_analysis_frame

        if n < 10:
            return False
        if n < 60:
            return since >= 30
        if n < 300:
            return since >= 45
        return since >= 60

    def run_analysis(self) -> None:
        """Heavy path: re-run full segmentation pipeline on accumulated buffer.

        Uses a sliding window of MAX_ANALYSIS_WINDOW features to prevent
        unbounded growth (e.g., when a demo video loops).
        Thread-safe -- results are swapped atomically.
        """
        n_total = len(self.features_list)
        window_start = max(0, n_total - self.MAX_ANALYSIS_WINDOW)

        features = np.stack(self.features_list[window_start:], axis=0) if n_total > 0 else np.zeros((0, len(FEATURE_INDICES) * 2))
        valid_indices = self.valid_indices[window_start:]

        if features.shape[0] < 5:
            return

        segments = segment_motions(features, valid_indices, self.fps)
        segments = cluster_segments(segments, self.cluster_threshold, fps=self.fps)
        cluster_analysis = analyze_consistency(segments)

        # Build frame->segment lookup (offset by window_start for valid index lookups)
        frame_to_seg: dict[int, dict] = {}
        for seg in segments:
            for vi in range(seg["start_valid"], seg["end_valid"]):
                frame_to_seg[vi + window_start] = seg

        n_clusters = len(set(s.get("cluster", 0) for s in segments)) if segments else 0

        # Compute cluster representatives in RAW hip-centered space
        # (so they match the skeleton graph display, not SRP space)
        reps: dict[int, np.ndarray] = {}
        cluster_segs: dict[int, list] = defaultdict(list)
        for seg in segments:
            cid = seg.get("cluster")
            if cid is not None and "features" in seg:
                cluster_segs[cid].append(seg)
        raw_feats = self.raw_features_list[window_start:]
        for cid, segs in cluster_segs.items():
            # Per-cluster target length from median segment length
            seg_lengths = [s["end_valid"] - s["start_valid"] for s in segs]
            target_len = max(int(np.median(seg_lengths)), 10)  # at least 10 frames
            target_len = min(target_len, 90)  # cap at 3s at 30fps
            raw_resampled = []
            for s in segs:
                sv = s["start_valid"]
                ev = s["end_valid"]
                seg_raw = np.stack(raw_feats[sv:ev], axis=0) if ev > sv else s["features"]
                raw_resampled.append(_resample_segment(seg_raw, target_len))
            reps[cid] = np.mean(raw_resampled, axis=0)  # shape (target_len, feat_dim)

        # Per-cluster movement quality analysis
        cluster_quality: dict[int, dict] = {}
        for cid, segs in cluster_segs.items():
            if cid not in reps or len(segs) < 2:
                continue
            rep = reps[cid]  # (target_len, feat_dim)
            target_len = rep.shape[0]
            # Resample each segment's raw features to target_len
            resampled_list = []
            raw_list = []
            for s in segs:
                sv = s["start_valid"]
                ev = s["end_valid"]
                seg_raw = np.stack(raw_feats[sv:ev], axis=0) if ev > sv else s["features"]
                raw_list.append(seg_raw)
                resampled_list.append(_resample_segment(seg_raw, target_len))
            resampled_arr = np.stack(resampled_list, axis=0)  # (N, T, D)
            cluster_quality[cid] = self._quality_tracker.analyze_cluster_quality(
                cluster_id=cid,
                resampled_reps=resampled_arr,
                raw_reps=raw_list,
                fps=self.fps,
            )

        # Atomic swap under lock
        with self._lock:
            self._segments = segments
            self._cluster_analysis = cluster_analysis
            self._n_clusters = n_clusters
            self._frame_to_segment = frame_to_seg
            self._cluster_representatives = reps
            self._reps_version += 1
            self._cluster_quality = cluster_quality
            self._last_analysis_frame = n_total

        # Store new cluster mean features in VectorAI for cross-session search
        if self._vectorai_store is not None:
            for cid, segs in cluster_segs.items():
                if cid in self._stored_clusters:
                    continue
                if len(segs) < 2:
                    continue
                # Use the mean SRP feature vector (not raw hip-centered)
                seg_means = []
                for s in segs:
                    if "features" in s:
                        seg_means.append(np.mean(s["features"], axis=0))
                if not seg_means:
                    continue
                mean_feat = np.mean(seg_means, axis=0).astype(np.float32)
                activity_label = self._activity_labels.get(cid, "unknown")
                risk_score = 0.0
                cq = cluster_quality.get(cid)
                if cq and cq.get("enough_data"):
                    risk_score = cq.get("composite_fatigue", 0.0)
                try:
                    self._vectorai_store.store_motion_segment(
                        features=mean_feat,
                        activity_label=activity_label,
                        session_id=self._session_id,
                        person_id=self._person_id,
                        risk_score=risk_score,
                    )
                    self._stored_clusters.add(cid)
                except Exception as e:
                    print(f"[vectorai] WARNING: store_motion_segment in analysis failed: {e}", flush=True)

    def reset(self) -> None:
        """Reset all accumulated state (e.g., on video loop)."""
        self.landmarks_list.clear()
        self.features_list.clear()
        self.raw_features_list.clear()
        self.valid_indices.clear()
        self.frame_count = 0
        self._segments = []
        self._cluster_analysis = {}
        self._n_clusters = 0
        self._frame_to_segment = {}
        self._cluster_representatives = {}
        self._reps_version = 0
        self._last_sent_reps_version = 0
        self._last_analysis_frame = 0
        self._umap_mapper = None
        self._umap_embeddings = []
        self._umap_cluster_ids = []
        self._umap_last_fit_count = 0
        self._umap_refit_pending = False
        self._umap_refit_result = None
        self._current_srp_joints = None
        self._current_joint_vis = None
        self._norm_dims = None
        self._joint_filters.clear()
        self._activity_labels = {}
        self._pending_classification = set()
        self._frozen = False
        self._head_filters.clear()
        self.velocity_list.clear()
        self._rolling_velocity = 0.0
        self._peak_velocity = 0.0
        self._velocity_window.clear()
        self._fatigue_index = 0.0
        self._fatigue_ema = 0.0
        self._quality_tracker.reset()
        self._cluster_quality = {}

    def note_loop_boundary(self) -> None:
        """Freeze analysis when a video loop is detected.

        Called when video_time jumps backward.  The first pass produced a
        complete analysis — subsequent loops would only add noise (GPU
        inference non-determinism prevents clean cross-loop matching).
        We freeze the feature buffer so the existing segmentation/clustering
        persists, while still processing frames for live skeleton display.
        """
        # Run a final analysis if one hasn't completed yet
        if self._last_analysis_frame == 0 and len(self.features_list) >= 5:
            self.run_analysis()
        self._frozen = True
        # Force re-send of cluster_representatives on next _build_response()
        # (the flush in the WebSocket handler will call it immediately)
        with self._lock:
            self._last_sent_reps_version = 0
        # Reset temporal filters — the discontinuity would cause ringing
        self._joint_filters.clear()
        self._head_filters.clear()

    # --- UMAP embedding methods ---

    def needs_umap_refit(self) -> bool:
        """True when UMAP needs a refit: first fit at 50 features, then every 200."""
        n = len(self.features_list)
        if n < 20:
            return False
        if self._umap_mapper is None:
            return n >= 50  # first fit at 50 features
        return n - self._umap_last_fit_count >= 200  # refit every 200 (was 50)

    def run_umap_fit(self) -> dict[str, Any]:
        """Fit or incrementally extend UMAP on accumulated features."""
        import umap

        features = np.stack(self.features_list, axis=0)
        n = features.shape[0]
        n_neighbors = min(15, n - 1)

        if self._umap_mapper is None:
            # First fit: fit_transform all features
            self._umap_mapper = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="euclidean",
                n_components=3,
                random_state=42,
            )
            embedding = self._umap_mapper.fit_transform(features)
        else:
            # Subsequent: only transform new features, keep old positions stable
            new_features = features[self._umap_last_fit_count:]
            new_embedding = self._umap_mapper.transform(new_features)
            old_emb = np.array(self._umap_embeddings[:self._umap_last_fit_count])
            embedding = np.vstack([old_emb, new_embedding])

        # Build cluster ID list for each point
        cluster_ids: list[int | None] = []
        for i in range(n):
            seg = self._frame_to_segment.get(i)
            cluster_ids.append(seg.get("cluster") if seg else None)

        self._umap_embeddings = embedding.tolist()
        self._umap_cluster_ids = cluster_ids
        self._umap_last_fit_count = n

        return {
            "type": "full",
            "points": self._umap_embeddings,
            "cluster_ids": self._umap_cluster_ids,
            "current_idx": n - 1,
        }

    def run_umap_transform(self, feat: np.ndarray) -> dict[str, Any] | None:
        """Find the nearest existing embedding point in feature space.

        Instead of projecting a single point through UMAP (which is very noisy),
        find the nearest existing feature and snap the current index to it.
        """
        if self._umap_mapper is None or not self._umap_embeddings:
            return None

        try:
            # Compare against features that have embedding points
            n_emb = len(self._umap_embeddings)
            features = np.stack(self.features_list[:n_emb], axis=0)
            distances = np.linalg.norm(features - feat.reshape(1, -1), axis=1)
            nearest_idx = int(np.argmin(distances))

            return {
                "type": "current_only",
                "current_idx": nearest_idx,
            }
        except Exception:
            return None

    def absorb(self, other: "StreamingAnalyzer") -> None:
        """Merge another analyzer's data into this one.

        Concatenates landmarks, features, and valid_indices (with offset),
        resets UMAP state for full refit, and forces reanalysis.
        """
        offset = self.frame_count

        # Concatenate buffers
        self.landmarks_list.extend(other.landmarks_list)
        self.features_list.extend(other.features_list)
        self.raw_features_list.extend(other.raw_features_list)
        self.valid_indices.extend(vi + offset for vi in other.valid_indices)
        self.frame_count += other.frame_count

        # Reset UMAP state (needs full refit with merged data)
        self._umap_mapper = None
        self._umap_embeddings = []
        self._umap_cluster_ids = []
        self._umap_last_fit_count = 0

        # Force reanalysis
        self._last_analysis_frame = 0

    @staticmethod
    def _has_real_depth(landmarks_xyzv: np.ndarray) -> bool:
        """Check if landmarks have real 3D depth (non-zero Z on anchor joints)."""
        anchor_joints = [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]
        for j in anchor_joints:
            if j >= landmarks_xyzv.shape[0]:
                return False
            if abs(float(landmarks_xyzv[j, 2])) < 1e-6:
                return False
        return True

    def get_srp_joints(self) -> list[list[float]] | None:
        """Returns current SRP-normalized 14 feature joints as list of [x, y] or [x, y, z]."""
        return self._current_srp_joints

    def get_joint_visibility(self) -> list[float] | None:
        """Returns visibility (0-1) for each of the 14 feature joints."""
        return self._current_joint_vis

    def get_representative_joints(self) -> list[list[float]] | None:
        """Return representative joints for the current frame's cluster/phase.

        Maps the current frame's position within its segment to the
        corresponding frame in the cluster's mean resampled trajectory,
        then reshapes the feature vector back to (14, D) joint coordinates.
        """
        n = len(self.features_list)
        if n == 0:
            return None

        valid_idx = n - 1
        with self._lock:
            seg_info = self._frame_to_segment.get(valid_idx)
            if seg_info is None:
                return None
            cid = seg_info.get("cluster")
            if cid is None or cid not in self._cluster_representatives:
                return None
            rep = self._cluster_representatives[cid]

        # Determine phase: position within current segment
        start_vi = seg_info["start_valid"]
        end_vi = seg_info["end_valid"]
        seg_len = end_vi - start_vi
        if seg_len <= 0:
            return None
        frame_in_seg = valid_idx - start_vi
        phase = frame_in_seg / max(seg_len - 1, 1)
        rep_frame_idx = min(int(phase * (rep.shape[0] - 1)), rep.shape[0] - 1)

        feat_vec = rep[rep_frame_idx]  # shape (feat_dim,)
        feat_dim = feat_vec.shape[0]
        n_joints = len(FEATURE_INDICES)  # 14

        if feat_dim == n_joints * 2:
            joints = feat_vec.reshape(n_joints, 2)
        elif feat_dim == n_joints * 3:
            joints = feat_vec.reshape(n_joints, 3)
        else:
            return None

        return joints.tolist()

    def _build_response(
        self,
        frame_idx: int,
        landmarks_xyzv: np.ndarray | None,
        seg_info: dict | None,
    ) -> dict[str, Any]:
        """Build the per-frame JSON response."""
        n_valid = len(self.features_list)
        total_frames = self.frame_count

        # Determine phase
        calibration_end = max(30, int(total_frames * 0.2))
        if frame_idx < calibration_end or len(self._segments) == 0:
            phase = "calibrating"
        elif seg_info and seg_info.get("is_anomaly"):
            phase = "anomaly"
        else:
            phase = "normal"

        # Landmarks as normalized [0, 1] dicts for frontend rendering
        landmarks_out = None
        bbox = None
        if landmarks_xyzv is not None:
            landmarks_out = []
            xs, ys = [], []
            for i in range(min(NUM_MP_LANDMARKS, landmarks_xyzv.shape[0])):
                lm = landmarks_xyzv[i]
                vis = float(lm[3])
                landmarks_out.append({
                    "x": float(lm[0]),
                    "y": float(lm[1]),
                    "visibility": vis,
                })
                if vis > 0.3:
                    xs.append(float(lm[0]))
                    ys.append(float(lm[1]))

            if xs and ys:
                pad = 0.05
                bbox = {
                    "x1": max(0, min(xs) - pad),
                    "y1": max(0, min(ys) - pad),
                    "x2": min(1, max(xs) + pad),
                    "y2": min(1, max(ys) + pad),
                }

        # Cluster info
        cluster_id = seg_info.get("cluster") if seg_info else None
        consistency_score = seg_info.get("consistency_score", 0.0) if seg_info else None
        is_anomaly = seg_info.get("is_anomaly", False) if seg_info else False

        # Build cluster summary
        cluster_summary = {}
        with self._lock:
            for cid, cinfo in self._cluster_analysis.items():
                entry: dict[str, Any] = {
                    "count": cinfo["count"],
                    "mean_score": round(cinfo.get("mean_score", 0.0), 3),
                    "anomaly_count": len(cinfo.get("anomaly_segments", [])),
                }
                if cid in self._activity_labels:
                    entry["activity_label"] = self._activity_labels[cid]
                # Attach per-cluster quality metrics
                cq = self._cluster_quality.get(cid)
                if cq and cq.get("enough_data"):
                    entry["composite_fatigue"] = cq.get("composite_fatigue", 0.0)
                    entry["cusum_onset_rep"] = cq.get("cusum_onset_rep")
                    entry["ewma_alarming_joints"] = cq.get("ewma_alarming_joints", [])
                cluster_summary[str(cid)] = entry

        # Serialize representative trajectories when they change.
        # After freeze, re-send a few times to ensure the frontend receives them.
        cluster_reps_out: dict[str, list[list[list[float]]]] | None = None
        with self._lock:
            version_changed = self._reps_version != self._last_sent_reps_version
            if version_changed:
                cluster_reps_out = {}
                for cid, rep in self._cluster_representatives.items():
                    # rep shape: (T, feat_dim) → reshape to (T, n_joints, D)
                    n_joints = len(FEATURE_INDICES)
                    feat_dim = rep.shape[1]
                    d = feat_dim // n_joints  # 2 or 3
                    T = rep.shape[0]
                    traj = rep.reshape(T, n_joints, d)
                    cluster_reps_out[str(cid)] = [
                        [[round(float(v), 3) for v in joint] for joint in frame]
                        for frame in traj
                    ]
                self._last_sent_reps_version = self._reps_version

        # Velocity data
        vel = self.velocity_list[-1] if self.velocity_list else 0.0

        result: dict[str, Any] = {
            "frame_index": frame_idx,
            "phase": phase,
            "n_segments": len(self._segments),
            "n_clusters": self._n_clusters,
            "landmarks": landmarks_out,
            "bbox": bbox,
            "cluster_id": cluster_id,
            "consistency_score": round(consistency_score, 3) if consistency_score is not None else None,
            "is_anomaly": is_anomaly,
            "cluster_summary": cluster_summary,
            "representative_joints": self.get_representative_joints(),
            "velocity": round(vel, 4),
            "rolling_velocity": round(self._rolling_velocity, 4),
            "fatigue_index": round(self._fatigue_index, 3),
            "peak_velocity": round(self._peak_velocity, 4),
        }
        if cluster_reps_out is not None:
            result["cluster_representatives"] = cluster_reps_out

        # Movement quality metrics (per-frame biomechanics, form score, phase)
        quality = self._quality_tracker.get_frame_quality()
        if quality:
            result["quality"] = quality


        return result

    def get_clusters_needing_classification(self, min_segments: int = 3) -> list[int]:
        """Return cluster IDs that have enough segments but no activity label yet."""
        result = []
        with self._lock:
            for cid, cinfo in self._cluster_analysis.items():
                if (
                    cinfo["count"] >= min_segments
                    and cid not in self._activity_labels
                    and cid not in self._pending_classification
                ):
                    result.append(cid)
        return result

    def get_cluster_frame_indices(self, cluster_id: int) -> list[int]:
        """Return original frame indices belonging to a cluster's segments."""
        indices = []
        with self._lock:
            for seg in self._segments:
                if seg.get("cluster") == cluster_id:
                    start_vi = seg["start_valid"]
                    end_vi = seg["end_valid"]
                    for vi in range(start_vi, end_vi):
                        if vi < len(self.valid_indices):
                            indices.append(self.valid_indices[vi])
        return indices

    def get_cluster_bbox(self, cluster_id: int) -> tuple[float, float, float, float] | None:
        """Return the average normalized bbox for a cluster's segments.

        Uses the landmarks from frames in that cluster.
        Returns (x1, y1, x2, y2) in [0, 1] or None.
        """
        xs, ys = [], []
        w, h = self._img_w, self._img_h
        with self._lock:
            for seg in self._segments:
                if seg.get("cluster") != cluster_id:
                    continue
                start_vi = seg["start_valid"]
                end_vi = seg["end_valid"]
                for vi in range(start_vi, end_vi):
                    if vi < len(self.valid_indices):
                        fi = self.valid_indices[vi]
                        if fi < len(self.landmarks_list) and self.landmarks_list[fi] is not None:
                            lm = self.landmarks_list[fi]
                            for i in range(lm.shape[0]):
                                if lm[i, 3] > 0.3:
                                    xs.append(float(lm[i, 0]) / w)
                                    ys.append(float(lm[i, 1]) / h)
        if not xs or not ys:
            return None
        pad = 0.05
        return (
            max(0.0, min(xs) - pad),
            max(0.0, min(ys) - pad),
            min(1.0, max(xs) + pad),
            min(1.0, max(ys) + pad),
        )

    def mark_classification_pending(self, cluster_id: int) -> None:
        """Mark a cluster as having a pending classification request."""
        with self._lock:
            self._pending_classification.add(cluster_id)

    def set_activity_label(self, cluster_id: int, label: str) -> None:
        """Store the activity label for a cluster (thread-safe)."""
        with self._lock:
            self._activity_labels[cluster_id] = label
            self._pending_classification.discard(cluster_id)

    def get_final_summary(self) -> dict[str, Any]:
        """Return a final summary after all frames are processed."""
        summary_clusters = {}
        for cid, cinfo in self._cluster_analysis.items():
            entry: dict[str, Any] = {
                "count": cinfo["count"],
                "mean_score": round(cinfo.get("mean_score", 0.0), 3),
                "anomaly_count": len(cinfo.get("anomaly_segments", [])),
            }
            if cid in self._activity_labels:
                entry["activity_label"] = self._activity_labels[cid]
            cq = self._cluster_quality.get(cid)
            if cq and cq.get("enough_data"):
                entry["composite_fatigue"] = cq.get("composite_fatigue", 0.0)
            summary_clusters[str(cid)] = entry

        return {
            "total_frames": self.frame_count,
            "valid_frames": len(self.features_list),
            "n_segments": len(self._segments),
            "n_clusters": self._n_clusters,
            "cluster_summary": summary_clusters,
        }
