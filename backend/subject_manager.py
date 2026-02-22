"""Manage mapping from YOLO track_ids to StreamingAnalyzer instances."""

from __future__ import annotations

import logging
from typing import Any

from streaming_analyzer import StreamingAnalyzer

logger = logging.getLogger(__name__)


class SubjectManager:
    """Maps track_ids to StreamingAnalyzer instances with lifecycle management."""

    def __init__(self, fps: float = 30.0, cluster_threshold: float = 2.0, risk_modifiers: Any = None, vectorai_store: Any = None):
        self.fps = fps
        self.cluster_threshold = cluster_threshold
        self.risk_modifiers = risk_modifiers
        self.vectorai_store = vectorai_store
        self.analyzers: dict[int, StreamingAnalyzer] = {}
        self._track_to_label: dict[int, str] = {}
        self._next_label_idx = 1
        self._loop_detected: bool = False  # when True, new analyzers start frozen

        # Per-subject stability tracking
        self._subject_track_history: dict[int, list[int]] = {}  # subject_id -> recent track_ids
        self._last_seen_frame: dict[int, int] = {}  # subject_id -> last frame seen
        self._track_change_count: dict[int, int] = {}  # subject_id -> track changes in window

    def reset(self) -> None:
        """Reset all state for a new session/video loop."""
        self.analyzers.clear()
        self._track_to_label.clear()
        self._next_label_idx = 1
        self._loop_detected = False
        self._subject_track_history.clear()
        self._last_seen_frame.clear()
        self._track_change_count.clear()

    def get_or_create_analyzer(self, track_id: int) -> StreamingAnalyzer:
        """Get existing analyzer or create new one for a track_id."""
        if track_id not in self.analyzers:
            analyzer = StreamingAnalyzer(
                fps=self.fps,
                cluster_threshold=self.cluster_threshold,
                risk_modifiers=self.risk_modifiers,
                vectorai_store=self.vectorai_store,
            )
            self.analyzers[track_id] = analyzer
            self._track_to_label[track_id] = f"S{self._next_label_idx}"
            self._next_label_idx += 1
        return self.analyzers[track_id]

    def record_subject_track(self, subject_id: int, track_id: int, frame_idx: int) -> None:
        """Record a subject's track_id for stability tracking.

        Logs warnings when a subject's track_id changes frequently, which
        indicates identity instability.
        """
        self._last_seen_frame[subject_id] = frame_idx

        history = self._subject_track_history.setdefault(subject_id, [])
        if not history or history[-1] != track_id:
            if history:
                logger.warning(
                    "Subject %d track changed: %d -> %d (frame %d)",
                    subject_id, history[-1], track_id, frame_idx,
                )
                self._track_change_count[subject_id] = self._track_change_count.get(subject_id, 0) + 1
            history.append(track_id)
            # Keep only last 30 entries
            if len(history) > 30:
                history[:] = history[-30:]

        # Warn if unstable (>3 changes in recent history)
        changes = self._track_change_count.get(subject_id, 0)
        if changes > 3 and changes % 5 == 0:
            logger.warning(
                "Subject %d is UNSTABLE: %d track changes so far",
                subject_id, changes,
            )

    def merge_short_lived_analyzer(self, keep_id: int, merge_id: int, max_frames: int = 30) -> bool:
        """Merge a short-lived analyzer into the correct one.

        If merge_id's analyzer has fewer than max_frames, discard its data
        (it was likely an incorrect brief re-ID). Returns True if merged.
        """
        if merge_id not in self.analyzers or keep_id not in self.analyzers:
            return False
        if self.analyzers[merge_id].frame_count < max_frames:
            logger.info(
                "Discarding short-lived analyzer for subject %d (%d frames) in favor of %d",
                merge_id, self.analyzers[merge_id].frame_count, keep_id,
            )
            del self.analyzers[merge_id]
            self._last_seen_frame.pop(merge_id, None)
            self._subject_track_history.pop(merge_id, None)
            self._track_change_count.pop(merge_id, None)
            return True
        return False

    def note_loop(self) -> None:
        """Mark that a video loop was detected.

        Freezes all existing analyzers (preserving their analysis results)
        so subsequent loops don't re-accumulate duplicate features.
        """
        for analyzer in self.analyzers.values():
            analyzer.note_loop_boundary()

    def get_label(self, track_id: int) -> str:
        """Get the human-readable label for a track_id."""
        return self._track_to_label.get(track_id, f"S{track_id}")

    def cleanup_stale(self, current_frame: int, max_missing: int = 90) -> list[int]:
        """Remove analyzers for subjects not seen in max_missing frames.

        Returns list of removed track_ids.
        """
        stale = []
        for track_id, analyzer in self.analyzers.items():
            # Never clean up frozen analyzers — they hold cached analysis
            if analyzer._frozen:
                continue
            if current_frame - analyzer.last_seen_frame > max_missing:
                stale.append(track_id)
        for track_id in stale:
            del self.analyzers[track_id]
            self._last_seen_frame.pop(track_id, None)
            self._subject_track_history.pop(track_id, None)
            self._track_change_count.pop(track_id, None)
            # Keep label mapping so re-appearing IDs don't reuse labels
        return stale

    def absorb_analyzer(self, target_id: int, source_id: int) -> None:
        """Merge source analyzer into target, then delete source.

        Used when IdentityResolver discovers two track_ids belong to the same subject.
        """
        if target_id not in self.analyzers or source_id not in self.analyzers:
            return
        self.analyzers[target_id].absorb(self.analyzers[source_id])
        del self.analyzers[source_id]
        self._last_seen_frame.pop(source_id, None)
        self._subject_track_history.pop(source_id, None)
        self._track_change_count.pop(source_id, None)

    def merge_subject(self, from_id: int, to_id: int) -> None:
        """Merge one subject into another by absorbing analyzers.

        If either subject doesn't have an analyzer, this is a no-op.
        After merge, from_id's analyzer is removed.

        Args:
            from_id: Subject ID to merge from (will be removed).
            to_id: Subject ID to merge into (will absorb features).
        """
        if from_id not in self.analyzers or to_id not in self.analyzers:
            return
        self.analyzers[to_id].absorb(self.analyzers[from_id])
        del self.analyzers[from_id]
        self._last_seen_frame.pop(from_id, None)
        self._subject_track_history.pop(from_id, None)
        self._track_change_count.pop(from_id, None)

    def get_active_track_ids(self, current_frame: int = -1, hysteresis_frames: int = 45) -> list[int]:
        """Return list of active track IDs, including recently-seen subjects.

        When current_frame is provided, subjects seen within hysteresis_frames
        (~1.5s at 30fps) are included even if not in the current frame.
        This prevents the frontend from deleting subjects during brief occlusions.
        """
        if current_frame < 0:
            return list(self.analyzers.keys())

        ids = set()
        for sid in self.analyzers:
            last = self._last_seen_frame.get(sid, 0)
            if current_frame - last <= hysteresis_frames:
                ids.add(sid)
        return list(ids)

    def get_all_analyzers(self) -> list[StreamingAnalyzer]:
        """Return all active StreamingAnalyzer instances."""
        return list(self.analyzers.values())
