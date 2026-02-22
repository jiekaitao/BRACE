"""Manage mapping from YOLO track_ids to StreamingAnalyzer instances."""

from __future__ import annotations

from typing import Any

from streaming_analyzer import StreamingAnalyzer


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

    def reset(self) -> None:
        """Reset all state for a new session/video loop."""
        self.analyzers.clear()
        self._track_to_label.clear()
        self._next_label_idx = 1
        self._loop_detected = False

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

    def get_active_track_ids(self) -> list[int]:
        """Return list of currently active track IDs."""
        return list(self.analyzers.keys())

    def get_all_analyzers(self) -> list[StreamingAnalyzer]:
        """Return all active StreamingAnalyzer instances."""
        return list(self.analyzers.values())
