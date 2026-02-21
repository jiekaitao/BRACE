"""Player risk engine for basketball game analysis.

Combines per-frame biomechanical signals (injury risks, fatigue, form score)
into actionable GREEN/YELLOW/RED status and pull-from-game recommendations.
"""

from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


class RiskStatus(enum.Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class InjuryEvent:
    """A consolidated sustained risk period."""
    joint: str
    risk_type: str
    severity: str  # "medium" or "high"
    onset_frame: int
    onset_time: float
    duration_frames: int = 0
    duration_sec: float = 0.0
    max_value: float = 0.0
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "risk_type": self.risk_type,
            "severity": self.severity,
            "onset_frame": self.onset_frame,
            "onset_time": round(self.onset_time, 2),
            "duration_frames": self.duration_frames,
            "duration_sec": round(self.duration_sec, 2),
            "max_value": round(self.max_value, 2),
            "active": self.active,
        }


@dataclass
class PlayerWorkload:
    """Cumulative workload tracking for a player."""
    total_frames: int = 0
    active_frames: int = 0
    high_effort_frames: int = 0
    activity_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total = max(self.total_frames, 1)
        return {
            "total_frames": self.total_frames,
            "active_frames": self.active_frames,
            "high_effort_frames": self.high_effort_frames,
            "high_effort_pct": round(self.high_effort_frames / total * 100, 1),
            "activity_distribution": dict(self.activity_distribution),
        }


@dataclass
class PlayerRiskState:
    """Per-player risk state."""
    status: RiskStatus = RiskStatus.GREEN
    injury_events: list[InjuryEvent] = field(default_factory=list)
    workload: PlayerWorkload = field(default_factory=PlayerWorkload)
    status_history: list[dict[str, Any]] = field(default_factory=list)
    pull_recommended: bool = False
    pull_reasons: list[str] = field(default_factory=list)

    # Sliding windows for form/fatigue (last N frames)
    form_window: list[float] = field(default_factory=list)
    fatigue_window: list[float] = field(default_factory=list)

    # Tracking for risk consolidation: key -> (consecutive_frames, gap_frames)
    risk_streaks: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Tracking for RED duration
    red_start_frame: int | None = None
    red_continuous_frames: int = 0


# Activity-specific fatigue thresholds: (yellow_threshold, red_threshold)
_FATIGUE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "basketball_landing": (0.45, 0.65),
    "basketball_cutting": (0.50, 0.70),
    "basketball_shooting": (0.55, 0.75),
    "basketball_defense": (0.55, 0.75),
    "basketball_dribbling": (0.55, 0.75),
    "jump": (0.45, 0.65),
    "running": (0.50, 0.70),
}
_DEFAULT_FATIGUE_THRESHOLD = (0.50, 0.70)

# Form score window size (frames)
_FORM_WINDOW_SIZE = 90  # ~3 seconds at 30fps
_FATIGUE_WINDOW_SIZE = 90


class PlayerRiskEngine:
    """Combines biomechanical signals into risk status and pull recommendations.

    Args:
        fps: Video frame rate.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._states: dict[int, PlayerRiskState] = {}

        # Frame thresholds derived from fps
        self._high_sustain_frames = int(2.0 * fps)    # 2s for high severity
        self._medium_sustain_frames = int(5.0 * fps)   # 5s for medium severity
        self._gap_tolerance_frames = int(1.0 * fps)    # 1s gap tolerance
        self._close_after_frames = int(2.0 * fps)      # close event after 2s absent
        self._red_pull_frames = int(30.0 * fps)         # 30s continuous RED

    def _get_state(self, subject_id: int) -> PlayerRiskState:
        if subject_id not in self._states:
            self._states[subject_id] = PlayerRiskState()
        return self._states[subject_id]

    def process_frame(
        self,
        subject_id: int,
        frame_idx: int,
        video_time: float,
        quality: dict[str, Any] | None,
        cluster_quality: dict[str, Any] | None,
        activity_profile_name: str | None = None,
    ) -> None:
        """Process one frame of data for a player.

        Args:
            subject_id: Player identifier.
            frame_idx: Current frame index.
            video_time: Current video time in seconds.
            quality: Per-frame quality dict from MovementQualityTracker.get_frame_quality().
            cluster_quality: Per-cluster quality dict from analyze_cluster_quality().
            activity_profile_name: Name of the matched movement profile (e.g. "basketball_landing").
        """
        state = self._get_state(subject_id)

        # Extract signals
        form_score = None
        fatigue_score = None
        injury_risks: list[dict[str, Any]] = []

        if quality:
            form_score = quality.get("form_score")
            injury_risks = quality.get("injury_risks", [])

        if cluster_quality:
            fatigue_score = cluster_quality.get("composite_fatigue")

        # Update sliding windows
        if form_score is not None:
            state.form_window.append(form_score)
            if len(state.form_window) > _FORM_WINDOW_SIZE:
                state.form_window = state.form_window[-_FORM_WINDOW_SIZE:]

        if fatigue_score is not None:
            state.fatigue_window.append(fatigue_score)
            if len(state.fatigue_window) > _FATIGUE_WINDOW_SIZE:
                state.fatigue_window = state.fatigue_window[-_FATIGUE_WINDOW_SIZE:]

        # Update workload
        self._update_workload(state, fatigue_score, activity_profile_name)

        # Consolidate injury risks into events
        self._consolidate_injury_risks(state, injury_risks, frame_idx, video_time)

        # Determine status
        prev_status = state.status
        state.status = self._determine_status(
            state, fatigue_score, activity_profile_name
        )

        # Track RED duration
        if state.status == RiskStatus.RED:
            if prev_status != RiskStatus.RED:
                state.red_start_frame = frame_idx
                state.red_continuous_frames = 1
            else:
                state.red_continuous_frames += 1
        else:
            state.red_start_frame = None
            state.red_continuous_frames = 0

        # Check pull recommendation
        self._check_pull_recommendation(state, frame_idx)

        # Record status history (sampled at ~1Hz)
        if state.workload.total_frames % int(self.fps) == 0:
            state.status_history.append({
                "time": round(video_time, 2),
                "frame": frame_idx,
                "status": state.status.value,
                "fatigue": round(fatigue_score, 3) if fatigue_score is not None else None,
                "form": round(form_score, 1) if form_score is not None else None,
            })

    def _consolidate_injury_risks(
        self,
        state: PlayerRiskState,
        frame_risks: list[dict[str, Any]],
        frame_idx: int,
        video_time: float,
    ) -> None:
        """Accumulate transient per-frame risks into consolidated InjuryEvents."""
        # Build set of active risk keys this frame
        active_keys: set[str] = set()
        risk_values: dict[str, float] = {}
        risk_info: dict[str, dict[str, Any]] = {}

        for risk in frame_risks:
            key = f"{risk['joint']}:{risk['risk']}:{risk['severity']}"
            active_keys.add(key)
            risk_values[key] = risk.get("value", 0.0)
            risk_info[key] = risk

        # Update streaks
        for key in list(state.risk_streaks.keys()):
            streak = state.risk_streaks[key]
            if key in active_keys:
                streak["consecutive"] += 1
                streak["gap"] = 0
                streak["last_frame"] = frame_idx
                streak["max_value"] = max(streak["max_value"], risk_values.get(key, 0.0))
            else:
                streak["gap"] += 1

        # Add new streaks
        for key in active_keys:
            if key not in state.risk_streaks:
                state.risk_streaks[key] = {
                    "consecutive": 1,
                    "gap": 0,
                    "start_frame": frame_idx,
                    "start_time": video_time,
                    "last_frame": frame_idx,
                    "max_value": risk_values.get(key, 0.0),
                }

        # Check for new events and close old ones
        for key, streak in list(state.risk_streaks.items()):
            parts = key.split(":")
            severity = parts[2] if len(parts) >= 3 else "medium"

            # Determine sustain threshold
            if severity == "high":
                threshold = self._high_sustain_frames
            else:
                threshold = self._medium_sustain_frames

            total_active = streak["consecutive"]

            # Create event if sustained long enough
            if total_active >= threshold:
                # Check if there's already an active event for this key
                existing = None
                for ev in state.injury_events:
                    if (ev.active and ev.joint == parts[0] and
                            ev.risk_type == parts[1] and ev.severity == severity):
                        existing = ev
                        break

                if existing is None:
                    event = InjuryEvent(
                        joint=parts[0],
                        risk_type=parts[1],
                        severity=severity,
                        onset_frame=streak["start_frame"],
                        onset_time=streak["start_time"],
                        duration_frames=total_active,
                        duration_sec=total_active / self.fps,
                        max_value=streak["max_value"],
                        active=True,
                    )
                    state.injury_events.append(event)
                else:
                    # Update existing event duration
                    existing.duration_frames = streak["last_frame"] - existing.onset_frame
                    existing.duration_sec = existing.duration_frames / self.fps
                    existing.max_value = max(existing.max_value, streak["max_value"])

            # Close events when gap exceeds tolerance
            if streak["gap"] > self._close_after_frames:
                for ev in state.injury_events:
                    if (ev.active and ev.joint == parts[0] and
                            ev.risk_type == parts[1] and ev.severity == severity):
                        ev.active = False
                # Remove streak
                del state.risk_streaks[key]

    def _determine_status(
        self,
        state: PlayerRiskState,
        fatigue_score: float | None,
        activity_profile_name: str | None,
    ) -> RiskStatus:
        """Three-factor worst-case-wins status determination."""
        factors: list[RiskStatus] = []

        # Factor 1: Fatigue
        factors.append(self._fatigue_factor(fatigue_score, activity_profile_name))

        # Factor 2: Injury events
        factors.append(self._injury_factor(state))

        # Factor 3: Form score
        factors.append(self._form_factor(state))

        # Worst-case wins
        if RiskStatus.RED in factors:
            return RiskStatus.RED
        if RiskStatus.YELLOW in factors:
            return RiskStatus.YELLOW
        return RiskStatus.GREEN

    def _fatigue_factor(
        self, fatigue_score: float | None, activity_profile_name: str | None
    ) -> RiskStatus:
        if fatigue_score is None:
            return RiskStatus.GREEN

        yellow_th, red_th = _FATIGUE_THRESHOLDS.get(
            activity_profile_name or "", _DEFAULT_FATIGUE_THRESHOLD
        )

        if fatigue_score >= red_th:
            return RiskStatus.RED
        if fatigue_score >= yellow_th:
            return RiskStatus.YELLOW
        return RiskStatus.GREEN

    def _injury_factor(self, state: PlayerRiskState) -> RiskStatus:
        active_events = [e for e in state.injury_events if e.active]
        active_high = [e for e in active_events if e.severity == "high"]

        if len(active_high) >= 3:
            return RiskStatus.RED
        if len(active_events) >= 2:
            return RiskStatus.YELLOW
        return RiskStatus.GREEN

    def _form_factor(self, state: PlayerRiskState) -> RiskStatus:
        if len(state.form_window) < 30:
            return RiskStatus.GREEN

        current_form = state.form_window[-1]
        # Check if form is declining: compare recent avg to earlier avg
        mid = len(state.form_window) // 2
        early_avg = sum(state.form_window[:mid]) / max(mid, 1)
        recent_avg = sum(state.form_window[mid:]) / max(len(state.form_window) - mid, 1)
        declining = recent_avg < early_avg

        if current_form < 40 and declining:
            return RiskStatus.RED
        if current_form < 65 and declining:
            return RiskStatus.YELLOW
        return RiskStatus.GREEN

    def _check_pull_recommendation(self, state: PlayerRiskState, frame_idx: int) -> None:
        """Check if player should be pulled from game."""
        reasons: list[str] = []

        # Condition 1: RED for 30+ continuous seconds
        if state.red_continuous_frames >= self._red_pull_frames:
            reasons.append(
                f"RED status for {state.red_continuous_frames / self.fps:.0f}s continuously"
            )

        # Condition 2: RED + active high-severity injury event
        if state.status == RiskStatus.RED:
            active_high = [e for e in state.injury_events if e.active and e.severity == "high"]
            if active_high:
                joints = ", ".join(set(e.joint for e in active_high))
                reasons.append(f"RED status with active high-severity injury ({joints})")

        state.pull_recommended = len(reasons) > 0
        state.pull_reasons = reasons

    def _update_workload(
        self,
        state: PlayerRiskState,
        fatigue_score: float | None,
        activity_profile_name: str | None,
    ) -> None:
        state.workload.total_frames += 1
        state.workload.active_frames += 1

        if fatigue_score is not None and fatigue_score > 0.5:
            state.workload.high_effort_frames += 1

        if activity_profile_name:
            state.workload.activity_distribution[activity_profile_name] = (
                state.workload.activity_distribution.get(activity_profile_name, 0) + 1
            )

    def get_player_summary(self, subject_id: int) -> dict[str, Any]:
        """Get serializable summary for a player."""
        state = self._get_state(subject_id)
        return {
            "risk_status": state.status.value,
            "injury_events": [e.to_dict() for e in state.injury_events],
            "workload": state.workload.to_dict(),
            "risk_history": state.status_history,
            "pull_recommended": state.pull_recommended,
            "pull_reasons": state.pull_reasons,
        }

    def get_status(self, subject_id: int) -> str:
        """Get current risk status string for a player."""
        state = self._get_state(subject_id)
        return state.status.value

    def get_all_statuses(self) -> dict[int, dict[str, Any]]:
        """Get all players' current status for broadcast."""
        result = {}
        for sid, state in self._states.items():
            result[sid] = {
                "risk_status": state.status.value,
                "pull_recommended": state.pull_recommended,
            }
        return result
