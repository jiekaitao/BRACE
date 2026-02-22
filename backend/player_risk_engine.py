"""Player risk engine for real-time injury risk tracking during games.

Tracks per-player injury events, workload, and determines risk status
(GREEN/YELLOW/RED) with pull recommendations for coaching staff.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class RiskStatus(enum.IntEnum):
    """Traffic-light risk status for a player."""
    GREEN = 0
    YELLOW = 1
    RED = 2


@dataclass
class InjuryEvent:
    """A single detected injury risk event."""
    risk_name: str        # e.g. "acl_valgus", "hip_drop"
    severity: str         # "medium" or "high"
    joint: str            # "left_knee", "pelvis", etc.
    timestamp: float      # video time in seconds
    frame_index: int
    description: str = "" # human-readable description

    @property
    def key(self) -> str:
        """Consolidation key: same risk at same joint."""
        return f"{self.risk_name}:{self.joint}"


@dataclass
class PlayerWorkload:
    """Accumulated workload metrics for a player."""
    total_frames: int = 0
    active_seconds: float = 0.0
    high_intensity_seconds: float = 0.0
    rest_seconds: float = 0.0

    @property
    def intensity_ratio(self) -> float:
        """Fraction of active time spent at high intensity."""
        if self.active_seconds <= 0:
            return 0.0
        return self.high_intensity_seconds / self.active_seconds

    @property
    def fatigue_estimate(self) -> float:
        """Estimated fatigue level (0.0 = fresh, 1.0 = exhausted).

        Based on intensity ratio and total active time.
        Longer high-intensity periods increase fatigue faster.
        """
        if self.active_seconds <= 0:
            return 0.0
        # Base fatigue from intensity
        base = self.intensity_ratio
        # Time factor: ramps up over 10 minutes (600s) of active time
        time_factor = min(1.0, self.active_seconds / 600.0)
        # Rest recovery: reduce fatigue based on rest proportion
        total_time = self.active_seconds + self.rest_seconds
        rest_factor = 1.0
        if total_time > 0:
            rest_ratio = self.rest_seconds / total_time
            rest_factor = max(0.3, 1.0 - rest_ratio * 0.5)
        return min(1.0, base * 0.6 + time_factor * 0.4) * rest_factor


@dataclass
class PlayerRiskState:
    """Current risk state for a player."""
    status: RiskStatus = RiskStatus.GREEN
    events: list[InjuryEvent] = field(default_factory=list)
    workload: PlayerWorkload = field(default_factory=PlayerWorkload)
    pull_recommended: bool = False
    pull_reason: str = ""


class PlayerRiskEngine:
    """Tracks injury risks and workload per player, determines risk status.

    Processes per-frame quality data from MovementQualityTracker and
    maintains a running state of injury events, workload metrics,
    and risk status (GREEN/YELLOW/RED).
    """

    def __init__(
        self,
        yellow_event_count: int = 3,
        red_event_count: int = 6,
        consolidation_window_sec: float = 5.0,
        fatigue_yellow_threshold: float = 0.6,
        fatigue_red_threshold: float = 0.8,
        fps: float = 30.0,
    ):
        self._yellow_count = yellow_event_count
        self._red_count = red_event_count
        self._consolidation_window = consolidation_window_sec
        self._fatigue_yellow = fatigue_yellow_threshold
        self._fatigue_red = fatigue_red_threshold
        self._fps = fps

        self._state = PlayerRiskState()
        self._raw_events: list[InjuryEvent] = []  # before consolidation

    def process_frame(
        self,
        quality: dict[str, Any] | None,
        frame_index: int,
        video_time: float = 0.0,
        velocity: float = 0.0,
    ) -> PlayerRiskState:
        """Process a single frame's quality data and update risk state.

        Args:
            quality: The 'quality' dict from MovementQualityTracker containing
                     'injury_risks' list and biomechanical metrics.
            frame_index: Current frame number.
            video_time: Current video timestamp in seconds.
            velocity: Current movement velocity (for workload tracking).

        Returns:
            Updated PlayerRiskState.
        """
        # Update workload
        self._state.workload.total_frames += 1
        frame_sec = 1.0 / self._fps if self._fps > 0 else 1.0 / 30.0

        # Classify intensity based on velocity
        HIGH_VELOCITY_THRESHOLD = 0.3  # normalized velocity threshold
        REST_VELOCITY_THRESHOLD = 0.05

        if velocity >= HIGH_VELOCITY_THRESHOLD:
            self._state.workload.active_seconds += frame_sec
            self._state.workload.high_intensity_seconds += frame_sec
        elif velocity >= REST_VELOCITY_THRESHOLD:
            self._state.workload.active_seconds += frame_sec
        else:
            self._state.workload.rest_seconds += frame_sec

        # Extract injury risks from quality data
        if quality and "injury_risks" in quality:
            for risk in quality["injury_risks"]:
                event = InjuryEvent(
                    risk_name=risk.get("risk_name", "unknown"),
                    severity=risk.get("severity", "medium"),
                    joint=risk.get("joint", "unknown"),
                    timestamp=video_time,
                    frame_index=frame_index,
                    description=risk.get("description", ""),
                )
                self._raw_events.append(event)

        # Consolidate and determine status
        self._state.events = self._consolidate_injury_risks(self._raw_events)
        self._state.status = self._determine_status(self._state.events)
        self._state.pull_recommended, self._state.pull_reason = (
            self._check_pull_recommendation()
        )

        return self._state

    def _consolidate_injury_risks(
        self, events: list[InjuryEvent]
    ) -> list[InjuryEvent]:
        """Consolidate events with same risk_name+joint within time window.

        Multiple detections of the same risk at the same joint within
        consolidation_window_sec are treated as a single event (the first one).
        """
        if not events:
            return []

        consolidated: list[InjuryEvent] = []
        # Track last event time per consolidation key
        last_time: dict[str, float] = {}

        for event in events:
            key = event.key
            if key in last_time:
                if event.timestamp - last_time[key] <= self._consolidation_window:
                    # Within window — skip (consolidated with previous)
                    continue
            # New event or beyond window
            consolidated.append(event)
            last_time[key] = event.timestamp

        return consolidated

    def _determine_status(
        self, consolidated_events: list[InjuryEvent]
    ) -> RiskStatus:
        """Determine risk status based on consolidated event count and severity."""
        n = len(consolidated_events)

        # Any high severity event is at least YELLOW
        has_high = any(e.severity == "high" for e in consolidated_events)

        if n >= self._red_count:
            return RiskStatus.RED
        elif n >= self._yellow_count or has_high:
            return RiskStatus.YELLOW

        # Check fatigue-based status
        fatigue = self._state.workload.fatigue_estimate
        if fatigue >= self._fatigue_red:
            return RiskStatus.RED
        elif fatigue >= self._fatigue_yellow:
            return RiskStatus.YELLOW

        return RiskStatus.GREEN

    def _check_pull_recommendation(self) -> tuple[bool, str]:
        """Check if player should be pulled based on risk + fatigue."""
        reasons: list[str] = []

        if self._state.status == RiskStatus.RED:
            n = len(self._state.events)
            reasons.append(f"RED status: {n} injury risk events detected")

        fatigue = self._state.workload.fatigue_estimate
        if fatigue >= self._fatigue_red:
            reasons.append(f"High fatigue ({fatigue:.0%})")
        elif (
            fatigue >= self._fatigue_yellow
            and self._state.status >= RiskStatus.YELLOW
        ):
            reasons.append(
                f"Elevated fatigue ({fatigue:.0%}) combined with YELLOW risk status"
            )

        # High severity events
        high_events = [e for e in self._state.events if e.severity == "high"]
        if len(high_events) >= 2:
            reasons.append(f"{len(high_events)} high-severity risk events")

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    def get_player_summary(self) -> dict[str, Any]:
        """Return a summary dict of the player's current risk state."""
        return {
            "status": self._state.status.name,
            "event_count": len(self._state.events),
            "events": [
                {
                    "risk_name": e.risk_name,
                    "severity": e.severity,
                    "joint": e.joint,
                    "timestamp": e.timestamp,
                    "frame_index": e.frame_index,
                    "description": e.description,
                }
                for e in self._state.events
            ],
            "workload": {
                "total_frames": self._state.workload.total_frames,
                "active_seconds": round(self._state.workload.active_seconds, 1),
                "high_intensity_seconds": round(
                    self._state.workload.high_intensity_seconds, 1
                ),
                "rest_seconds": round(self._state.workload.rest_seconds, 1),
                "intensity_ratio": round(self._state.workload.intensity_ratio, 3),
                "fatigue_estimate": round(
                    self._state.workload.fatigue_estimate, 3
                ),
            },
            "pull_recommended": self._state.pull_recommended,
            "pull_reason": self._state.pull_reason,
        }

    @property
    def state(self) -> PlayerRiskState:
        """Current risk state."""
        return self._state

    def reset(self) -> None:
        """Reset all state for a new game/session."""
        self._state = PlayerRiskState()
        self._raw_events.clear()
