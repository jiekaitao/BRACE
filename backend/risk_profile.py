"""Personalized risk threshold scaling based on injury profiles."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RiskModifiers:
    """Per-metric threshold multipliers. Lower = more sensitive."""

    fppa_scale: float = 1.0
    hip_drop_scale: float = 1.0
    trunk_lean_scale: float = 1.0
    asymmetry_scale: float = 1.0
    angular_velocity_scale: float = 1.0
    monitor_joints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "RiskModifiers":
        if not d:
            return RiskModifiers()
        return RiskModifiers(
            fppa_scale=d.get("fppa_scale", 1.0),
            hip_drop_scale=d.get("hip_drop_scale", 1.0),
            trunk_lean_scale=d.get("trunk_lean_scale", 1.0),
            asymmetry_scale=d.get("asymmetry_scale", 1.0),
            angular_velocity_scale=d.get("angular_velocity_scale", 1.0),
            monitor_joints=d.get("monitor_joints", []),
        )


# Default thresholds (from movement_quality.py evaluate_injury_risks)
DEFAULT_THRESHOLDS = {
    "fppa": {"medium": 15.0, "high": 25.0},
    "hip_drop": {"medium": 8.0, "high": 12.0},
    "trunk_lean": {"medium": 15.0, "high": 25.0},
    "asymmetry": {"medium": 15.0, "high": 25.0},
    "angular_velocity": {"medium": 500.0},  # single threshold
}


def apply_modifiers(
    modifiers: RiskModifiers | None,
) -> dict[str, dict[str, float]]:
    """Apply risk modifiers to default thresholds.

    Returns modified thresholds dict with same structure as DEFAULT_THRESHOLDS.
    """
    if modifiers is None:
        return {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}

    result = {}
    for metric, thresholds in DEFAULT_THRESHOLDS.items():
        scale = 1.0
        if metric == "fppa":
            scale = modifiers.fppa_scale
        elif metric == "hip_drop":
            scale = modifiers.hip_drop_scale
        elif metric == "trunk_lean":
            scale = modifiers.trunk_lean_scale
        elif metric == "asymmetry":
            scale = modifiers.asymmetry_scale
        elif metric == "angular_velocity":
            scale = modifiers.angular_velocity_scale

        result[metric] = {k: v * scale for k, v in thresholds.items()}

    return result
