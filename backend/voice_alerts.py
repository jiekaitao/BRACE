"""Voice alert generation for real-time injury risk coaching."""

import time
from typing import Any

# Risk type to spoken descriptions
RISK_DESCRIPTIONS: dict[str, str] = {
    "acl_valgus": "knee valgus detected",
    "hip_drop": "excessive hip drop",
    "trunk_lean": "trunk leaning too far",
    "asymmetry": "movement asymmetry",
    "angular_velocity_spike": "joint moving too fast",
}

JOINT_SPOKEN: dict[str, str] = {
    "left_knee": "left knee",
    "right_knee": "right knee",
    "pelvis": "hips",
    "trunk": "trunk",
    "bilateral": "both sides",
    "left_elbow": "left elbow",
    "right_elbow": "right elbow",
}


class VoiceAlertGenerator:
    """Generates text alerts from injury risk evaluations with cooldown and dedup."""

    def __init__(self, cooldown_sec: float = 8.0, sustained_threshold_sec: float = 3.0):
        self.cooldown_sec = cooldown_sec
        self.sustained_threshold_sec = sustained_threshold_sec
        self._last_alert_time: float = -cooldown_sec
        self._last_alert_text: str = ""
        # Track sustained medium risks: (risk_type, joint) -> first_seen_time
        self._sustained_tracker: dict[tuple[str, str], float] = {}

    def generate_alert_text(
        self,
        injury_risks: list[dict[str, Any]],
        active_guideline: dict[str, Any] | None = None,
        current_time: float | None = None,
    ) -> str | None:
        """Generate alert text from current injury risks.

        Returns None if:
        - No actionable risks
        - Within cooldown period
        - Same alert was just spoken
        """
        now = current_time if current_time is not None else time.monotonic()

        if not injury_risks:
            self._sustained_tracker.clear()
            return None

        # Check cooldown
        if now - self._last_alert_time < self.cooldown_sec:
            return None

        # Find highest priority risk
        high_risks = [r for r in injury_risks if r.get("severity") == "high"]
        medium_risks = [r for r in injury_risks if r.get("severity") == "medium"]

        alert_risk = None

        if high_risks:
            # High severity: alert immediately
            alert_risk = high_risks[0]
        elif medium_risks:
            # Medium severity: only alert if sustained
            for risk in medium_risks:
                key = (risk.get("risk", ""), risk.get("joint", ""))
                if key not in self._sustained_tracker:
                    self._sustained_tracker[key] = now
                elif now - self._sustained_tracker[key] >= self.sustained_threshold_sec:
                    alert_risk = risk
                    break

        if alert_risk is None:
            return None

        # Build alert text
        risk_type = alert_risk.get("risk", "unknown")
        joint = alert_risk.get("joint", "")
        severity = alert_risk.get("severity", "medium")

        desc = RISK_DESCRIPTIONS.get(risk_type, risk_type.replace("_", " "))
        joint_name = JOINT_SPOKEN.get(joint, joint)

        if severity == "high":
            text = f"Warning: {desc} at your {joint_name}. Please correct your form."
        else:
            text = f"Watch your {joint_name}: {desc}."

        # Add guideline context
        if active_guideline and active_guideline.get("display_name"):
            exercise = active_guideline["display_name"]
            text = f"During {exercise}: {text}"

        # Dedup: don't repeat the exact same alert
        if text == self._last_alert_text:
            return None

        self._last_alert_time = now
        self._last_alert_text = text

        # Reset sustained tracker for alerted risks
        key = (risk_type, joint)
        self._sustained_tracker.pop(key, None)

        return text
