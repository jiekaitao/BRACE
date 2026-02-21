"""Movement-specific biomechanical guidelines for injury risk thresholds.

Provides clinically informed thresholds for different exercise types.
Gemini activity labels are matched to movement profiles, each with
specific risk thresholds and coaching cues.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskThreshold:
    """A single risk threshold for a specific metric."""
    metric: str          # "fppa", "hip_drop", "trunk_lean", "asymmetry", "angular_velocity"
    medium: float        # threshold for medium severity
    high: float          # threshold for high severity
    joint: str           # "left_knee", "pelvis", "trunk", "bilateral"
    risk_name: str       # "acl_valgus", "hip_drop", "trunk_lean", "asymmetry", "angular_velocity_spike"
    enabled: bool = True # False to suppress this risk entirely for this movement


@dataclass
class MovementProfile:
    """A movement-specific set of thresholds and coaching cues."""
    name: str                                # "squat", "lunge", "running", etc.
    display_name: str                        # "Squat", "Forward Lunge", etc.
    keywords: list[str] = field(default_factory=list)  # Gemini label substrings to match
    thresholds: list[RiskThreshold] = field(default_factory=list)
    form_cues: list[str] = field(default_factory=list)  # Short coaching tips


# --- Generic (conservative defaults, matching current hardcoded values) ---

GENERIC_PROFILE = MovementProfile(
    name="generic",
    display_name="Generic",
    keywords=[],
    thresholds=[
        RiskThreshold("fppa", 15.0, 25.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 15.0, 25.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 15.0, 25.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Maintain neutral spine",
        "Control movement speed",
        "Keep joints aligned",
    ],
)

# --- Squat (Hewett 2005 squat-specific) ---

SQUAT_PROFILE = MovementProfile(
    name="squat",
    display_name="Squat",
    keywords=["squat", "goblet", "front squat", "back squat", "overhead squat"],
    thresholds=[
        RiskThreshold("fppa", 12.0, 20.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 12.0, 20.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 10.0, 15.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 20.0, 30.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Keep knees tracking over toes",
        "Chest up, maintain upright torso",
        "Push hips back as you descend",
        "Drive through heels to stand",
    ],
)

# --- Lunge (hip drop & asymmetry expected) ---

LUNGE_PROFILE = MovementProfile(
    name="lunge",
    display_name="Lunge",
    keywords=["lunge", "lunging", "split squat", "bulgarian"],
    thresholds=[
        RiskThreshold("fppa", 15.0, 25.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 15.0, 25.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop", enabled=False),
        RiskThreshold("trunk_lean", 20.0, 30.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry", enabled=False),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Keep front knee over ankle",
        "Maintain upright torso",
        "Lower until back knee near floor",
        "Step long enough for 90\u00b0 at both knees",
    ],
)

# --- Running (gait-specific, tighter for repetitive impact) ---

RUNNING_PROFILE = MovementProfile(
    name="running",
    display_name="Running",
    keywords=["running", "jogging", "sprinting", "sprint"],
    thresholds=[
        RiskThreshold("fppa", 10.0, 18.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 10.0, 18.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 6.0, 10.0, "pelvis", "hip_drop"),       # Noehren 2013
        RiskThreshold("trunk_lean", 12.0, 20.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 12.0, 20.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Land with foot under hips",
        "Keep hips level through stride",
        "Lean slightly forward from ankles",
        "Quick, light cadence",
    ],
)

# --- Walking (gait-specific) ---

WALKING_PROFILE = MovementProfile(
    name="walking",
    display_name="Walking",
    keywords=["walking"],
    thresholds=[
        RiskThreshold("fppa", 12.0, 20.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 12.0, 20.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 6.0, 10.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 10.0, 18.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 12.0, 20.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 300.0, 300.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 300.0, 300.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Maintain upright posture",
        "Keep hips level",
        "Swing arms naturally",
    ],
)

# --- Jump (ACL-focused, Hewett 2005 landing) ---

JUMP_PROFILE = MovementProfile(
    name="jump",
    display_name="Jump",
    keywords=["jump", "jumping", "box jump", "plyometric", "hop", "landing"],
    thresholds=[
        RiskThreshold("fppa", 10.0, 18.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 10.0, 18.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 10.0, 15.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 20.0, 30.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Land softly with knees bent",
        "Keep knees over toes on landing",
        "Absorb impact through hips and ankles",
        "Avoid stiff-legged landings",
    ],
)

# --- Deadlift (trunk lean expected due to hip hinge) ---

DEADLIFT_PROFILE = MovementProfile(
    name="deadlift",
    display_name="Deadlift",
    keywords=["deadlift", "dead lift", "hip hinge", "romanian"],
    thresholds=[
        RiskThreshold("fppa", 12.0, 20.0, "left_knee", "acl_valgus", enabled=False),
        RiskThreshold("fppa", 12.0, 20.0, "right_knee", "acl_valgus", enabled=False),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 25.0, 40.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Keep bar close to body",
        "Hinge at hips, not lower back",
        "Maintain flat back throughout",
        "Drive hips forward to lock out",
    ],
)

# --- Push-up (upper body focus) ---

PUSH_UP_PROFILE = MovementProfile(
    name="push_up",
    display_name="Push-up",
    keywords=["push-up", "push up", "pushup"],
    thresholds=[
        RiskThreshold("fppa", 15.0, 25.0, "left_knee", "acl_valgus", enabled=False),
        RiskThreshold("fppa", 15.0, 25.0, "right_knee", "acl_valgus", enabled=False),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop", enabled=False),
        RiskThreshold("trunk_lean", 15.0, 25.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 20.0, 30.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Keep body in a straight line",
        "Elbows at 45\u00b0 to body",
        "Lower chest to near the floor",
        "Engage core throughout",
    ],
)

# --- Plank (core stability focus) ---

PLANK_PROFILE = MovementProfile(
    name="plank",
    display_name="Plank",
    keywords=["plank", "side plank"],
    thresholds=[
        RiskThreshold("fppa", 15.0, 25.0, "left_knee", "acl_valgus", enabled=False),
        RiskThreshold("fppa", 15.0, 25.0, "right_knee", "acl_valgus", enabled=False),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop", enabled=False),
        RiskThreshold("trunk_lean", 8.0, 15.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Maintain straight line from head to heels",
        "Engage core and glutes",
        "Don't let hips sag or pike",
    ],
)

# --- Basketball profiles (sport-specific, ACL-focused) ---

BASKETBALL_LANDING_PROFILE = MovementProfile(
    name="basketball_landing",
    display_name="Basketball Landing",
    keywords=["dunk", "dunking", "block", "blocking", "rebound", "rebounding",
              "alley-oop", "alley oop", "slam dunk"],
    thresholds=[
        # Tightest FPPA for ACL risk on landing (Hewett 2005 landing biomechanics)
        RiskThreshold("fppa", 8.0, 15.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 8.0, 15.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 15.0, 25.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 600.0, 600.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 600.0, 600.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Land on both feet with knees bent",
        "Avoid single-leg stiff landing",
        "Keep knees tracking over toes on impact",
        "Absorb through hips and ankles",
    ],
)

BASKETBALL_CUTTING_PROFILE = MovementProfile(
    name="basketball_cutting",
    display_name="Basketball Cutting",
    keywords=["crossover", "euro step", "eurostep", "direction change",
              "cutting", "juke", "ankle breaker"],
    thresholds=[
        # Tight FPPA for plant-leg ACL risk on direction changes
        RiskThreshold("fppa", 10.0, 16.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 10.0, 16.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 18.0, 28.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 20.0, 30.0, "bilateral", "asymmetry", enabled=False),
        RiskThreshold("angular_velocity", 550.0, 550.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 550.0, 550.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Decelerate before cutting",
        "Plant foot firmly, knee over toe",
        "Keep center of mass low",
        "Avoid excessive knee valgus on plant leg",
    ],
)

BASKETBALL_SHOOTING_PROFILE = MovementProfile(
    name="basketball_shooting",
    display_name="Basketball Shooting",
    keywords=["shooting", "jump shot", "free throw", "layup", "lay-up",
              "three pointer", "mid-range", "floater", "hook shot"],
    thresholds=[
        # Moderate thresholds — asymmetry expected (dominant hand side)
        RiskThreshold("fppa", 12.0, 20.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 12.0, 20.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 10.0, 15.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 20.0, 30.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 20.0, 30.0, "bilateral", "asymmetry", enabled=False),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Align feet with basket on set shots",
        "Bend knees for power on jump shots",
        "Land balanced after release",
        "Follow through with wrist",
    ],
)

BASKETBALL_DRIBBLING_PROFILE = MovementProfile(
    name="basketball_dribbling",
    display_name="Basketball Dribbling",
    keywords=["dribbling", "ball handling", "between the legs", "behind the back"],
    thresholds=[
        # Moderate thresholds for general ball handling
        RiskThreshold("fppa", 12.0, 20.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 12.0, 20.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 18.0, 28.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 500.0, 500.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Stay low with knees bent",
        "Keep head up, eyes on court",
        "Protect ball with off hand",
        "Maintain athletic stance",
    ],
)

BASKETBALL_DEFENSE_PROFILE = MovementProfile(
    name="basketball_defense",
    display_name="Basketball Defense",
    keywords=["defensive slide", "guarding", "defensive stance", "on-ball defense",
              "closeout", "close out", "defensive shuffle"],
    thresholds=[
        # Low angular velocity thresholds for controlled lateral movement
        RiskThreshold("fppa", 12.0, 20.0, "left_knee", "acl_valgus"),
        RiskThreshold("fppa", 12.0, 20.0, "right_knee", "acl_valgus"),
        RiskThreshold("hip_drop", 8.0, 12.0, "pelvis", "hip_drop"),
        RiskThreshold("trunk_lean", 15.0, 25.0, "trunk", "trunk_lean"),
        RiskThreshold("asymmetry", 15.0, 25.0, "bilateral", "asymmetry"),
        RiskThreshold("angular_velocity", 400.0, 400.0, "left_knee", "angular_velocity_spike"),
        RiskThreshold("angular_velocity", 400.0, 400.0, "right_knee", "angular_velocity_spike"),
    ],
    form_cues=[
        "Stay low in athletic stance",
        "Slide feet, don't cross over",
        "Keep hips square to opponent",
        "Maintain active hands",
    ],
)


# Ordered by specificity (more specific keywords first).
# "walking" must be before "running" wouldn't matter since keywords don't overlap,
# but deadlift/push_up/plank have unique keywords.
# Ordered by specificity: profiles with keywords that are substrings of other
# profiles' keywords must come first (e.g., lunge has "split squat" which
# contains "squat", so lunge must be checked before squat).
_PROFILES: list[MovementProfile] = [
    LUNGE_PROFILE,     # "split squat" must match before "squat"
    DEADLIFT_PROFILE,  # "romanian deadlift" before generic matches
    PUSH_UP_PROFILE,
    PLANK_PROFILE,
    # Basketball profiles before generic "jump"/"running" so basketball-specific
    # keywords like "dunking", "shooting", "defensive slide" match first.
    BASKETBALL_LANDING_PROFILE,
    BASKETBALL_CUTTING_PROFILE,
    BASKETBALL_SHOOTING_PROFILE,
    BASKETBALL_DRIBBLING_PROFILE,
    BASKETBALL_DEFENSE_PROFILE,
    SQUAT_PROFILE,
    JUMP_PROFILE,
    RUNNING_PROFILE,
    WALKING_PROFILE,
]


def match_guideline(activity_label: str | None) -> MovementProfile:
    """Return the best-matching movement profile for a Gemini activity label.

    Returns GENERIC_PROFILE if no match or label is None.
    """
    if not activity_label:
        return GENERIC_PROFILE

    label = activity_label.lower().strip()

    for profile in _PROFILES:
        for keyword in profile.keywords:
            if keyword in label:
                return profile

    return GENERIC_PROFILE
