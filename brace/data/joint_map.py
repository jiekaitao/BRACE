"""Kinect v2 joint constants and mapping to SRP-relevant joints."""

# Kinect v2 has 25 joints, indexed 0-24.
KINECT_JOINT_NAMES = {
    0: "SpineBase",
    1: "SpineMid",
    2: "Neck",
    3: "Head",
    4: "ShoulderLeft",
    5: "ElbowLeft",
    6: "WristLeft",
    7: "HandLeft",
    8: "ShoulderRight",
    9: "ElbowRight",
    10: "WristRight",
    11: "HandRight",
    12: "HipLeft",
    13: "KneeLeft",
    14: "AnkleLeft",
    15: "FootLeft",
    16: "HipRight",
    17: "KneeRight",
    18: "AnkleRight",
    19: "FootRight",
    20: "SpineShoulder",
    21: "HandTipLeft",
    22: "ThumbLeft",
    23: "HandTipRight",
    24: "ThumbRight",
}

# SRP anchor joints (used for body-frame construction)
HIP_LEFT = 12
HIP_RIGHT = 16
SHOULDER_LEFT = 4
SHOULDER_RIGHT = 8
SPINE_BASE = 0

# Ankle joints (used for gait cycle detection)
ANKLE_LEFT = 14
ANKLE_RIGHT = 18
FOOT_LEFT = 15
FOOT_RIGHT = 19

# Feature landmarks: 14 joints covering upper and lower body
# These are the joints whose positions form our feature vector.
FEATURE_LANDMARKS = [
    SHOULDER_LEFT,   # 4
    5,               # ElbowLeft
    6,               # WristLeft
    SHOULDER_RIGHT,  # 8
    9,               # ElbowRight
    10,              # WristRight
    HIP_LEFT,        # 12
    13,              # KneeLeft
    ANKLE_LEFT,      # 14
    FOOT_LEFT,       # 15
    HIP_RIGHT,       # 16
    17,              # KneeRight
    ANKLE_RIGHT,     # 18
    FOOT_RIGHT,      # 19
]

NUM_KINECT_JOINTS = 25
NUM_FEATURE_JOINTS = len(FEATURE_LANDMARKS)
FEATURE_DIM = NUM_FEATURE_JOINTS * 3  # 14 joints * 3 coords = 42

# Joint name lookup for feature landmarks
FEATURE_JOINT_NAMES = {idx: KINECT_JOINT_NAMES[idx] for idx in FEATURE_LANDMARKS}
