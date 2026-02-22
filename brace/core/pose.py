"""Pose estimation utilities: COCO-to-MediaPipe mapping, constants, and optional MediaPipe Tasks API wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Default model path (only needed for Pipeline B / offline MediaPipe landmarker)
DEFAULT_MODEL = Path(__file__).resolve().parent.parent.parent.parent / "PT_Hackathon" / "EXPERIMENT_PT_coach" / "models" / "pose_landmarker_heavy.task"

# MediaPipe has 33 landmarks
NUM_MP_LANDMARKS = 33

MP_LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

# Feature landmarks for SRP (major body joints, skipping face/hands)
FEATURE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
FEATURE_NAMES = {i: MP_LANDMARK_NAMES[i] for i in FEATURE_INDICES}
FEATURE_DIM = len(FEATURE_INDICES) * 2  # 14 joints * 2 (x, y) = 28D
FEATURE_DIM_3D = len(FEATURE_INDICES) * 3  # 14 joints * 3 (x, y, z) = 42D

# Anchor joints for body frame
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Bones for skeleton drawing
MP_BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # torso + arms
    (11, 23), (12, 24), (23, 24),  # torso to hips
    (23, 25), (25, 27), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 32),  # right leg
    (15, 17), (15, 19), (15, 21),  # left hand
    (16, 18), (16, 20), (16, 22),  # right hand
    (27, 29), (28, 30),  # heels
]


# COCO 17-keypoint indices (from YOLOv8-pose)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Mapping from COCO index -> MediaPipe landmark index
# All 14 feature joints + 4 SRP anchors are covered.
# Head keypoints (nose, ears) are included for head-tracking (concussion detection).
# Unmapped MediaPipe joints (eyes, mouth, fingers, heels) get zeros.
_COCO_TO_MP = {
    0: 0,    # nose -> nose
    3: 7,    # left_ear -> left_ear
    4: 8,    # right_ear -> right_ear
    5: 11,   # left_shoulder -> left_shoulder
    6: 12,   # right_shoulder -> right_shoulder
    7: 13,   # left_elbow -> left_elbow
    8: 14,   # right_elbow -> right_elbow
    9: 15,   # left_wrist -> left_wrist
    10: 16,  # right_wrist -> right_wrist
    11: 23,  # left_hip -> left_hip
    12: 24,  # right_hip -> right_hip
    13: 25,  # left_knee -> left_knee
    14: 26,  # right_knee -> right_knee
    15: 27,  # left_ankle -> left_ankle
    16: 28,  # right_ankle -> right_ankle
}


def coco_keypoints_to_landmarks(
    keypoints: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Convert 17 COCO keypoints from YOLO-pose to (33, 4) MediaPipe-format landmarks.

    Args:
        keypoints: (17, 3) array of [x_pixel, y_pixel, confidence] from YOLO-pose.
        img_w: frame width in pixels.
        img_h: frame height in pixels.

    Returns:
        (33, 4) array: [x_px, y_px, z=0, visibility] in pixel coordinates.
        Unmapped MediaPipe joints are filled with zeros.
        Feet (31, 32) are approximated using ankle positions.
    """
    out = np.zeros((NUM_MP_LANDMARKS, 4), dtype=np.float32)

    for coco_idx, mp_idx in _COCO_TO_MP.items():
        x, y, conf = keypoints[coco_idx]
        out[mp_idx, 0] = x
        out[mp_idx, 1] = y
        out[mp_idx, 2] = 0.0  # no depth from COCO
        out[mp_idx, 3] = conf

    # Approximate feet using ankle coords (negligible error for SRP)
    # left_foot_index (31) <- left_ankle (27)
    out[31] = out[27].copy()
    # right_foot_index (32) <- right_ankle (28)
    out[32] = out[28].copy()

    return out


# COCO-WholeBody 133-keypoint layout:
#   0-16:  17 COCO body keypoints (same as YOLO-pose)
#   17-22: 6 foot keypoints (left big toe, left small toe, left heel,
#          right big toe, right small toe, right heel)
#   23-90: 68 face keypoints
#   91-132: 42 hand keypoints (21 left + 21 right)

# Mapping from COCO-WholeBody index -> MediaPipe landmark index
# Body keypoints: same mapping as _COCO_TO_MP
# Foot keypoints: mapped to MediaPipe heel/foot indices
# Face/hand keypoints: mapped to approximate MediaPipe face landmarks where possible
COCO_WHOLEBODY_TO_MP = {
    # Body (same as _COCO_TO_MP)
    0: 0,    # nose -> nose
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
    # Feet (wholebody indices 17-22)
    17: 31,  # left_big_toe -> left_foot_index
    18: 31,  # left_small_toe -> left_foot_index (best approx)
    19: 29,  # left_heel -> left_heel
    20: 32,  # right_big_toe -> right_foot_index
    21: 32,  # right_small_toe -> right_foot_index (best approx)
    22: 30,  # right_heel -> right_heel
    # Face: approximate mappings for key facial landmarks
    # Wholebody face kpts 23-90 are ordered: outer contour, eyebrows, nose, eyes, mouth
    # We map a few key ones to MediaPipe face landmarks
    23: 7,   # left ear area (face contour start) -> left_ear
    39: 8,   # right ear area (face contour end) -> right_ear
    # Mouth corners
    84: 9,   # mouth left corner -> mouth_left
    78: 10,  # mouth right corner -> mouth_right
}


def wholebody133_to_mediapipe33(
    keypoints: np.ndarray,
) -> np.ndarray:
    """Convert 133 COCO-WholeBody keypoints to (33, 4) MediaPipe-format landmarks.

    Args:
        keypoints: (133, 4) array of [x_pixel, y_pixel, z, confidence].
                   Z column contains real metric depth from RTMW3D.

    Returns:
        (33, 4) array: [x_px, y_px, z, visibility] in pixel coordinates.
        Z values are preserved from the input (real depth, not zero).
        Unmapped MediaPipe joints are filled with zeros.
    """
    out = np.zeros((NUM_MP_LANDMARKS, 4), dtype=np.float32)

    for wb_idx, mp_idx in COCO_WHOLEBODY_TO_MP.items():
        if wb_idx >= keypoints.shape[0]:
            continue
        x, y, z, conf = keypoints[wb_idx]
        # Only overwrite if this keypoint has higher confidence than existing
        if conf > out[mp_idx, 3]:
            out[mp_idx, 0] = x
            out[mp_idx, 1] = y
            out[mp_idx, 2] = z  # real depth preserved
            out[mp_idx, 3] = conf

    # Eye landmarks: use COCO body eyes (indices 1-4) if face mapping missed them
    # COCO body: 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
    for coco_idx, mp_idx in [(1, 2), (2, 5), (3, 7), (4, 8)]:
        if coco_idx < keypoints.shape[0]:
            x, y, z, conf = keypoints[coco_idx]
            if conf > out[mp_idx, 3]:
                out[mp_idx, 0] = x
                out[mp_idx, 1] = y
                out[mp_idx, 2] = z
                out[mp_idx, 3] = conf

    return out


def create_landmarker(model_path: str | Path | None = None):
    """Create a MediaPipe PoseLandmarker for video mode.

    Requires mediapipe to be installed (Pipeline B / offline use only).
    """
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    if model_path is None:
        model_path = DEFAULT_MODEL
    model_path = str(model_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def landmarks_to_numpy(landmarks, img_w: int, img_h: int) -> np.ndarray:
    """Convert MediaPipe landmarks to (33, 4) numpy array: x_px, y_px, z, visibility."""
    out = np.zeros((NUM_MP_LANDMARKS, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        if i >= NUM_MP_LANDMARKS:
            break
        out[i, 0] = lm.x * img_w       # pixel x
        out[i, 1] = lm.y * img_h       # pixel y
        out[i, 2] = lm.z               # relative depth
        out[i, 3] = lm.visibility
    return out


def extract_poses_from_video(
    video_path: str | Path,
    model_path: str | Path | None = None,
    max_frames: int = 0,
) -> tuple[list[np.ndarray | None], int, int, float]:
    """Run pose estimation on every frame of a video.

    Requires mediapipe and opencv-python (Pipeline B / offline use only).

    Returns:
        landmarks_list: list of (33, 4) arrays (or None if no pose detected).
        width: video width in pixels.
        height: video height in pixels.
        fps: video FPS.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    landmarker = create_landmarker(model_path)
    landmarks_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_idx * 1000.0 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = landmarks_to_numpy(result.pose_landmarks[0], width, height)
            landmarks_list.append(lm)
        else:
            landmarks_list.append(None)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Pose estimation: {frame_idx}/{total_frames} frames...")

    cap.release()
    landmarker.close()
    print(f"  Pose estimation complete: {frame_idx} frames, {sum(1 for l in landmarks_list if l is not None)} with detections")
    return landmarks_list, width, height, fps
