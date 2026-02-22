# BRACE - Challenges

## VR Integration (Meta Quest 3)

### Challenge: Bandwidth over Tailscale VPN
The web frontend sends ~600KB/s per subject (landmarks, SRP joints, UMAP embeddings, cluster representatives). With 5 subjects at 30fps, this exceeds reasonable bandwidth for VR over Tailscale. We solved this by adding a `client=vr` query parameter that triggers a stripped-down response format — unselected subjects get only bbox + label (~80 bytes), and even selected subjects omit landmarks, SRP joints, UMAP, and cluster representatives. This achieved ~10x bandwidth reduction.

### Challenge: Subject Selection in VR
The web frontend handles subject selection entirely client-side (click on canvas → identify closest skeleton). VR can't do this the same way since it renders bounding boxes in 3D, not skeleton overlays. We added a `select_subject` WebSocket message type so the VR client can tell the server which subject it cares about, and the server only sends detailed analysis data for that subject.

### Challenge: Coordinate System Mismatch
BRACE uses image-space normalized coordinates [0,1] with y=0 at the top (standard for computer vision). Unity's viewport has y=0 at the bottom. The VR client must flip Y when converting bounding boxes to 3D world space.

### Challenge: Frame Capture on Quest 3
The Passthrough Camera API requires specific Android API levels (32+), Vulkan graphics, and has inherent 40-60ms latency from camera to texture. GPU-to-CPU readback for JPEG encoding adds additional overhead. We recommend on-demand capture and downscaling to 480p before encoding.

### Challenge: Latency Budget
End-to-end latency (camera capture → network → GPU inference → response) is 130-260ms. This is acceptable for bounding box overlays but too high for precise skeleton-to-body alignment. We chose bounding boxes + floating stats panels rather than skeleton overlays for the VR visualization.

### Challenge: Silent Failure Chain on Quest 3
The VR pipeline has multiple stages (camera → JPEG → WebSocket → server → JSON → bounding boxes), and if any stage fails silently the user sees nothing but the controller beam with no indication of what's wrong. Root causes included: (1) `[SerializeField]` references defaulting to null when not manually wired in Inspector, (2) WebSocket connection failures producing no on-screen feedback, (3) `Shader.Find()` returning null for shaders not included in the Android build. We fixed this by adding `FindObjectOfType` auto-wiring as fallback for all component references, a shader fallback chain (URP Unlit → Sprites/Default → Unlit/Color), auto-reconnect on WebSocket disconnect, and a DebugHUD overlay that shows live pipeline status (WS connected/disconnected, camera capturing/waiting, frames sent/received, box count).

## Concussion Risk Detection

### Challenge: Concussion Rating Always 100/100
The concussion rating always showed 100/100 during any motion because: (1) Head keypoints (nose, ears) were dropped in the COCO→MediaPipe mapping, so the head position was never actually tracked. (2) The `raw_joints` array is always shape (14,2) — the feature joint subset — so the `len(raw_joints)==33` check always failed. (3) The fallback used `_jerk_ema * 500.0` (whole-body SRP jerk) which instantly saturated to 100. We fixed this by adding nose/ear mappings to `_COCO_TO_MP`, extracting and smoothing head landmarks separately in the streaming analyzer, and replacing the broken jerk-based score with proper head-specific kinematics: linear acceleration (g) from nose displacement, angular velocity (rad/s) from ear-to-ear angle changes, and an adaptive baseline z-score for spike detection. A peak-hold + exponential decay mechanism prevents flickering. Normal basketball now scores 0-15; only actual head impacts spike above 50.
