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

### Challenge: Bounding Boxes Huge and Misplaced in VR (Camera Projection Mismatch)
The `/dev/streams/` debug page renders bounding boxes correctly (server-side, drawn directly on the image), but on the Quest 3 headset the same boxes appear very large and scattered. The root cause is a camera projection mismatch: the bbox coordinates are normalized [0,1] relative to the **passthrough camera's image**, but the Unity code converts them to 3D world space using `Camera.ViewportToWorldPoint`, which maps through the **VR rendering camera's frustum**. These two cameras have different FOVs and projections — the passthrough camera is wider-angle — so the mapping diverges badly at the edges. Combined with `assumedDepth = 3f` (3 meters), even small angular errors become large positional errors in world space. The fix must happen in the Unity code: either use the passthrough camera's actual intrinsics to project [0,1] image coords → 3D rays, or apply a scale/offset correction to remap passthrough image coords into VR camera viewport space before calling `ViewportToWorldPoint`.

### Challenge: OpenCV LK Pyramid Size Mismatch (Quest 3)
BoT-SORT's Camera Motion Compensation (CMC) uses Lucas-Kanade sparse optical flow to estimate camera motion between consecutive frames. When the Quest 3 passthrough camera sends frames with inconsistent resolutions (due to encoding changes, resolution renegotiation, or capture timing), the optical flow pyramid built from the previous frame doesn't match the current frame's size, causing `cv2.error: (-215:Assertion failed) prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size()`. The existing try/except caught and retried, but the error spammed logs on every frame. We fixed it by tracking the previous frame size in `BoTSortTracker` and proactively resetting `cmc.prev_img = None` when the resolution changes, preventing the error entirely.

## VectorAI Integration

### Challenge: Dimension Mismatch Breaking All Person Embeddings
The `person_embeddings` collection was created with 768 dimensions (for CLIP embeddings), but the identity resolver sends 512D OSNet embeddings during normal operation. Every `UpsertVector` call failed with a dimension mismatch error, producing repeated `[vectorai] store_person_embedding failed: Failed to upsert vector` log spam. The root cause was that `_clip_reid_extractor` was loaded at startup but never passed to `IdentityResolver` as `cross_cut_extractor` — so CLIP was dead code, and all embeddings were 512D OSNet. We fixed this by: (1) changing the collection dimension to 512, (2) adding dimension validation at startup that detects stale collections and recreates them, (3) adding a 512D guard in the identity resolver so CLIP 768D embeddings from post-cut mode are never sent to VectorAI, (4) passing the CLIP extractor through as `cross_cut_extractor` so it actually works for cross-cut matching.

### Challenge: VectorAI Treated as Optional Despite Being Required
All VectorAI code used "graceful degradation" — failures were silently swallowed with `if _vectorai_store is None: return`. This meant the feature was broken for months without anyone noticing. Motion segment storage was scaffolded but never called from the pipeline, and `VectorActivityClassifier` was passed to the basketball processor but never used. We made VectorAI a hard requirement: the backend raises at startup if VectorAI is unreachable, imports are no longer wrapped in try/except, and motion segment storage is now wired into `StreamingAnalyzer.run_analysis()` so cross-session movement search actually has data.

## Concussion Risk Detection

### Challenge: Concussion Rating Always 100/100
The concussion rating always showed 100/100 during any motion because: (1) Head keypoints (nose, ears) were dropped in the COCO→MediaPipe mapping, so the head position was never actually tracked. (2) The `raw_joints` array is always shape (14,2) — the feature joint subset — so the `len(raw_joints)==33` check always failed. (3) The fallback used `_jerk_ema * 500.0` (whole-body SRP jerk) which instantly saturated to 100. We fixed this by adding nose/ear mappings to `_COCO_TO_MP`, extracting and smoothing head landmarks separately in the streaming analyzer, and replacing the broken jerk-based score with proper head-specific kinematics: linear acceleration (g) from nose displacement, angular velocity (rad/s) from ear-to-ear angle changes, and an adaptive baseline z-score for spike detection. A peak-hold + exponential decay mechanism prevents flickering. Normal basketball now scores 0-15; only actual head impacts spike above 50.
