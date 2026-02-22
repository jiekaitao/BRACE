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

### Challenge: Head-to-Ground Impacts Missed Entirely
The `ConcussionMonitor` only detected person-to-person collisions. When a player's head strikes the ground (e.g., after a tackle), the impact was missed because there's no second person involved. We added a ground-impact detection pipeline that checks three conditions simultaneously: (1) high downward head velocity (>1.5 m/s peak in recent history), (2) sudden deceleration (current velocity drops below 30% of peak), and (3) head at ground level (head Y >= 85% of lowest visible keypoint Y). All three must be true to filter out controlled rolls/slides where the head doesn't abruptly stop. Ground impacts use a shorter impact duration (6ms vs 8ms for body-to-body) since the ground is harder, and the head's full pre-impact speed is the delta-v (ground has infinite mass). Scoring reuses the existing Rowson & Duma biomechanical model.

## Team Detection

### Challenge: Gemini Color Strings Fragile for Team Clustering
Gemini 2.5 Flash returns jersey color as a free-text string ("dark blue", "navy", "blue"), and the original `cluster_teams()` function groups by exact string match. This meant the same team could be split into multiple "teams" if Gemini returned slightly different color descriptions across subjects. We solved this by adding `cluster_teams_visual()` — a K-Means clustering approach on HSV color histograms extracted from actual player crops. Each crop's upper 60% (torso region) is converted to HSV, binned into a 16-hue x 8-saturation = 128-dim normalized histogram, then all subjects are clustered via K-Means (k=2). This is robust to lighting variation and Gemini wording inconsistency. Representative team colors are derived from the mean HSV of each cluster's torso pixels and converted to hex for frontend display.

### Challenge: Uniform-Color Histograms Break K-Means in Tests
During testing, synthetic solid-color crops produced delta-function histograms (single bin = 1.0, 127 bins = 0.0). In 128-dim space, the Euclidean distance between any two such histograms is always sqrt(2) unless they hit the exact same bin. Adjacent hue bins (e.g., bin 6 vs bin 7 for two shades of red) were equidistant from blue bins (86, 87), so K-Means couldn't distinguish teams. Adding +-15 pixel noise to test crops produced realistic multi-bin histogram spread, fixing the clustering.

## Concussion Risk Detection

## Gemini Activity Classification

### Challenge: Activity Classification Never Triggered in Live Sessions
The `_classify_cluster_webcam()` function existed in `main.py` but was never called — it was an orphaned helper with no trigger in the WebSocket frame loop. Even after wiring it up, the `min_segments=3` threshold in `get_clusters_needing_classification()` was too strict for real-time webcam sessions: continuous motion typically produces 1 long segment per cluster, never reaching the 3-segment threshold. We lowered the threshold to 1 segment and added the classification trigger after each re-analysis pass. Additionally, using `gemini-2.5-pro` for a simple image classification task caused frequent refusals ("cannot be determined", "I am sorry") — switching to `gemini-2.5-flash` with a stronger prompt ("Never say 'I cannot' or 'sorry'. Always give your best guess.") resolved the refusal issue and reduced classification latency.

### Challenge: Identity Tracking Instability in Crowded Scenes (12+ People)
In demo videos with large fitness classes (12+ people), subject identities constantly swapped — the selected person's bounding box would jump to another person, sidebar buttons flickered, and the analyzer accumulated data for the wrong person. The root cause was a three-layer problem:

**Backend**: The identity resolver's spatial fallback thresholds were too loose (IoU=0.1, centroid distance=0.15), allowing adjacent people to be falsely re-matched. Gallery size was too small (20 embeddings) for robust matching in crowds, and the single-person bias (lowered threshold to 0.30 when only 1 subject inactive) fired even with 12 subjects. There was no cross-frame temporal consistency — the resolver matched identities independently per frame, allowing oscillation between two identities on alternate frames.

**Frontend**: Subjects were deleted from the Map the instant they left `active_track_ids` (no grace period), and auto-selection overrode user clicks every frame at 30-60fps, making it impossible to maintain a selection during brief occlusions.

We fixed this with coordinated changes across 4 files:
1. **Identity resolver thresholds** (`identity_resolver.py`): Raised match_threshold (0.40→0.45), tightened spatial IoU (0.1→0.3) and centroid distance (0.15→0.08), increased gallery size (20→40), added per-frame embedding extraction (interval 2→1), disabled single-person bias when >3 subjects, and added an embedding quality gate (confidence≥0.5).
2. **Temporal consistency** (`identity_resolver.py`): Added a sliding-window assignment tracker that penalizes rapid identity switching. Confirmed long-lived subjects get double switch penalty (0.2), making it very hard to incorrectly reassign a stable identity.
3. **Subject manager hysteresis** (`subject_manager.py` + `main.py`): `active_track_ids` now includes subjects seen within the last 45 frames (~1.5s), preventing the frontend from prematurely deleting briefly-occluded subjects. Added stability logging to detect problematic subjects.
4. **Frontend sticky selection** (`useAnalysisWebSocket.ts`): Added 1500ms grace period before deleting subjects, `userExplicitlySelectedRef` flag to prevent auto-selection from overriding user clicks, and stable sidebar buttons based on all subjects in the Map (including grace period).

### Challenge: Concussion Rating Always 100/100
The concussion rating always showed 100/100 during any motion because: (1) Head keypoints (nose, ears) were dropped in the COCO→MediaPipe mapping, so the head position was never actually tracked. (2) The `raw_joints` array is always shape (14,2) — the feature joint subset — so the `len(raw_joints)==33` check always failed. (3) The fallback used `_jerk_ema * 500.0` (whole-body SRP jerk) which instantly saturated to 100. We fixed this by adding nose/ear mappings to `_COCO_TO_MP`, extracting and smoothing head landmarks separately in the streaming analyzer, and replacing the broken jerk-based score with proper head-specific kinematics: linear acceleration (g) from nose displacement, angular velocity (rad/s) from ear-to-ear angle changes, and an adaptive baseline z-score for spike detection. A peak-hold + exponential decay mechanism prevents flickering. Normal basketball now scores 0-15; only actual head impacts spike above 50.
