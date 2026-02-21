# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT:
NEVER EVER push to the GitHub repo by yourself

For all the major challenges you encounter, add it to documentation/challenges.md as we will need to talk about the challenges when the hackathon ends.

## Build & Run Commands

### Docker (primary development)
```bash
# Auto-detect hardware and start (recommended):
./run.sh                                # auto-detects Mac/NVIDIA, starts in background
./run.sh --dev                          # dev mode with hot-reload (docker-compose.dev.yml)
./run.sh down                           # stop everything
./run.sh logs                           # tail logs
./stop.sh                               # stop Docker + tear down Tailscale Funnel

# Manual profile selection:
docker compose --profile cpu up -d      # Mac / CPU-only
docker compose --profile nvidia up -d   # NVIDIA GPU

# Restart backend (picks up bind-mounted .py changes):
docker compose --profile cpu restart backend-cpu      # Mac
docker compose --profile nvidia restart backend       # NVIDIA

# Rebuild frontend (required for TS/TSX changes):
docker compose build frontend && docker compose --profile cpu up -d frontend

docker compose --profile cpu logs -f backend-cpu      # Tail backend logs (Mac)
docker compose --profile nvidia logs -f backend       # Tail backend logs (NVIDIA)
```

**Profiles**: The system uses Docker Compose profiles to auto-detect hardware:
- `cpu` — Mac / non-GPU machines. Uses `Dockerfile.cpu` (Python 3.12 slim, CPU PyTorch, onnxruntime CPU).
- `nvidia` — NVIDIA GPU machines. Uses `Dockerfile` (CUDA 12.8, TensorRT FP16, onnxruntime-gpu).

The `run.sh` script auto-detects which profile to use based on the OS and GPU availability. It also configures Tailscale Funnel for public HTTPS/WSS access (ports 443 & 8443).

**Docker services**: Frontend (3000), Backend (8001→8000), MongoDB (27017, internal), VectorAI (5555→50051, optional).

**Hot-reload in production compose**: These files are bind-mounted and take effect on `docker compose restart backend` (or `backend-cpu` on Mac): `main.py`, `streaming_analyzer.py`, `subject_manager.py`, `gemini_classifier.py`, `tensorrt_utils.py`, `identity_resolver.py`, `multi_person_tracker.py`, `botsort_tracker.py`, `movement_quality.py`, `movement_guidelines.py`, `motion_segments.py`, `risk_profile.py`, `chat_agent.py`, `voice_alerts.py`, `vectorai_store.py`, `vector_movement_search.py`. All other backend changes require rebuilding the backend image. New files require adding a bind-mount in `docker-compose.yml` and running `docker compose up -d backend` (recreate, not just restart).

**Dev mode** (`docker-compose.dev.yml`): Volume-mounts entire `backend/` and `brace/` directories with `uvicorn --reload` for automatic Python hot-reload.

**Frontend changes always require rebuild** (`docker compose build frontend && docker compose up -d frontend`). Do NOT pass `NEXT_PUBLIC_WS_URL` or `NEXT_PUBLIC_API_URL` as Docker build args — the frontend auto-detects the correct protocol and port at runtime via `frontend/src/lib/api.ts`.

### Python (core library + tests)
```bash
pip install -e .                                        # Install brace package
python -m pytest tests/ -v                              # Run all tests
python -m pytest tests/test_srp.py -v                   # Single test file
python -m pytest tests/test_anomaly.py::test_normal_scores_low  # Single test
```

Note: Some tests require GPU-only dependencies (`ultralytics`, `motor`) and will only pass inside Docker.

### Frontend (TypeScript)
```bash
cd frontend && npx tsc --noEmit         # Type-check (run before committing)
```

### Remote Access (Tailscale)
```bash
# One-time setup (needs sudo once):
sudo tailscale set --operator=$USER

# Serve frontend (HTTPS) and backend (WSS) over Tailscale:
tailscale serve --bg --https=443 http://localhost:3000
tailscale serve --bg --https=8443 http://localhost:8001

# Access from any device on the tailnet:
# https://baby-gator.tailea0e34.ts.net/          (frontend)
# wss://baby-gator.tailea0e34.ts.net:8443/       (backend WebSocket)
tailscale serve status                   # Check what's being served
tailscale serve reset                    # Remove all serve configs
```

HTTPS is required for `getUserMedia` (webcam) from remote devices. The frontend detects `https:` and automatically connects to the backend on port 8443 instead of 8000.

## Architecture

### Real-Time Streaming System (primary)

```
Client (Browser or Quest 3 VR)
  ↓ JPEG frames + video_time (binary WebSocket)
FastAPI backend (port 8000, exposed as 8001)
  ↓
PoseBackend.process_frame(rgb) → list[PipelineResult]
  ↓
IdentityResolver.resolve() → stable subject IDs (CLIP-ReID + body proportions)
  ↓
SubjectManager → per-subject StreamingAnalyzer
  ↓
StreamingAnalyzer.process_frame(landmarks):
  OneEuroFilter smoothing → SRP normalize → segment → cluster → consistency
  MovementQualityTracker → biomechanics, form score, fatigue detection
  ↓
MultiSubjectFrameResponse JSON → WebSocket → client
  ↓
Web: useAnalysisWebSocket hook → subjectsRef → Canvas/Three.js at 60fps
VR:  _build_vr_response() strips heavy fields → bounding boxes + stats panels
```

### WebSocket Client Types

The `/ws/analyze` endpoint accepts a `client` query parameter:
- `client=web` (default) — full response with landmarks, SRP joints, UMAP embeddings, cluster representatives, SMPL
- `client=vr` — stripped response (~10x smaller): unselected subjects get bbox+label only (~80 bytes), selected subject gets analysis data minus skeleton/visualization fields

VR clients send `{"type": "select_subject", "subject_id": N}` text messages to select which subject receives detailed data. Send `null` subject_id to deselect.

See `documentation/quest3_integration_guide.md` for comprehensive Unity/Quest 3 integration instructions.

### Two Pipeline Backends

Selected via `PIPELINE_BACKEND` env var (default: `legacy`):

| | Legacy | Advanced |
|---|---|---|
| Detector | YOLO11-pose (17 COCO kpts) | YOLO11 detect + RTMW3D (133 kpts) |
| Tracker | ByteTrack (built-in) | BoT-SORT (BoxMOT) |
| Depth | 2D only (z=0) → 28D features | Real 3D Z → 42D features |
| SMPL | None | HybrIK (every 3rd frame) |
| Interface | `LegacyPoseBackend` | `AdvancedPoseBackend` |

Both implement `PoseBackend` ABC → `process_frame(rgb) → list[PipelineResult]`.

**YOLO model**: `YOLO_MODEL` env var (default: `yolo11m-pose.pt`), auto-exported to TensorRT FP16. NEVER set `model.predictor = None` — this reloads the 114MB TRT engine. Clear tracker lists in-place instead.

### SRP Normalization (shared core math)

Scale/Rotation/Position invariance for cross-session comparison:
1. **Origin**: pelvis midpoint `(hip_L + hip_R) / 2`
2. **Scale**: coordinates in hip-width units
3. **Rotation**: Gram-Schmidt body frame from hip vector (x-axis) + shoulder-pelvis vector (y-axis)

Variants: `normalize_frame()` (2D), `normalize_frame_3d_real()` (3D), `normalize_frame_visual()` (position+scale only, no rotation — used for skeleton display).

### Joint Index Systems

- **MediaPipe 33**: Hips=23,24. Shoulders=11,12. Used by frontend and streaming pipeline.
- **COCO 17**: YOLO11-pose output. Mapped via `coco_keypoints_to_landmarks()`.
- **COCO-WholeBody 133**: RTMW3D output. Mapped via `wholebody133_to_mediapipe33()`.
- **Feature joints (14)**: `FEATURE_INDICES = [11,12,13,14,15,16,23,24,25,26,27,28,31,32]` — shoulders, elbows, wrists, hips, knees, ankles, feet. Used for SRP features (14×2=28D or 14×3=42D).

### Frontend State Management

Next.js 15 + React 19 + TypeScript. Mixed pattern to avoid re-rendering at 60fps:
- **React state** (triggers renders): phase, nSegments, nClusters, selectedSubjectId, activeTrackIds — throttled to ~4/sec
- **Refs** (canvas reads directly): `subjectsRef` (Map of all subject data), `selectedSubjectRef`, `highlightedClusterRef`

### Skeleton-Video Alignment

Demo mode uses `DelayedVideoCanvas` to buffer video frames and display them delayed by the pipeline RTT (EMA-smoothed). This aligns the skeleton overlay with the displayed frame. Z-index stacking: video element → DelayedVideoCanvas (z:5) → AnalysisCanvas skeleton (z:10) → VideoControls (z:20). The video must remain visible (not `opacity:0`) or Chrome blocks autoplay.

### Personalization & Injury Intake Flow

```
Home (/) → Personal or Team path selection
  ↓
Onboarding (/onboarding) — 4 steps (personal) or 2 steps (team):
  1. Login/Register (username + UUID session token)
  2. Chat with InjuryChatAgent (Gemini 2.5 Pro, multi-turn) → extracts InjuryProfile
  3. Review injury profile (severity, timeframe per injury)
  4. Mode selection (webcam / upload / demo)
  ↓
Dashboard (/dashboard) — personalized entry point for returning users
  ↓
Analyze (/analyze) — real-time motion analysis with personalized thresholds
```

`RiskModifiers` (from `risk_profile.py`) scale biomechanical thresholds per-injury (e.g., ACL → lower FPPA threshold). Applied per-frame in `MovementQualityTracker`. Stored in MongoDB via `AuthUser.injury_profile` + `AuthUser.risk_modifiers`.

### Voice Coaching

Real-time injury risk alerts via ElevenLabs TTS. `VoiceAlertGenerator` (`voice_alerts.py`) deduplicates alerts with 8s cooldown, requires 3s sustained medium-risk before alerting. Frontend toggle via `useVoiceCoaching` hook.

### Offline Analysis Pipelines (secondary)

**Video analysis**: `scripts/analyze_video.py` — MediaPipe → SRP → segment → cluster → annotated MP4.

**Kinect validation**: `scripts/05_full_demo.py` — 3D skeleton CSVs → SRP → gait cycles → baseline → anomaly scoring → plots.

## Key Backend Files

- `backend/main.py` — FastAPI server, WebSocket frame loop, VR client mode (`_build_vr_response`), video loop detection
- `backend/streaming_analyzer.py` — Per-subject analysis engine (OneEuroFilter, SRP, clustering, UMAP)
- `backend/subject_manager.py` — track_id → StreamingAnalyzer mapping, stale cleanup, loop freeze
- `backend/pipeline_interface.py` — `PipelineResult` dataclass + `PoseBackend` ABC
- `backend/legacy_backend.py` / `advanced_backend.py` — Pipeline implementations
- `backend/identity_resolver.py` — Cross-cut re-ID, gallery management, subject identity lifecycle
- `backend/botsort_tracker.py` — BoT-SORT via BoxMOT with CMC error recovery
- `backend/gemini_classifier.py` — Gemini 2.5 Pro activity labels (google-genai SDK, NOT google-generativeai)
- `backend/movement_quality.py` — Movement quality assessment: bone-length projection filter, biomechanics (FPPA, hip drop, trunk lean, BAI, angular velocity), Isolation Forest anomaly scoring, rule-based injury risk thresholds, CoM estimation, signal processing (SPARC, LDLJ, sample entropy, spectral median frequency), statistics (EWMA, CUSUM), kinematic chain sequencing, form scoring, composite fatigue
- `backend/movement_guidelines.py` — Movement-specific biomechanical profiles (squat, lunge, running, etc.) with per-exercise risk thresholds and coaching cues
- `backend/db.py` — MongoDB layer via motor (async) / pymongo (sync)
- `backend/auth.py` — Session-based authentication (username + UUID token)
- `backend/chat_agent.py` — Gemini 2.5 Pro injury intake agent (multi-turn, extracts structured `InjuryProfile` JSON). Uses `gemini-2.5-pro` model, NOT flash.
- `backend/risk_profile.py` — `RiskModifiers` dataclass, per-injury threshold scaling (FPPA, hip drop, trunk lean, asymmetry, angular velocity), `apply_modifiers()` for personalization
- `backend/voice_alerts.py` — `VoiceAlertGenerator` with cooldown/dedup for real-time injury risk alerts
- `backend/vector_movement_search.py` — `MovementSearchEngine` for semantic cross-session motion similarity via VectorAI
- `backend/tts_elevenlabs.py` — ElevenLabs TTS integration
- `backend/vectorai_store.py` — VectorAI gRPC integration for semantic movement search (optional, graceful degradation)
- `brace/core/motion_segments.py` — SRP normalization, velocity segmentation, agglomerative clustering, consistency

## Key Frontend Files

**Pages:**
- `frontend/src/app/page.tsx` — Landing page with personal/team path selection
- `frontend/src/app/onboarding/page.tsx` — Multi-step onboarding (login, injury chat, profile review, mode select)
- `frontend/src/app/dashboard/page.tsx` — Personalized dashboard for returning users
- `frontend/src/app/analyze/page.tsx` — Main analysis page (webcam/video/demo modes)
- `frontend/src/app/dev/` — Development sandbox pages (auth, chat, components, risk-profile, voice, timeline)

**Hooks:**
- `frontend/src/hooks/useAnalysisWebSocket.ts` — WebSocket lifecycle, frame capture (480p, JPEG 0.65), multi-subject state
- `frontend/src/hooks/useChat.ts` — Injury intake chat agent (sends to `/api/chat`, extracts `InjuryProfile`, `confirmProfile()`)
- `frontend/src/hooks/useVoiceCoaching.ts` — ElevenLabs TTS voice alert management

**Components:**
- `frontend/src/components/AnalysisCanvas.tsx` — 2D skeleton overlay (visibility threshold 0.3), receipt-time interpolation at 60fps
- `frontend/src/components/DelayedVideoCanvas.tsx` — Buffered video display delayed by pipeline RTT for skeleton alignment
- `frontend/src/components/SkeletonGraph.tsx` — Hip-centered skeleton with depth-based rendering (Three.js + React Three Fiber)
- `frontend/src/components/MovementQualityPanel.tsx` — Movement status bar, form quality, joint quality, biomechanics, fatigue timeline
- `frontend/src/components/ChatPanel.tsx` — Injury intake chat interface
- `frontend/src/components/AnimatedSkeletonDemo.tsx` — Real-time animated skeleton with injury-specific joint angle visualization
- `frontend/src/components/TeamSportBrowser.tsx` — Sport selection for team monitoring mode
- `frontend/src/components/InjuryProfileCard.tsx` — Display/edit injury profile with severity badges

**Lib:**
- `frontend/src/lib/api.ts` — `getApiBase()` / `getWsBase()` — auto-detects HTTP/HTTPS and correct backend port (8000 local, 8443 Tailscale)
- `frontend/src/lib/types.ts` — All TypeScript interfaces (SubjectState, PipelineResult, FrameQuality, InjuryProfile, RiskModifiers, etc.)
- `frontend/src/lib/syntheticMotion.ts` — Synthetic squat cycle generation, joint angle computation, injury-joint chain mappings (frontend port of risk_profile.py)
- `frontend/src/lib/auth.ts` — Auth helpers for session management
- `frontend/src/contexts/AuthContext.tsx` — Auth state with `injury_profile` and `risk_modifiers` on `AuthUser`

## Important Patterns

- **Docker import paths**: Backend files are bind-mounted flat to `/app/` in Docker, not `/app/backend/`. Use try/except for cross-module imports: `try: from backend.foo import Bar` / `except ImportError: from foo import Bar`. Exception: `brace/core/motion_segments.py` is mounted at `/app/brace/core/motion_segments.py`. Adding a new bind-mount requires `docker compose up -d backend` (recreate), not just `restart`.
- **Visibility threshold**: 0.3 throughout — landmarks below this are considered out-of-frame. Backend holds last good filtered position; frontend fades to 0.1 opacity.
- **Feature dimension lock**: First frame determines 28D (2D) or 42D (3D) for the session.
- **Clustering**: Agglomerative with spectral distance (mean pose + FFT power spectrum), average linkage, distance normalized by `sqrt(feat_dim)`. Default `cluster_threshold=2.0`, `min_segment_sec=2.0`.
- **Scene cuts**: `backend.on_scene_cut()` + `resolver.on_scene_cut()` reset tracker and re-ID gallery. CLIP-ReID only runs 5 frames post-cut.
- **Video loop**: Frontend detects `currentTime` jump backward >1s, clears landmark frames, sets `replaying` state. Backend freezes analyzers via `SubjectManager.note_loop()`. Do NOT call `pipeline.reset()` on loops — use `resolver.on_scene_cut()` + `manager.note_loop()` instead.
- **Sliding window**: StreamingAnalyzer caps at `MAX_ANALYSIS_WINDOW=2000` features.
- **Gemini**: Lazy-init, cached, rate-limited (2s), background thread via `asyncio.ensure_future()` (not `create_task()`) for `run_in_executor()` futures. ~$0.00011/call.
- **Frame capture tuning**: `CAPTURE_H=480`, `JPEG_QUALITY=0.65`, `MAX_IN_FLIGHT=5`, playback rate floor 0.8. These prevent video slowdown on RTX 5090.
- **Skeleton interpolation**: Uses `performance.now()` receipt time (NOT `video.currentTime`) for smooth 60fps interpolation. `video.currentTime` updates in discrete ~33ms steps causing jitter.
- **Movement quality**: `MovementQualityTracker` runs per-frame (bone-length projection filter, biomechanical angles, joint angular velocity, Isolation Forest anomaly scoring, rule-based injury risk thresholds, Center of Mass estimation, form score, phase detection) and per-cluster (ROM decay, SPARC, LDLJ, EWMA, CUSUM, cross-correlation, sample entropy, spectral median frequency, kinematic chain sequencing, composite fatigue). Composite fatigue is a weighted score: ROM 0.25, EWMA 0.20, CUSUM 0.20, correlation 0.15, SPARC 0.08, LDLJ 0.07, spread 0.05. Form score = deviation from cluster representative trajectory (0-100). The `quality` dict is included in the WebSocket response and stored on `SubjectState.quality` in the frontend.
- **VR bbox coordinate flip**: BRACE uses image-space y=0 at top; Unity viewport y=0 at bottom. VR clients must flip Y when converting bounding boxes.
- **VectorAI graceful degradation**: If VectorAI DB is unavailable, the system continues without semantic search — all `_vectorai_store` checks are guarded.
- **Gemini SDK**: Both `GeminiActivityClassifier` and `InjuryChatAgent` use `gemini-2.5-pro` via the `google-genai` SDK (NOT `google-generativeai`). Classifier sends image frames; chat agent does multi-turn text extracting structured JSON `{"injuries": [...], "complete": bool}`.
- **Risk personalization**: `RiskModifiers` scales thresholds per-metric (`fppa_scale`, `hip_drop_scale`, etc.) + `monitor_joints` list. Applied in `MovementQualityTracker` at runtime. Persisted in MongoDB via auth profile endpoints.
- **Voice alert dedup**: `VoiceAlertGenerator` has 8s cooldown per alert type, 3s sustained threshold for medium-severity risks, and deduplicates across joints. Alerts are human-readable via `RISK_DESCRIPTIONS` and `JOINT_SPOKEN` mappings.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_BACKEND` | `legacy` | `legacy` or `advanced` |
| `GOOGLE_GEMINI_API_KEY` | — | Gemini 2.5 Pro (used by both activity classifier and chat agent) |
| `DISABLE_REID` | — | Set to `1` for pure ByteTrack (no appearance matching) |
| `YOLO_MODEL` | `yolo11m-pose.pt` | YOLO model file (auto-exports to TensorRT FP16) |
| `ELEVENLABS_API_KEY` | — | ElevenLabs TTS for voice coaching |
| `MONGODB_URI` | `mongodb://mongo:27017/brace` | MongoDB connection string |
| `VECTORAI_HOST` / `VECTORAI_PORT` | `vectorai` / `50051` | VectorAI gRPC (optional) |
| `NEXT_PUBLIC_WS_URL` | auto-detected | WebSocket endpoint override (build-time). Leave unset for auto-detection. |
| `NEXT_PUBLIC_API_URL` | auto-detected | HTTP API base override (build-time). Leave unset for auto-detection. |

## Testing

350 tests using synthetic skeleton data (sinusoidal motion with fixed anchor joints). All tests must pass before any commit. Key test files: `test_movement_quality.py` (99), `test_movement_guidelines.py` (33), `test_motion_clustering.py` (23), `test_gemini_classifier.py` (22), `test_botsort_tracker.py` (16), `test_identity_resolver.py` (15), `test_pipeline_interface.py` (20).

Some tests (`test_auth.py`, `test_botsort_tracker.py`, `test_chat_agent.py`, `test_identity_resolver.py`, `test_multi_person.py`, `test_pipeline_interface.py`, `test_db.py`) require GPU-only dependencies and will only pass inside Docker containers.
