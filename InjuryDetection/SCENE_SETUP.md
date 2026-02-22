# Scene Setup Guide

How to wire the BRACE components in Unity after opening the project.

## Step 1: Fix Missing Script References

Opening the scene after this update will show missing script warnings for the deleted `CubeSpawner` and `SelectableCube` scripts.

1. In the Hierarchy, find and **delete the "Cube Spawner" GameObject** (root level)
2. If any runtime-spawned cubes remain from a previous play session, delete those too

## Step 2: Create the BraceManager GameObject

1. **Right-click in Hierarchy → Create Empty**, name it `BraceManager`
2. Add these components (Add Component button in Inspector):

| Component | Settings |
|---|---|
| **FrameCapture** | `Target Height`: 480, `JPEG Quality`: 65 |
| **BraceWebSocket** | `Server Url`: `ws://<YOUR_SERVER_IP>:8001/ws/analyze?mode=webcam&client=vr` |
| **BoundingBoxRenderer** | `Assumed Depth`: 3, `Smooth Alpha`: 0.3 |
| **DebugHUD** | `Show HUD`: checked (shows live status overlay) |

3. **All references auto-wire** — every component uses `FindObjectOfType<BraceWebSocket>()` as fallback, so you do NOT need to manually drag references in Inspector. Just having them on the same GameObject is enough.

## Step 3: Verify ControllerRaycast & InfoPanel

These already exist in the scene from the original project. They now auto-find BraceWebSocket via `FindObjectOfType`, so **no manual wiring needed**.

## Step 5: Verify Passthrough

Ensure passthrough is configured (should already be set from the original project):

1. **OVRCameraRig** → OVRManager → `Insight Passthrough` should be **enabled**
2. There should be an **OVR Passthrough Layer** component set to **Underlay**
3. The camera background should be **Solid Color** with **alpha = 0** (transparent)

## Final Hierarchy

```
OVRCameraRig (prefab)
  ├── TrackingSpace
  │   ├── CenterEyeAnchor (Main Camera)
  │   ├── LeftHandAnchor / ...
  │   └── RightHandAnchor / ...
  │       └── [ControllerRaycast + LineRenderer]
  ├── [OVRManager — passthrough enabled]
  ├── [OVR Passthrough Layer — underlay]
  └── [InfoPanel]

BraceManager (new)
  ├── [BraceWebSocket]
  ├── [FrameCapture]
  └── [BoundingBoxRenderer]

EventSystem
Directional Light
Global Volume
OVR Passthrough Layer Script
```

## Data Flow

```
Quest 3 Camera
  → FrameCapture (JPEG 480p @ 30fps)
  → BraceWebSocket (binary WS: timestamp + JPEG)
  → BRACE Server (YOLO + pose analysis)
  → BraceWebSocket.LatestResponse (JSON → BraceResponse)
  → BoundingBoxRenderer (world-space boxes per subject)
  → ControllerRaycast (raycast → select subject → WS text message)
  → Server sends detailed data for selected subject
  → InfoPanel (form score, biomechanics, injury risks, fatigue)
```

## Testing Without Quest 3

You can test in Unity Editor with a regular webcam:
1. Set `Server Url` to your local BRACE server: `ws://localhost:8001/ws/analyze?mode=webcam&client=vr`
2. Start the BRACE backend: `./run.sh`
3. Press Play in Unity — FrameCapture will use any available webcam
4. Bounding boxes will appear in the Game view (not in VR, but validates the pipeline)

## Server URL Examples

| Network | URL |
|---|---|
| Same machine | `ws://localhost:8001/ws/analyze?mode=webcam&client=vr` |
| Local network | `ws://192.168.1.100:8001/ws/analyze?mode=webcam&client=vr` |
| Tailscale (TLS) | `wss://baby-gator.tailea0e34.ts.net:8443/ws/analyze?mode=webcam&client=vr` |
