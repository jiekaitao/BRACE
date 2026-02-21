# Mobile Pipeline Modules

This folder contains drop-in native modules for the concussion sideline pipeline:

- `ios/SidelineCaptureManager.swift`
- `ios/TripodUploadQueue.swift`
- `ios/LiveLandmarkWebSocket.swift`
- `android/app/src/main/java/com/brace/sideline/CollisionCaptureController.kt`
- `android/app/src/main/java/com/brace/sideline/LiveStreamSocketClient.kt`

It also includes a complete iOS demo app project generator:

- `ios/BraceTripodDemo/project.yml` (generate with `xcodegen`)
- `ios/BraceTripodDemo/Sources/*`

Demo runbook:

- `/Users/paulbobev/Projects/BRACE/documentation/tailscale_demo_setup.md`

## Runtime flow (event-trigger mode)

1. Open WebSocket to `/live-stream` and send 60 FPS head landmarks.
2. When server sends `collision_start`, begin 240 FPS recording.
3. When server sends `collision_end`, stop recording after `stop_after_ms` (default 1000 ms).
4. On whistle, compress/upload the clip to `POST /upload-clip` with `play_id`, `player_id`, and `file`.

The backend response body follows the required concussion report schema.

## Runtime flow (tripod auto mode)

1. Call `startTripodAutoMode(playID:playerID:coachEmail:)` on `SidelineCaptureManager`.
2. Manager records continuous 240 FPS segments and keeps a rolling segment history.
3. Feed 30 FPS play-state signals with `ingestPlaySignal(...)` from your lightweight detector.
4. On whistle or inactivity play-end, manager auto-finalizes clip window (`pre-roll + play + post-roll`).
5. Finalized clip is compressed to MP4 and pushed into durable `TripodUploadQueue`.
6. Queue worker uploads when network is available; on success local clip file is deleted automatically.

## Practical fixes implemented

- Foreground + power: tripod mode enforces active foreground and disables idle sleep (`isIdleTimerDisabled`) during session.
- Offline delivery: uploads are durable with queue persistence + exponential retry backoff.

## Backend coach delivery

`POST /upload-clip` supports extra optional form fields:

- `coach_email`
- `coach_id`

Backend queues notification delivery to coach with durable outbox:

- Webhook: `COACH_REPORT_WEBHOOK_URL`
- SMTP: `COACH_SMTP_HOST`, `COACH_SMTP_PORT`, `COACH_SMTP_USER`, `COACH_SMTP_PASSWORD`, `COACH_SMTP_FROM`, `COACH_SMTP_USE_TLS`
- Default recipient fallback: `DEFAULT_COACH_EMAIL`
