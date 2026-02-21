# Tripod Auto Mode Implementation Plan

Last updated: 2026-02-21 (implementation complete for demo handoff)
Owner: Codex
Scope: Automate iPhone 240 FPS capture/upload/delete and coach report delivery with practical production fixes.

## Goals

- Automate play clip lifecycle for tripod operation:
  - Always capture at 240 FPS in rolling segments.
  - Auto-detect play end and submit clip.
  - Delete local clip after successful upload.
- Add practical fix #1: explicit foreground/power operating mode for long sideline sessions.
- Add practical fix #2: offline-safe upload queue with retry/backoff and eventual delivery.
- Send concussion report to coach after backend analysis.

## Work Plan

- [x] Create implementation plan file.
- [x] Implement iOS tripod operating mode helpers (foreground + idle timer/power behavior).
- [x] Implement iOS rolling 240 FPS segmented recording for auto mode.
- [x] Implement iOS automated play-end finalization hooks (whistle/play-state endpoint hooks).
- [x] Implement iOS durable upload queue with retry/backoff.
- [x] Ensure iOS successful upload deletes local clip.
- [x] Implement backend coach report notifier with durable outbox (webhook + optional SMTP).
- [x] Trigger coach notification from `/upload-clip`.
- [x] Add/extend backend tests for notifier and endpoint contract.
- [x] Update mobile docs with integration/runbook.

## Update Log

- Added iOS `TripodUploadQueue` actor with durable queue file, exponential backoff retry, and clip deletion on success.
- Upgraded `SidelineCaptureManager` with tripod auto mode, segmented continuous 240 FPS recording, play finalization hooks, offline queue worker, and foreground/power guard.
- Added backend `CoachReportNotifier` with durable pending/sent outbox and webhook/SMTP channels.
- Wired `/upload-clip` to enqueue coach notifications via `coach_email` and `coach_id` form fields.
- Added notifier tests and extended endpoint contract tests to validate notification queuing.
- Updated `mobile/README.md` with tripod auto mode runbook, practical-fix notes, and coach delivery environment variables.
- Built full iOS demo app scaffold at `mobile/ios/BraceTripodDemo` (SwiftUI UI, controls, logs, report display, xcodegen project config, and unit tests).
- Added lightweight backend entrypoint `backend/concussion_demo_app.py` for demo-only startup (no full BRACE stack boot required).
- Added runnable backend launcher `scripts/run_concussion_demo_server.sh` with Tailscale URL hints.
- Added operator runbook `documentation/tailscale_demo_setup.md` for iPhone + Tailscale demo flow.
- Updated iOS demo defaults/placeholders to support `http/ws` Tailscale demo URLs.
- Validation:
  - `python -m pytest -q tests/test_concussion_pipeline.py tests/test_concussion_endpoints.py tests/test_concussion_notifier.py` -> 7 passed.
  - `python -m py_compile backend/concussion_pipeline.py backend/main.py backend/concussion_demo_app.py` -> pass.
  - `./scripts/run_concussion_demo_server.sh --help` -> pass.
  - `./scripts/run_concussion_demo_server.sh` + `curl http://127.0.0.1:8443/health` -> `{"status":"ok"}`.
  - `xcodebuild -project mobile/ios/BraceTripodDemo/BraceTripodDemo.xcodeproj -scheme BraceTripodDemo -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.0' test` -> 3 passed.
  - `xcrun swiftc -typecheck mobile/ios/*.swift` -> pass with deprecation warnings only.

## Notes

- iOS continuous camera capture must remain in foreground (Guided Access recommended).
- Device should be externally powered during games.
