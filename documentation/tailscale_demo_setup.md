# BRACE Demo Setup (iPhone + Tailscale)

Last updated: 2026-02-21

## Goal

Run the full sideline demo flow:

1. iPhone records 240 FPS clips in tripod auto mode.
2. iPhone streams 60 FPS landmarks to `/live-stream`.
3. Backend returns collision triggers and accepts `/upload-clip`.
4. Backend generates risk report and queues coach notification.

## 1. Prerequisites

- Mac with Xcode + command line tools.
- iPhone with developer mode enabled.
- Same Tailscale tailnet on Mac and iPhone.
- Python 3.11 virtual environment on Mac.

## 2. Start backend

From the repo root:

```bash
cd /Users/paulbobev/Projects/BRACE
source .venv/bin/activate
./scripts/run_concussion_demo_server.sh
```

You should see URL hints like:

- `http://100.x.y.z:8443`
- `ws://100.x.y.z:8443/live-stream`

Health check:

```bash
curl http://127.0.0.1:8443/health
```

## 3. Generate/open iOS demo app

```bash
cd /Users/paulbobev/Projects/BRACE/mobile/ios/BraceTripodDemo
xcodegen generate
open /Users/paulbobev/Projects/BRACE/mobile/ios/BraceTripodDemo/BraceTripodDemo.xcodeproj
```

In Xcode:

1. Select `BraceTripodDemo` target.
2. Set Signing Team.
3. Select your iPhone.
4. Run.

## 4. Configure app on phone

In the app:

1. `Upload base URL`: `http://<mac-tailnet-ip>:8443`
2. Leave `Live stream override` empty (auto-derives `/live-stream`), or set:
   - `ws://<mac-tailnet-ip>:8443/live-stream`
3. Tap:
   - `Start Tripod Mode`
   - `Connect Live Stream`
   - `Start Synthetic Feed` (for no-external-sensor demo)

Expected outcome:

- Live logs show collision start/end events.
- Clip uploads are queued then sent.
- Latest report panel displays JSON metrics/risk/recommendation.

## 5. EDUROAM note

You do not need direct LAN connectivity between phone and Mac.

- If both devices are on Tailscale, traffic routes over tailnet.
- WebSockets remain the right transport for live landmarks (`/live-stream`).

## 6. Coach notification channels (optional)

Set environment variables before starting backend:

- Webhook: `COACH_REPORT_WEBHOOK_URL`
- SMTP: `COACH_SMTP_HOST`, `COACH_SMTP_PORT`, `COACH_SMTP_USER`, `COACH_SMTP_PASSWORD`, `COACH_SMTP_FROM`
- Fallback recipient: `DEFAULT_COACH_EMAIL`

## 7. Demo reset / cleanup

- Stop tripod mode in app.
- Stop backend (`Ctrl+C`).
- Uploaded clips: `/Users/paulbobev/Projects/BRACE/data/concussion_clips`
- Coach outbox: `/Users/paulbobev/Projects/BRACE/data/coach_reports`
