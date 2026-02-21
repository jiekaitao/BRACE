# BRACE Tripod Demo iOS App

This is a complete demo iOS app target for BRACE sideline concussion workflow.

## Generate project

```bash
cd /Users/paulbobev/Projects/BRACE/mobile/ios/BraceTripodDemo
xcodegen generate
```

This creates:

- `/Users/paulbobev/Projects/BRACE/mobile/ios/BraceTripodDemo/BraceTripodDemo.xcodeproj`

## Run tests

```bash
xcodebuild \
  -project /Users/paulbobev/Projects/BRACE/mobile/ios/BraceTripodDemo/BraceTripodDemo.xcodeproj \
  -scheme BraceTripodDemo \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  test
```

## Start backend for demo

```bash
cd /Users/paulbobev/Projects/BRACE
./scripts/run_concussion_demo_server.sh
```

## Run on iPhone

1. Open `/Users/paulbobev/Projects/BRACE/mobile/ios/BraceTripodDemo/BraceTripodDemo.xcodeproj`.
2. Select your Apple Team in Signing.
3. Connect iPhone over USB and trust the device.
4. Build and run the `BraceTripodDemo` scheme.
5. Enter Tailscale URLs:
   - Upload base URL: `http://<your-tailnet-host-or-ip>:8443` (or `https://...` if you terminate TLS)
   - Live stream URL: leave blank for auto-derive, or set `ws://<your-tailnet-host-or-ip>:8443/live-stream`
6. Tap:
   - `Start Tripod Mode`
   - `Connect Live Stream`
   - `Start Synthetic Feed` for a fully automated demo without live keypoint producer.

The app will display upload queue depth, logs, and latest risk report.
