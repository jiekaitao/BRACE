#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: run_concussion_demo_server.sh

Environment overrides:
  HOST=<bind host>          (default: 0.0.0.0)
  PORT=<bind port>          (default: 8443)
  WORKERS=<uvicorn workers> (default: 1)
  LOG_LEVEL=<log level>     (default: info)
USAGE
  exit 0
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8443}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "Starting BRACE concussion demo server on ${HOST}:${PORT}"
echo "Endpoints:"
echo "  - POST /upload-clip"
echo "  - WS   /live-stream"
echo "  - GET  /health"
echo

if command -v tailscale >/dev/null 2>&1; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
  if [[ -n "$TS_IP" ]]; then
    echo "Tailscale URL hints:"
    echo "  - Upload base URL: http://${TS_IP}:${PORT}"
    echo "  - Live stream URL: ws://${TS_IP}:${PORT}/live-stream"
    echo
  fi
fi

exec python -m uvicorn backend.concussion_demo_app:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --log-level "$LOG_LEVEL"
