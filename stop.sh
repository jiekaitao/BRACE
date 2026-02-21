#!/usr/bin/env bash
# Stop all BRACE services, Cloudflare Tunnel, and Tailscale Funnel.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[stop] Stopping all Docker services..."
docker compose --profile nvidia --profile cpu down

if pkill -f "cloudflared tunnel run" 2>/dev/null; then
    echo "[stop] Stopped Cloudflare Tunnel."
fi

if command -v tailscale &>/dev/null; then
    tailscale funnel reset 2>/dev/null && echo "[stop] Reset Tailscale Funnel." || true
fi

echo "[stop] Done."
