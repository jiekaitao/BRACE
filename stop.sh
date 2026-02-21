#!/usr/bin/env bash
# Stop all BRACE services and tear down Tailscale Funnel.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[stop] Stopping all Docker services..."
docker compose --profile nvidia --profile cpu down

if command -v tailscale &>/dev/null; then
    echo "[stop] Tearing down Tailscale Funnel..."
    tailscale funnel --https=443 off 2>/dev/null || true
    tailscale funnel --https=8443 off 2>/dev/null || true
fi

echo "[stop] Done."
