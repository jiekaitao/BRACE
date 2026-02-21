#!/usr/bin/env bash
# Auto-detect hardware and launch BRACE with the right Docker Compose profile.
#
# Usage:
#   ./run.sh              # auto-detect, start in background
#   ./run.sh --dev        # auto-detect, use dev compose (hot-reload)
#   ./run.sh --nvidia     # force NVIDIA GPU profile
#   ./run.sh --cpu        # force CPU profile
#   ./run.sh down         # stop everything

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse arguments ──
DEV_MODE=false
FORCE_PROFILE=""
ACTION="up"

for arg in "$@"; do
    case "$arg" in
        --dev)     DEV_MODE=true ;;
        --nvidia)  FORCE_PROFILE="nvidia" ;;
        --cpu)     FORCE_PROFILE="cpu" ;;
        down|stop) ACTION="down" ;;
        logs)      ACTION="logs" ;;
        build)     ACTION="build" ;;
    esac
done

# ── Auto-detect hardware profile ──
detect_profile() {
    if [ -n "$FORCE_PROFILE" ]; then
        echo "$FORCE_PROFILE"
        return
    fi

    # Check if running on macOS
    if [ "$(uname)" = "Darwin" ]; then
        echo "cpu"
        return
    fi

    # Check for NVIDIA GPU on Linux
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "nvidia"
        return
    fi

    # Check if nvidia-docker runtime is available
    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo "nvidia"
        return
    fi

    # Default to CPU
    echo "cpu"
}

PROFILE=$(detect_profile)

# ── Select compose file ──
if [ "$DEV_MODE" = true ]; then
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.dev.yml"
else
    COMPOSE_FILES="-f docker-compose.yml"
fi

echo "╔══════════════════════════════════════════╗"
echo "║           BRACE Launcher                 ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Profile:  $PROFILE"
echo "║  Mode:     $([ "$DEV_MODE" = true ] && echo "development" || echo "production")"
echo "║  Action:   $ACTION"
echo "╚══════════════════════════════════════════╝"
echo ""

case "$ACTION" in
    up)
        echo "[run] Starting BRACE with profile=$PROFILE ..."
        docker compose $COMPOSE_FILES --profile "$PROFILE" up -d

        # ── Expose via Cloudflare Tunnel (braceml.com) ──
        if command -v cloudflared &>/dev/null; then
            # Kill any existing tunnel process
            pkill -f "cloudflared tunnel run" 2>/dev/null || true
            sleep 1
            echo "[run] Starting Cloudflare Tunnel..."
            cloudflared tunnel run brace >> /tmp/cloudflared.log 2>&1 &
            echo "[run]   https://braceml.com → frontend"
            echo "[run]   https://ws.braceml.com → backend (WebSocket)"
        fi

        # ── Tailscale Funnel (tailnet fallback) ──
        if command -v tailscale &>/dev/null; then
            TS_HOST=$(tailscale status --self --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null) || true
            if [ -n "$TS_HOST" ]; then
                tailscale funnel --bg --https=443 http://localhost:3000 2>/dev/null && \
                    echo "[run]   Funnel → https://$TS_HOST/" || true
            fi
        fi

        echo ""
        echo "[run] Services started:"
        echo "  Frontend:  http://localhost:3000"
        echo "  Backend:   http://localhost:8001"
        echo "  MongoDB:   mongodb://localhost:27017"
        echo ""
        echo "[run] View logs: ./run.sh logs"
        echo "[run] Stop:      ./run.sh down"
        ;;
    down)
        echo "[run] Stopping all services..."
        docker compose $COMPOSE_FILES --profile nvidia --profile cpu down
        ;;
    logs)
        docker compose $COMPOSE_FILES --profile "$PROFILE" logs -f
        ;;
    build)
        echo "[run] Building with profile=$PROFILE ..."
        docker compose $COMPOSE_FILES --profile "$PROFILE" build
        ;;
esac
