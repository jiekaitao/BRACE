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

FRONTEND_PID_FILE="$SCRIPT_DIR/.frontend.pid"

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
        # ── Guard: prevent starting if already running ──
        RUNNING_CONTAINERS=$(docker compose $COMPOSE_FILES --profile "$PROFILE" ps -q 2>/dev/null | head -1)
        if [ -n "$RUNNING_CONTAINERS" ]; then
            echo "[run] BRACE is already running! Use './run.sh down' first, or './run.sh logs' to view logs."
            exit 1
        fi

        echo "[run] Starting BRACE with profile=$PROFILE ..."

        if [ "$DEV_MODE" = true ]; then
            # Dev mode: run frontend locally for hot-reload, skip it in Docker
            docker compose $COMPOSE_FILES --profile "$PROFILE" up -d --scale frontend=0
        else
            docker compose $COMPOSE_FILES --profile "$PROFILE" up -d
        fi

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

        # ── Start local Next.js dev server in dev mode ──
        if [ "$DEV_MODE" = true ]; then
            echo ""
            echo "[run] Starting local Next.js dev server (HMR enabled)..."
            cd "$SCRIPT_DIR/frontend"
            npm run dev &
            FRONTEND_PID=$!
            echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
            cd "$SCRIPT_DIR"
            echo "[run] Next.js PID: $FRONTEND_PID (saved to .frontend.pid)"
        fi

        echo ""
        echo "[run] Services started:"
        if [ "$DEV_MODE" = true ]; then
            echo "  Frontend:  http://localhost:3000 (local, HMR enabled)"
        else
            echo "  Frontend:  http://localhost:3000 (Docker)"
        fi
        echo "  Backend:   http://localhost:8001"
        echo "  MongoDB:   mongodb://localhost:27017"
        echo ""
        echo "[run] Stop:      ./run.sh down"
        echo ""
        echo "[run] Tailing logs (Ctrl+C to detach — servers keep running)..."
        echo ""
        docker compose $COMPOSE_FILES --profile "$PROFILE" logs -f || true
        ;;
    down)
        # Kill local Next.js dev server if running
        if [ -f "$FRONTEND_PID_FILE" ]; then
            FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
            if kill -0 "$FRONTEND_PID" 2>/dev/null; then
                echo "[run] Stopping local Next.js dev server (PID $FRONTEND_PID)..."
                kill "$FRONTEND_PID" 2>/dev/null || true
            fi
            rm -f "$FRONTEND_PID_FILE"
        fi

        echo "[run] Stopping all Docker services..."
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
