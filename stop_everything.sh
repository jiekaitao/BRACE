#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# BRACE — Stop all native services
#
# Usage:
#   ./stop_everything.sh          # Stop backend, frontend, containers we started
#   ./stop_everything.sh --all    # Also stop MongoDB brew service
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_DIR=".pids"
STOP_ALL=false

for arg in "$@"; do
    case "$arg" in
        --all) STOP_ALL=true ;;
    esac
done

STOPPED=0

# ─── Stop backend ──────────────────────────────────────────
if [ -f "$PID_DIR/backend.pid" ]; then
    PID=$(cat "$PID_DIR/backend.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo "Stopped backend (PID $PID)"
        STOPPED=$((STOPPED + 1))
    else
        echo "Backend (PID $PID) was not running"
    fi
    rm -f "$PID_DIR/backend.pid"
else
    echo "No backend PID file found"
fi

# ─── Stop frontend ─────────────────────────────────────────
if [ -f "$PID_DIR/frontend.pid" ]; then
    PID=$(cat "$PID_DIR/frontend.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo "Stopped frontend (PID $PID)"
        STOPPED=$((STOPPED + 1))
    else
        echo "Frontend (PID $PID) was not running"
    fi
    rm -f "$PID_DIR/frontend.pid"
else
    echo "No frontend PID file found"
fi

# ─── Stop containers we started ────────────────────────────
if [ -f "$PID_DIR/mongo-container.name" ]; then
    CNAME=$(cat "$PID_DIR/mongo-container.name")
    if docker stop "$CNAME" >/dev/null 2>&1; then
        echo "Stopped MongoDB container ($CNAME)"
        STOPPED=$((STOPPED + 1))
    else
        echo "MongoDB container ($CNAME) was not running"
    fi
    rm -f "$PID_DIR/mongo-container.name"
else
    echo "No MongoDB container to stop (not started by run_everything.sh)"
fi

if [ -f "$PID_DIR/vectorai-container.name" ]; then
    CNAME=$(cat "$PID_DIR/vectorai-container.name")
    if docker stop "$CNAME" >/dev/null 2>&1; then
        echo "Stopped VectorAI container ($CNAME)"
        STOPPED=$((STOPPED + 1))
    else
        echo "VectorAI container ($CNAME) was not running"
    fi
    rm -f "$PID_DIR/vectorai-container.name"
else
    echo "No VectorAI container to stop (not started by run_everything.sh)"
fi

# ─── --all: stop brew MongoDB too ──────────────────────────
if $STOP_ALL; then
    if brew services list 2>/dev/null | grep -q "mongodb.*started"; then
        brew services stop mongodb-community 2>/dev/null && echo "Stopped MongoDB brew service"
    fi
fi

# ─── Clean up PID dir if empty ─────────────────────────────
rmdir "$PID_DIR" 2>/dev/null || true

echo ""
echo "Stopped $STOPPED service(s)."
