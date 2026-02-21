#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# BRACE — Full Native macOS Stack (MPS-accelerated)
#
# Launches MongoDB, VectorAI, Backend (FastAPI), and Frontend (Next.js)
# all natively on macOS with Apple Silicon GPU acceleration via MPS.
#
# Usage:
#   ./run_everything.sh --setup          # First-time: create venv, install deps
#   ./run_everything.sh                  # Start everything (foreground)
#   ./run_everything.sh --bg             # Start everything (background)
#   ./run_everything.sh --prod           # Production frontend (next build+start)
#   ./run_everything.sh --no-mongo       # Skip MongoDB
#   ./run_everything.sh --no-vectorai    # Skip VectorAI
#   ./run_everything.sh --model nano     # Use yolo11n-pose.pt
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV=".venv"
PID_DIR=".pids"
LOG_DIR=".logs"
PORT="${PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
YOLO_MODEL="${YOLO_MODEL:-yolo11m-pose.pt}"

# Flags
DO_SETUP=false
RUN_BG=false
PROD_MODE=false
SKIP_MONGO=false
SKIP_VECTORAI=false

# ─── Parse arguments ────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup)    DO_SETUP=true ;;
        --bg)       RUN_BG=true ;;
        --prod)     PROD_MODE=true ;;
        --no-mongo)    SKIP_MONGO=true ;;
        --no-vectorai) SKIP_VECTORAI=true ;;
        --model)
            shift
            case "${1:-medium}" in
                nano)   YOLO_MODEL="yolo11n-pose.pt" ;;
                medium) YOLO_MODEL="yolo11m-pose.pt" ;;
                large)  YOLO_MODEL="yolo11l-pose.pt" ;;
                xlarge) YOLO_MODEL="yolo11x-pose.pt" ;;
                *)      YOLO_MODEL="$1" ;;
            esac
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ─── Setup mode ─────────────────────────────────────────────
if $DO_SETUP; then
    echo "╔══════════════════════════════════════════╗"
    echo "║      BRACE Native Setup (macOS)          ║"
    echo "╚══════════════════════════════════════════╝"
    echo ""

    # Create venv
    if [ ! -d "$VENV" ]; then
        echo "[setup] Creating virtual environment..."
        python3 -m venv "$VENV"
    else
        echo "[setup] Virtual environment already exists."
    fi

    # Install PyTorch (standard — includes MPS support)
    echo "[setup] Installing PyTorch with MPS support..."
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch torchvision

    # Install brace package + backend deps
    echo "[setup] Installing brace package..."
    "$VENV/bin/pip" install -e .

    echo "[setup] Installing backend dependencies..."
    "$VENV/bin/pip" install -r backend/requirements.cpu.txt
    "$VENV/bin/pip" install onnxruntime

    # Frontend deps
    if [ -f "frontend/package.json" ]; then
        echo "[setup] Installing frontend dependencies..."
        (cd frontend && npm install)
    fi

    # Verify MPS
    echo ""
    echo "[setup] Checking MPS availability..."
    MPS_OK=$("$VENV/bin/python" -c "import torch; print(torch.backends.mps.is_available())" 2>&1)
    if [ "$MPS_OK" = "True" ]; then
        echo "[setup] ✓ MPS (Metal Performance Shaders) is available!"
    else
        echo "[setup] ✗ MPS not available. PyTorch will use CPU."
        echo "         (Requires macOS 12.3+ and Apple Silicon)"
    fi

    echo ""
    echo "[setup] Done! Run ./run_everything.sh to start the full stack."
    exit 0
fi

# ─── Pre-flight checks ─────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Error: Virtual environment not found."
    echo "Run:  ./run_everything.sh --setup"
    exit 1
fi

# Check for port conflicts
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo "⚠ Warning: Port $PORT is already in use."
    echo "  Docker stack may be running. Stop it first: ./run.sh down"
    echo "  Or use a different port: PORT=8002 ./run_everything.sh"
    exit 1
fi

if lsof -ti:$FRONTEND_PORT >/dev/null 2>&1; then
    echo "⚠ Warning: Port $FRONTEND_PORT is already in use."
    exit 1
fi

# ─── Load environment ──────────────────────────────────────
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Native defaults (override Docker-internal values)
export MONGODB_URI="${MONGODB_URI:-mongodb://localhost:27017/brace}"
export VECTORAI_HOST="${VECTORAI_HOST:-localhost}"
export VECTORAI_PORT="${VECTORAI_PORT:-5555}"
export YOLO_MODEL="$YOLO_MODEL"

# ─── Directories ────────────────────────────────────────────
mkdir -p "$PID_DIR" "$LOG_DIR"

# ─── Cleanup trap ───────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down..."

    # Kill backend
    if [ -f "$PID_DIR/backend.pid" ]; then
        kill "$(cat "$PID_DIR/backend.pid")" 2>/dev/null && echo "  Stopped backend"
        rm -f "$PID_DIR/backend.pid"
    fi

    # Kill frontend
    if [ -f "$PID_DIR/frontend.pid" ]; then
        kill "$(cat "$PID_DIR/frontend.pid")" 2>/dev/null && echo "  Stopped frontend"
        rm -f "$PID_DIR/frontend.pid"
    fi

    # Stop containers we started (not pre-existing ones)
    if [ -f "$PID_DIR/mongo-container.name" ]; then
        CNAME=$(cat "$PID_DIR/mongo-container.name")
        docker stop "$CNAME" >/dev/null 2>&1 && echo "  Stopped MongoDB container ($CNAME)"
        rm -f "$PID_DIR/mongo-container.name"
    fi

    if [ -f "$PID_DIR/vectorai-container.name" ]; then
        CNAME=$(cat "$PID_DIR/vectorai-container.name")
        docker stop "$CNAME" >/dev/null 2>&1 && echo "  Stopped VectorAI container ($CNAME)"
        rm -f "$PID_DIR/vectorai-container.name"
    fi

    echo "All services stopped."
}

trap cleanup EXIT INT TERM

# ─── Start MongoDB ──────────────────────────────────────────
start_mongo() {
    if $SKIP_MONGO; then
        echo "[mongo] Skipped (--no-mongo)"
        return
    fi

    # Check if mongod is already listening on 27017
    if lsof -ti:27017 >/dev/null 2>&1; then
        echo "[mongo] Already running on port 27017"
        return
    fi

    # Try existing Docker Compose container
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "brace-mongo-1"; then
        echo "[mongo] Starting existing container brace-mongo-1..."
        docker start brace-mongo-1 >/dev/null 2>&1
        # Wait for it to be ready
        for i in $(seq 1 10); do
            if lsof -ti:27017 >/dev/null 2>&1; then
                echo "[mongo] Ready (brace-mongo-1)"
                return
            fi
            sleep 1
        done
        # Compose container may not expose port — fall through
        echo "[mongo] brace-mongo-1 started but port 27017 not exposed, starting standalone..."
    fi

    # Start a new standalone container
    echo "[mongo] Starting standalone MongoDB container..."
    docker run -d \
        --name brace-mongo-native \
        -p 27017:27017 \
        -v brace-mongodb-native:/data/db \
        mongo:7.0 >/dev/null 2>&1 \
    || {
        # Container may already exist but be stopped
        docker start brace-mongo-native >/dev/null 2>&1 || {
            echo "[mongo] ERROR: Could not start MongoDB. Install Docker or run with --no-mongo"
            exit 1
        }
    }
    echo "brace-mongo-native" > "$PID_DIR/mongo-container.name"

    # Wait for it
    for i in $(seq 1 10); do
        if lsof -ti:27017 >/dev/null 2>&1; then
            echo "[mongo] Ready (brace-mongo-native)"
            return
        fi
        sleep 1
    done
    echo "[mongo] WARNING: MongoDB may not be ready yet"
}

# ─── Start VectorAI ────────────────────────────────────────
start_vectorai() {
    if $SKIP_VECTORAI; then
        echo "[vectorai] Skipped (--no-vectorai)"
        return
    fi

    # Check if already listening on 5555
    if lsof -ti:5555 >/dev/null 2>&1; then
        echo "[vectorai] Already running on port 5555"
        return
    fi

    # Try existing Docker Compose container
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "brace-vectorai-1"; then
        echo "[vectorai] Starting existing container brace-vectorai-1..."
        docker start brace-vectorai-1 >/dev/null 2>&1
        for i in $(seq 1 10); do
            if lsof -ti:5555 >/dev/null 2>&1; then
                echo "[vectorai] Ready (brace-vectorai-1)"
                return
            fi
            sleep 1
        done
        echo "[vectorai] brace-vectorai-1 started but port 5555 not reachable, starting standalone..."
    fi

    # Start a new standalone container
    echo "[vectorai] Starting standalone VectorAI container..."
    docker run -d \
        --name brace-vectorai-native \
        -p 5555:50051 \
        -v brace-vectorai-native:/data \
        williamimoh/actian-vectorai-db:1.0b >/dev/null 2>&1 \
    || {
        docker start brace-vectorai-native >/dev/null 2>&1 || {
            echo "[vectorai] WARNING: Could not start VectorAI. Continuing without it."
            return
        }
    }
    echo "brace-vectorai-native" > "$PID_DIR/vectorai-container.name"

    for i in $(seq 1 10); do
        if lsof -ti:5555 >/dev/null 2>&1; then
            echo "[vectorai] Ready (brace-vectorai-native)"
            return
        fi
        sleep 1
    done
    echo "[vectorai] WARNING: VectorAI may not be ready yet"
}

# ─── Start Backend ──────────────────────────────────────────
start_backend() {
    echo "[backend] Starting on port $PORT..."
    cd "$SCRIPT_DIR/backend"

    PIPELINE_BACKEND="${PIPELINE_BACKEND:-legacy}" \
    YOLO_MODEL="$YOLO_MODEL" \
    DISABLE_REID="${DISABLE_REID:-0}" \
    MONGODB_URI="$MONGODB_URI" \
    VECTORAI_HOST="$VECTORAI_HOST" \
    VECTORAI_PORT="$VECTORAI_PORT" \
    GOOGLE_GEMINI_API_KEY="${GOOGLE_GEMINI_API_KEY:-}" \
    ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-}" \
    "$SCRIPT_DIR/$VENV/bin/uvicorn" main:app --host 0.0.0.0 --port "$PORT" \
        > "$SCRIPT_DIR/$LOG_DIR/backend.log" 2>&1 &

    echo $! > "$SCRIPT_DIR/$PID_DIR/backend.pid"
    cd "$SCRIPT_DIR"
    echo "[backend] PID $(cat "$PID_DIR/backend.pid")"
}

# ─── Start Frontend ─────────────────────────────────────────
start_frontend() {
    echo "[frontend] Starting on port $FRONTEND_PORT..."
    cd "$SCRIPT_DIR/frontend"

    if $PROD_MODE; then
        npx next build > "$SCRIPT_DIR/$LOG_DIR/frontend.log" 2>&1
        npx next start --port "$FRONTEND_PORT" \
            >> "$SCRIPT_DIR/$LOG_DIR/frontend.log" 2>&1 &
    else
        npx next dev --port "$FRONTEND_PORT" \
            > "$SCRIPT_DIR/$LOG_DIR/frontend.log" 2>&1 &
    fi

    echo $! > "$SCRIPT_DIR/$PID_DIR/frontend.pid"
    cd "$SCRIPT_DIR"
    echo "[frontend] PID $(cat "$PID_DIR/frontend.pid")"
}

# ─── Wait for backend health ───────────────────────────────
wait_for_backend() {
    echo ""
    echo "Waiting for backend to be ready (model loading may take a moment)..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "[backend] Ready!"
            return 0
        fi
        # Check if process is still alive
        if [ -f "$PID_DIR/backend.pid" ] && ! kill -0 "$(cat "$PID_DIR/backend.pid")" 2>/dev/null; then
            echo "[backend] ERROR: Process exited. Check $LOG_DIR/backend.log"
            tail -20 "$LOG_DIR/backend.log" 2>/dev/null
            exit 1
        fi
        sleep 1
    done
    echo "[backend] WARNING: Not responding after 30s. Check $LOG_DIR/backend.log"
}

# ─── Main ───────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════╗"
echo "║    BRACE Native macOS Stack (MPS)            ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Model:     $YOLO_MODEL"
echo "║  Pipeline:  ${PIPELINE_BACKEND:-legacy}"
echo "║  Backend:   http://localhost:$PORT"
echo "║  Frontend:  http://localhost:$FRONTEND_PORT"
echo "╚══════════════════════════════════════════════╝"
echo ""

start_mongo
start_vectorai
echo ""
start_backend
start_frontend

wait_for_backend

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  All services running!                       ║"
echo "║                                              ║"
echo "║  Frontend:  http://localhost:$FRONTEND_PORT         ║"
echo "║  Backend:   http://localhost:$PORT              ║"
echo "║  WebSocket: ws://localhost:$PORT/ws             ║"
echo "║                                              ║"
echo "║  Logs: $LOG_DIR/backend.log                  ║"
echo "║        $LOG_DIR/frontend.log                 ║"
echo "║                                              ║"
echo "║  Press Ctrl+C to stop everything             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

if $RUN_BG; then
    echo "Running in background. Use ./stop_everything.sh to stop."
    # Disown children so they survive shell exit
    disown
    trap - EXIT INT TERM
    exit 0
fi

# Foreground: tail backend log to keep alive
tail -f "$LOG_DIR/backend.log"
