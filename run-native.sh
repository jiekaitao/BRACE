#!/usr/bin/env bash
# Run BRACE backend natively on Mac (no Docker required for backend).
# Uses MPS (Apple Silicon) or CPU for inference.
#
# Usage:
#   ./run-native.sh                 # Start backend on port 8001
#   ./run-native.sh --setup         # First-time setup (create venv, install deps)
#   ./run-native.sh --model nano    # Use nano model (6MB, faster, less accurate)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV=".venv"
PORT="${PORT:-8001}"
YOLO_MODEL="${YOLO_MODEL:-yolo11m-pose.pt}"
PIPELINE="${PIPELINE_BACKEND:-legacy}"

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --setup)
            echo "[setup] Creating virtual environment..."
            python3 -m venv "$VENV"
            echo "[setup] Installing PyTorch with MPS support..."
            "$VENV/bin/pip" install torch torchvision
            echo "[setup] Installing brace package..."
            "$VENV/bin/pip" install -e .
            echo "[setup] Installing backend dependencies..."
            "$VENV/bin/pip" install -r backend/requirements.cpu.txt
            "$VENV/bin/pip" install onnxruntime
            echo "[setup] Done! Run ./run-native.sh to start the backend."
            exit 0
            ;;
        --model)
            shift
            YOLO_MODEL="${1:-yolo11n-pose.pt}"
            ;;
        nano)
            YOLO_MODEL="yolo11n-pose.pt"
            ;;
    esac
done

if [ ! -d "$VENV" ]; then
    echo "Virtual environment not found. Run: ./run-native.sh --setup"
    exit 1
fi

echo "╔══════════════════════════════════════════╗"
echo "║       BRACE Native Mac Launcher          ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Model:    $YOLO_MODEL"
echo "║  Pipeline: $PIPELINE"
echo "║  Port:     $PORT"
echo "╚══════════════════════════════════════════╝"
echo ""

cd "$SCRIPT_DIR/backend"

PIPELINE_BACKEND="$PIPELINE" \
YOLO_MODEL="$YOLO_MODEL" \
DISABLE_REID="${DISABLE_REID:-0}" \
MONGODB_URI="${MONGODB_URI:-mongodb://localhost:27017/brace}" \
GOOGLE_GEMINI_API_KEY="${GOOGLE_GEMINI_API_KEY:-}" \
exec "$SCRIPT_DIR/$VENV/bin/uvicorn" main:app --host 0.0.0.0 --port "$PORT"
