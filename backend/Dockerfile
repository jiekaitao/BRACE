FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    libgl1 libglib2.0-0 curl git gcc g++ && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

# Install PyTorch with CUDA 12.8 first (biggest layer, cache-friendly)
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install brace package (core library)
COPY setup.py ./
COPY brace/ ./brace/
RUN pip install --no-cache-dir --break-system-packages -e .

# Install backend dependencies (includes rtmlib, transformers)
COPY backend/requirements.txt ./backend_requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r backend_requirements.txt

# Fix onnxruntime: ensure GPU version replaces CPU version (they conflict)
RUN pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null; \
    pip install --no-cache-dir --break-system-packages onnxruntime-gpu

# Try to install simple-romp for SMPL estimation (optional, may fail)
RUN pip install --no-cache-dir --break-system-packages simple-romp 2>/dev/null || \
    echo "[docker] simple-romp install failed (optional), SMPL estimation will be disabled"

# Download YOLO11-pose models and export TensorRT FP16 engines (one-time build cost)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt'); YOLO('yolo11x-pose.pt')" && \
    python -c "\
from ultralytics import YOLO; \
import torch; \
if torch.cuda.is_available(): \
    YOLO('yolo11m-pose.pt').export(format='engine', half=True); \
    YOLO('yolo11x-pose.pt').export(format='engine', half=True); \
else: print('No GPU — skipping TensorRT export') \
" || echo "[docker] TensorRT export failed (will retry at startup)"

# Pre-download CLIP model for cross-cut re-ID (speeds up first inference)
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch16'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16'); print('[docker] CLIP model pre-downloaded')" || echo "[docker] CLIP pre-download failed (optional)"

# Copy backend source
COPY backend/ ./

RUN mkdir -p /app/uploads

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
