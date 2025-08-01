# ---------- Base image ------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1

# ---------- System & Python packages ----------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential cmake git curl libgl1 && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python wheels (Torch first) -------------------------------
RUN python3 -m pip install --no-cache-dir -U pip

# 1️⃣  PyTorch family from CUDA-12.1 index
RUN python3 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.3.1 \
      torchvision==0.18.1 \
      torchaudio==2.3.1

# 2️⃣  Back to default index for remaining deps
COPY requirements.txt /tmp/
RUN python3 -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      xformers==0.0.27 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# ---------- Pre-download model weights --------------------------------
RUN python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                  local_dir="/models/sdxl", revision="fp16", max_workers=8)
snapshot_download("InstantX/InstantID",
                  local_dir="/models/instantid", max_workers=8)
PY

ENV MODEL_DIR=/models

# ---------- Copy your service code ------------------------------------
WORKDIR /app
COPY app ./app
COPY ip_adapter ./ip_adapter
COPY pipeline_stable_diffusion_xl_instantid*.py ./
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["/bin/bash", "start.sh"]
