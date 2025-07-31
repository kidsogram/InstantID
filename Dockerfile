FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
COPY requirements.txt /tmp/
RUN pip install -U pip && pip install -r /tmp/requirements.txt

# ⬇️  grab the weights at build-time so cold-start is instant
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                  local_dir="/models/sdxl", revision="fp16", max_workers=8)
snapshot_download("InstantX/InstantID", local_dir="/models/instantid", max_workers=8)
PY

ENV MODEL_DIR=/models

WORKDIR /app
COPY app ./app
COPY start.sh .

EXPOSE 8000
CMD ["/bin/bash", "start.sh"]
