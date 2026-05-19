# nvidia/cuda:13.0.1-devel-ubuntu24.04 ships gcc/g++/make/nvcc and Python 3.12 natively.
# CUDA 13.0 runtime libraries are also injected at runtime by the NVIDIA Container Toolkit.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Install Python 3.12 (ships natively in Ubuntu 24.04 — no PPA needed).
# "|| true" tolerates transient Ubuntu CDN hash-mismatch errors on multiverse/universe;
# python3.12 lives in main which always fetches cleanly.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* \
    && (apt-get update || true) \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3-pip \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg — copied from a static-binary image, no apt needed
COPY --from=mwader/static-ffmpeg:latest /ffmpeg  /usr/local/bin/ffmpeg
COPY --from=mwader/static-ffmpeg:latest /ffprobe /usr/local/bin/ffprobe

# Install PyTorch + torchvision from the cu126 wheel index first so that
# the subsequent requirements install does not overwrite them with a CPU build.
RUN pip install --upgrade pip \
    && pip install torch torchvision \
       --index-url https://download.pytorch.org/whl/cu130

# Install remaining Python dependencies (torch already satisfied, will be skipped)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Compile and install flash-attn using the nvcc that ships in this image.
# --no-build-isolation ensures the already-installed torch/numpy are visible.
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Bake application code into the image (no dev bind-mounts in production)
COPY . ./backend

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash forge \
    && chown -R forge:forge /app
USER forge

EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info", "--access-log"]
