# Stage 1: CUDA 13.0 development toolkit (nvcc, headers, libraries).
# Used only to supply /usr/local/cuda — we never run apt-get here.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS cuda-devel

# Stage 2: Runtime image — python:3.12 (Debian Bookworm, stable mirrors).
# We copy /usr/local/cuda from Stage 1 so nvcc is available at build time
# to compile flash-attn, without touching Ubuntu's broken apt repos.
FROM python:3.12

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Copy the CUDA toolkit (nvcc + headers + libs) from the devel stage.
COPY --from=cuda-devel /usr/local/cuda /usr/local/cuda

# ffmpeg — copied from a static-binary image, no apt needed
COPY --from=mwader/static-ffmpeg:latest /ffmpeg  /usr/local/bin/ffmpeg
COPY --from=mwader/static-ffmpeg:latest /ffprobe /usr/local/bin/ffprobe

# Install PyTorch + torchvision from the cu130 wheel index first so that
# the subsequent requirements install does not overwrite them with a CPU build.
RUN pip install --upgrade pip \
    && pip install torch torchvision \
       --index-url https://download.pytorch.org/whl/cu130

# Install remaining Python dependencies (torch already satisfied, will be skipped)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Compile and install flash-attn using the nvcc copied from the devel stage.
# --no-build-isolation ensures the already-installed torch/numpy are visible.
RUN pip install packaging setuptools wheel \
    && MAX_JOBS=4 pip install flash-attn --no-build-isolation

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
