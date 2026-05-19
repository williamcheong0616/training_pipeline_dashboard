# python:3.12 (full Debian Bookworm) ships with gcc/g++/make — no apt needed.
# CUDA 13.0 libraries are injected at runtime by the NVIDIA Container Toolkit
# (runtime: nvidia in docker-compose.yml) from the host driver installation.
FROM python:3.12

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg — copied from a static-binary image, no apt needed
COPY --from=mwader/static-ffmpeg:latest /ffmpeg  /usr/local/bin/ffmpeg
COPY --from=mwader/static-ffmpeg:latest /ffprobe /usr/local/bin/ffprobe

# Install PyTorch + torchvision from the cu130 wheel index first so that
# the subsequent requirements install does not overwrite them with a CPU build.
RUN pip install --upgrade pip \
    && pip install torch torchvision \
       --index-url https://download.pytorch.org/whl/cu130

# Install remaining Python dependencies (torch already satisfied, will be skipped).
# flash-attn is compiled here — MAX_JOBS limits parallel compilation to avoid OOM.
COPY requirements.txt .
RUN MAX_JOBS=4 pip install -r requirements.txt --no-build-isolation

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
