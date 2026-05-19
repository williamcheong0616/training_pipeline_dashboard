# ── Stage 1: dependency install ───────────────────────────────────────────────
# Using CUDA 12.6 runtime as the base; update this tag once NVIDIA publishes
# official CUDA 13.0 images and the cu130 PyTorch wheels are stable.
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 AS base

# Install Python 3.12 + minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3-pip python3.12-venv \
        curl git \
    && ln -sf python3.12 /usr/bin/python3 \
    && ln -sf python3    /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Stage 2: install Python packages (cached layer) ───────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Stage 3: final image ──────────────────────────────────────────────────────
FROM deps AS runner

# Bake application code into the image (no dev bind-mounts in production)
COPY . ./backend

# Non-root user for safety
RUN useradd -m -u 1000 -s /bin/bash forge \
    && chown -R forge:forge /app
USER forge

EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info", "--access-log"]
