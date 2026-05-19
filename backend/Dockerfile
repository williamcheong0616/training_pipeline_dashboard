# pytorch/pytorch already includes Python, pip, CUDA, and cuDNN — no apt needed.
# Update the tag here when a cu130-based image is published.
# Current: CUDA 12.4 + cuDNN 9 (closest available to cu130 requirements).
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system deps only (no Python install needed — it's in the base image)
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer — rebuilds only when requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

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
