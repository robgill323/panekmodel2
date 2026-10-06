# Throughline — CPU baseline image.
#
# NOT PUBLISHED, BY CONSTRUCTION. No registry appears in this file or in
# docker-compose.yml, there is no push step and no CI workflow builds it. It is
# built on the machine that runs it:
#
#     docker compose build
#
# The target rig (OS, GPU, driver, network) is not known, so nothing here
# assumes one. torch comes from the CPU wheel index by default; a CUDA build is
# a one-argument override (docker-compose.gpu.yml), UNTESTED-PENDING-RIG.
#
# Honest limit: whether this image builds and runs has to be established on a
# machine with a running Docker daemon. See docs/DEPLOYMENT.md §11.
#
# SECRETS: none enter any layer. THROUGHLINE_PASSWORD, YOUTUBE_API_KEY and
# HF_TOKEN arrive at RUN time from the compose environment; no ARG or ENV
# names them, .env is excluded by .dockerignore, and the only COPY sources are
# listed explicitly below (never `COPY . .`).

FROM python:3.12-slim

# ffmpeg is required by the Whisper fallback. It is off by default, but the
# image should not be the reason it cannot be switched on.
RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Never root. A real home directory matters: the transcript cache lives at
# ~/.panekmodel2_cache and model weights under ~/.cache, and those are what the
# compose volumes mount.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin throughline

# NLTK downloads to ~/nltk_data by default — outside the model-cache volume,
# so it would be re-fetched on every container recreate. An existing, writable
# NLTK_DATA directory is used as the download target instead.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    NLTK_DATA=/home/throughline/.cache/nltk_data

# The volume mount points must exist, owned by the runtime user, BEFORE the
# volumes are first attached: Docker seeds an empty named volume from the
# image's directory, ownership included. Without this they would be created
# root-owned and the non-root process could not write its own cache.
RUN install -d -o throughline -g throughline \
        /home/throughline/.panekmodel2_cache \
        /home/throughline/.cache \
        /home/throughline/.cache/nltk_data \
        /home/throughline/exports

WORKDIR /app

# ── torch, from whichever index the build selects ───────────────────
# Installed before the project so the project install finds the requirement
# already satisfied, rather than pulling the default PyPI build — which on
# Linux is the CUDA build plus ~2 GB of nvidia-* wheels a CPU baseline cannot
# use.
#
#   CPU (default): https://download.pytorch.org/whl/cpu
#   CUDA:          https://download.pytorch.org/whl/cu1XX — must match the
#                  host driver, which is a rig decision, so it is an argument.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "${TORCH_INDEX_URL}" torch

# ── the project ─────────────────────────────────────────────────────
# Explicit sources only. src/ carries the SPA under server/static.
COPY pyproject.toml README.md ./
COPY src ./src

# Empty (the default) installs from the version floors in pyproject.toml. That
# builds anywhere, which makes it the baseline, but it is NOT reproducible.
#
# For a reproducible image, generate a Linux lock with
# `scripts/lock.sh --docker` and build with
# REQUIREMENTS_FILE=requirements-docker.lock. The committed requirements.lock
# was resolved on macOS and will not satisfy --require-hashes on Linux.
ARG REQUIREMENTS_FILE=""
COPY requirements*.lock ./
RUN set -eux; \
    if [ -n "${REQUIREMENTS_FILE}" ]; then \
        pip install --require-hashes -r "${REQUIREMENTS_FILE}"; \
        pip install --no-deps .; \
    else \
        echo "NOTE: installing from pyproject floors - this image is not reproducible." >&2; \
        pip install .; \
    fi

COPY docker/healthcheck.py /usr/local/bin/throughline-healthcheck

USER throughline
WORKDIR /home/throughline
EXPOSE 8000

# Reads THROUGHLINE_PASSWORD from the environment rather than taking it as an
# argument, so it never appears in `ps` output or `docker inspect`. Exec form:
# no shell expands anything. start-period covers the first boot.
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD ["python", "/usr/local/bin/throughline-healthcheck"]

# 0.0.0.0 is correct INSIDE the container — the host-side port publish in
# docker-compose.yml decides real reachability. Because it is not loopback,
# the server refuses to start unless THROUGHLINE_PASSWORD is set: the
# enforcement, not a convention.
CMD ["panekmodel2", "ui", "--host", "0.0.0.0", "--port", "8000", "--no-open"]
