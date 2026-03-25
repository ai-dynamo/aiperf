# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.13-slim-bookworm AS base

ENV USERNAME=appuser
ENV APP_NAME=aiperf

# Create app user
RUN groupadd -r $USERNAME \
    && useradd -r -g $USERNAME $USERNAME

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create virtual environment
RUN mkdir -p /opt/$APP_NAME \
    && uv venv /opt/$APP_NAME/venv --python 3.13 --clear \
    && chown -R $USERNAME:$USERNAME /opt/$APP_NAME

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/$APP_NAME/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

#######################################
########## Local Development ##########
#######################################

FROM base AS local-dev

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# Will use the default aiperf user, but give sudo access
# Needed so files permissions aren't set to root ownership when writing from inside container

# Don't want username to be editable, just allow changing the uid and gid.
# Username is hardcoded in .devcontainer
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update -y \
    && apt-get install -y sudo gnupg2 gnupg1 \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/$USERNAME \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME \
    && chsh -s /bin/bash $USERNAME

# Install some useful tools for local development
RUN apt-get update -y \
    && apt-get install -y tmux vim git curl procps make

USER $USERNAME
ENV HOME=/home/$USERNAME
WORKDIR $HOME

# https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=$HOME/.commandhistory/.bash_history" \
    && mkdir -p $HOME/.commandhistory \
    && touch $HOME/.commandhistory/.bash_history \
    && echo "$SNIPPET" >> "$HOME/.bashrc"

RUN mkdir -p /home/$USERNAME/.cache/

ENTRYPOINT ["/bin/bash"]

############################################
############ Wheel Builder #################
############################################
FROM base AS wheel-builder

WORKDIR /workspace

# Copy the entire application
COPY pyproject.toml README.md LICENSE ATTRIBUTIONS.md ./src/ /workspace/

# Build the wheel
RUN uv build --wheel --out-dir /dist

############################################
############# Env Builder ##################
############################################
FROM base AS env-builder

WORKDIR /workspace

# NOTE: ffmpeg build disabled (nopw)
# # Build ffmpeg from source with libvpx
# RUN apt-get update -y && \
#     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#         build-essential \
#         nasm \
#         pkg-config \
#         wget \
#         yasm \
#         libvpx-dev \
#     && rm -rf /var/lib/apt/lists/*
#
# # Download and build ffmpeg with libvpx (VP9 codec)
# RUN wget https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz \
#     && tar -xf ffmpeg-8.0.1.tar.xz \
#     && cd ffmpeg-8.0.1 \
#     && ./configure \
#         --prefix=/opt/ffmpeg \
#         --disable-gpl \
#         --disable-nonfree \
#         --enable-shared \
#         --disable-static \
#         --enable-libvpx \
#         --disable-doc \
#         --disable-htmlpages \
#         --disable-manpages \
#         --disable-podpages \
#         --disable-txtpages \
#     && make -j$(nproc) \
#     && make install \
#     && cd .. \
#     && rm -rf ffmpeg-8.0.1 ffmpeg-8.0.1.tar.xz \
#     && cp -P /usr/lib/*/libvpx.so* /opt/ffmpeg/lib/ 2>/dev/null || \
#        cp -P /usr/lib/libvpx.so* /opt/ffmpeg/lib/ 2>/dev/null || { echo "Error: libvpx.so not found"; exit 1; }
#
# ENV PATH="/opt/ffmpeg/bin${PATH:+:${PATH}}" \
#     LD_LIBRARY_PATH="/opt/ffmpeg/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Minimal tools needed for cache downloads in this stage.
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Create directories for the nvs user (UID 1000 in NVIDIA distroless)
RUN mkdir -p /app /app/artifacts /app/.cache \
    && chown -R 1000:1000 /app \
    && chmod -R 755 /app

# Pre-cache tiktoken o200k_base encoding for --tokenizer builtin (MIT license, see ATTRIBUTIONS.md)
RUN mkdir -p /opt/tiktoken_cache \
    && CACHE_KEY=$(python -c "import hashlib; print(hashlib.sha1(b'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken').hexdigest())") \
    && wget -q -O /opt/tiktoken_cache/$CACHE_KEY \
       https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken

# Install only the dependencies using uv
COPY pyproject.toml .
RUN uv sync --active --no-install-project

# Copy the rest of the application
COPY --from=wheel-builder /dist /dist
RUN uv pip install /dist/aiperf-*.whl \
    && rm -rf /dist /workspace/pyproject.toml

# Remove setuptools as it is not needed for the runtime image
RUN uv pip uninstall setuptools

# Pre-cache the gpt2 tokenizer so K8s pods don't need network access at startup.
# This avoids 20-40s download delays that can kill ZMQ connections between pods.
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

############################################
############### Test Image #################
############################################
# Test stage: env-builder has aiperf, just add curl
FROM env-builder AS test

RUN apt-get update -y && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/aiperf/venv \
    PATH="/opt/aiperf/venv/bin:${PATH}" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

ENTRYPOINT ["/bin/bash", "-c"]

############################################
############# Runtime Image ################
############################################
FROM nvcr.io/nvidia/distroless/python:3.13-v4.0.1-dev AS runtime

# Include license and attribution files
COPY LICENSE ATTRIBUTIONS*.md /legal/

# Copy bash with executable permissions preserved using --chmod
COPY --from=env-builder --chown=1000:1000 --chmod=755 /bin/bash /bin/bash

# NOTE: ffmpeg build disabled (nopw)
# # Copy ffmpeg binaries and libraries (includes libvpx)
# COPY --from=env-builder --chown=1000:1000 /opt/ffmpeg /opt/ffmpeg
# ENV PATH="/opt/ffmpeg/bin${PATH:+:${PATH}}" \
#     LD_LIBRARY_PATH="/opt/ffmpeg/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Setup the directories with permissions for nvs user
COPY --from=env-builder --chown=1000:1000 /app /app
WORKDIR /app
ENV HOME=/app

# Copy the virtual environment and set up
COPY --from=env-builder --chown=1000:1000 /opt/aiperf/venv /opt/aiperf/venv

# Copy pre-cached tiktoken encoding for zero-network --tokenizer builtin
COPY --from=env-builder --chown=1000:1000 /opt/tiktoken_cache /opt/tiktoken_cache

# Copy pre-cached HuggingFace tokenizers so pods start without network access
COPY --from=env-builder --chown=1000:1000 /root/.cache/huggingface /app/.cache/huggingface

ENV VIRTUAL_ENV=/opt/aiperf/venv \
    PATH="/opt/aiperf/venv/bin:${PATH}" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

# Set bash as entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
