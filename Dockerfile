# check=skip=UndefinedVar
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.13-slim-bookworm@sha256:061b6e52a07ab675f0e4a9428c5a8ee6bed996983427f4691f6bebf29c56d9dc AS base

ENV USERNAME=appuser
ENV APP_NAME=aiperf

# Create app user
RUN groupadd -r $USERNAME \
    && useradd -r -g $USERNAME $USERNAME

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create virtual environment
RUN mkdir /opt/$APP_NAME \
    && uv venv /opt/$APP_NAME/venv --python 3.13 \
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

# Export-only stage: scratch-based so `docker buildx build --target
# wheel-artifact --output type=local,dest=<dir>` writes only the wheel file
# (a few MB) instead of the ~400 MB wheel-builder filesystem.
FROM scratch AS wheel-artifact
COPY --from=wheel-builder /dist/ /

############################################
############# Env Builder ##################
############################################
FROM base AS env-builder

WORKDIR /workspace

# Install build dependencies. The dpkg-installed.txt snapshot was dropped:
# nothing downstream consumes it, and shipping it alongside runtime-pkgs.txt
# in the artifact was misleading (build-only packages that never ship).
RUN mkdir -p /opt/licenses/dpkg \
    && apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        libogg-dev \
        libvorbis-dev \
        libvpx-dev \
        nasm \
        pkg-config \
        wget \
        yasm \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and build ffmpeg with libvpx (VP9 codec)
ARG FFMPEG_VERSION=8.0.1
RUN wget https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz \
    && tar -xf ffmpeg-${FFMPEG_VERSION}.tar.xz \
    && cd ffmpeg-${FFMPEG_VERSION} \
    && ./configure \
        --prefix=/opt/ffmpeg \
        --disable-gpl \
        --disable-nonfree \
        --enable-shared \
        --disable-static \
        --enable-libvorbis \
        --enable-libvpx \
        --disable-doc \
        --disable-htmlpages \
        --disable-manpages \
        --disable-podpages \
        --disable-txtpages \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && mkdir -p /opt/licenses/ffmpeg \
    && cp ffmpeg-${FFMPEG_VERSION}/COPYING.LGPLv2.1 /opt/licenses/ffmpeg/ \
    && cp ffmpeg-${FFMPEG_VERSION}/LICENSE.md /opt/licenses/ffmpeg/ \
    && rm -rf ffmpeg-${FFMPEG_VERSION} ffmpeg-${FFMPEG_VERSION}.tar.xz \
    && cp -P /usr/lib/*/libvpx.so* /opt/ffmpeg/lib/ 2>/dev/null || \
       cp -P /usr/lib/libvpx.so* /opt/ffmpeg/lib/ 2>/dev/null || { echo "Error: libvpx.so not found"; exit 1; } \
    && cp -P /usr/lib/*/libvorbis.so* /usr/lib/*/libvorbisenc.so* /opt/ffmpeg/lib/ 2>/dev/null || \
       cp -P /usr/lib/libvorbis.so* /usr/lib/libvorbisenc.so* /opt/ffmpeg/lib/ 2>/dev/null || { echo "Error: libvorbis.so not found"; exit 1; } \
    && cp -P /usr/lib/*/libogg.so* /opt/ffmpeg/lib/ 2>/dev/null || \
       cp -P /usr/lib/libogg.so* /opt/ffmpeg/lib/ 2>/dev/null || { echo "Error: libogg.so not found"; exit 1; }

# Collect copyright files for packages whose files we explicitly copy into the runtime.
# `dpkg -S` resolves paths against the dpkg database, which only tracks files at
# their ORIGINAL locations. /opt/ffmpeg/lib/libvpx.so*, libvorbis.so*, libogg.so*
# were copied from /usr/lib/, so querying /opt/ffmpeg/lib/ returns nothing for
# them — we must query the /usr/lib/ source paths instead. /bin/bash is still
# at its dpkg-tracked location.
RUN { dpkg -S /bin/bash 2>/dev/null; \
      for f in /usr/lib/*/libvpx.so* /usr/lib/libvpx.so* \
               /usr/lib/*/libvorbis.so* /usr/lib/libvorbis.so* \
               /usr/lib/*/libvorbisenc.so* /usr/lib/libvorbisenc.so* \
               /usr/lib/*/libogg.so* /usr/lib/libogg.so*; do \
        [ -e "$f" ] && dpkg -S "$f" 2>/dev/null; \
      done; \
    } | awk -F: '{print $1}' \
      | sort -u > /opt/licenses/dpkg/runtime-pkgs.txt \
    && while read pkg; do \
        [ -f "/usr/share/doc/${pkg}/copyright" ] && \
          cp "/usr/share/doc/${pkg}/copyright" "/opt/licenses/dpkg/${pkg}.copyright"; \
      done < /opt/licenses/dpkg/runtime-pkgs.txt

ENV PATH="/opt/ffmpeg/bin${PATH:+:${PATH}}"
ENV LD_LIBRARY_PATH="/opt/ffmpeg/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Create directories for the nvs user (UID 1000 in NVIDIA distroless)
RUN mkdir -p /app /app/artifacts /app/.cache \
    && chown -R 1000:1000 /app \
    && chmod -R 755 /app

# Install only the dependencies using uv
COPY pyproject.toml .
RUN uv sync --active --no-install-project

# Copy the rest of the application
COPY --from=wheel-builder /dist /dist
RUN uv pip install /dist/aiperf-*.whl \
    && rm -rf /dist /workspace/pyproject.toml

# Remove setuptools as it is not needed for the runtime image
RUN uv pip uninstall setuptools

# Pre-cache tiktoken o200k_base encoding for --tokenizer builtin (MIT license, see ATTRIBUTIONS.md)
RUN mkdir -p /opt/tiktoken_cache \
    && TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

############################################
######### Python License Collector #########
############################################
FROM env-builder AS python-licenses

COPY tools/generate_python_attributions.py /tmp/generate_python_attributions.py
COPY tools/requirements.licenses.txt /tmp/requirements.licenses.txt
COPY tools/licenses.toml /tmp/licenses.toml

# Layer 1: pip-licenses — snapshot venv diff to exclude the tool itself from output
RUN uv pip list --format=freeze | awk -F== '{print $1}' | sort > /tmp/venv-before.txt \
    && uv pip install -r /tmp/requirements.licenses.txt \
    && uv pip list --format=freeze | awk -F== '{print $1}' | sort > /tmp/venv-after.txt \
    && IGNORE=$(comm -13 /tmp/venv-before.txt /tmp/venv-after.txt | tr '\n' ' ') \
    && mkdir -p /opt/licenses/python \
    && pip-licenses \
        --ignore-packages $IGNORE \
        --format=json \
        --with-license-file \
        --output-file=/opt/licenses/python/licenses.json \
    && pip-licenses \
        --ignore-packages $IGNORE \
        --format=json-license-finder \
        --output-file=/opt/licenses/python/ATTRIBUTIONS-Python.json \
    && python3 /tmp/generate_python_attributions.py \
        /opt/licenses/python/licenses.json \
        /opt/licenses/python/ATTRIBUTIONS-Python.md \
        /opt/licenses/python/python-deps.csv \
        /tmp/licenses.toml \
    && rm /tmp/venv-before.txt /tmp/venv-after.txt

# Layer 2: cyclonedx-bom via uvx — installs in isolated env, scans specified venv only
RUN uvx --from cyclonedx-bom cyclonedx-py environment /opt/aiperf/venv/bin/python \
    --output-format JSON \
    --output-file /opt/licenses/python/sbom.cdx.json

# Layer 3: dpkg attribution CSV for runtime-distributed system packages
COPY tools/generate_dpkg_attributions.py /tmp/generate_dpkg_attributions.py
RUN python3 /tmp/generate_dpkg_attributions.py \
    /opt/licenses/dpkg/runtime-pkgs.txt \
    /opt/licenses/dpkg/dpkg-deps.csv \
    /tmp/licenses.toml

# Export-only stage: scratch-based so `docker buildx build --target
# licenses-artifact --output type=local,dest=<dir>` writes only the license
# tree (a few MB) instead of the ~1.3 GB python-licenses filesystem.
FROM scratch AS licenses-artifact
COPY --from=python-licenses /opt/licenses/ /

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
FROM nvcr.io/nvidia/distroless/python:3.13-v4.0.3-dev AS runtime

# Include project license and asset attributions
COPY LICENSE ATTRIBUTIONS.md /legal/

# Include dynamically collected third-party licenses
COPY --from=env-builder /opt/licenses/ /licenses/
COPY --from=python-licenses /opt/licenses/python/ATTRIBUTIONS-Python.md /licenses/python/ATTRIBUTIONS-Python.md

# Copy bash with executable permissions preserved using --chmod
COPY --from=env-builder --chown=1000:1000 --chmod=755 /bin/bash /bin/bash

# Copy ffmpeg binaries and libraries (includes libvpx)
COPY --from=env-builder --chown=1000:1000 /opt/ffmpeg /opt/ffmpeg
ENV PATH="/opt/ffmpeg/bin${PATH:+:${PATH}}"
ENV LD_LIBRARY_PATH="/opt/ffmpeg/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Setup the directories with permissions for nvs user
COPY --from=env-builder --chown=1000:1000 /app /app
WORKDIR /app
ENV HOME=/app

# Copy the virtual environment and set up
COPY --from=env-builder --chown=1000:1000 /opt/aiperf/venv /opt/aiperf/venv

# Copy pre-cached tiktoken encoding for zero-network --tokenizer builtin
COPY --from=env-builder --chown=1000:1000 /opt/tiktoken_cache /opt/tiktoken_cache

ENV VIRTUAL_ENV=/opt/aiperf/venv \
    PATH="/opt/aiperf/venv/bin:${PATH}" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

# Set bash as entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
