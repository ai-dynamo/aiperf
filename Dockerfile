# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.12-slim AS base

ENV USERNAME=appuser
ENV APP_NAME=aiperf

# Create app user
RUN groupadd -r $USERNAME \
    && useradd -r -g $USERNAME $USERNAME

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
############# Final Build ##################
############################################
FROM base AS final

# Create virtual environment
RUN mkdir /opt/$APP_NAME \
    && uv venv /opt/$APP_NAME/venv --python 3.12 \
    && chown -R $USERNAME:$USERNAME /opt/$APP_NAME

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/$APP_NAME/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

### INSTALLATION ###

# Copy pyproject first for better layer caching
COPY pyproject.toml .

# Install only the dependencies using uv
RUN uv sync --active --no-install-project

# Copy the rest of the application
COPY . .

# Install the project
RUN uv sync --active --no-dev

# Command to run the application
ENTRYPOINT ["aiperf"]
