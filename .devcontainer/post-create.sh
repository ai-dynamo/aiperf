#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


trap 'echo "❌ ERROR: Command failed at line $LINENO: $BASH_COMMAND"; echo "⚠️ This was unexpected and setup was not completed. Can try to resolve yourself and then manually run the rest of the commands in this file or file a bug."' ERR

retry() {
    # retries for connectivity issues in installs
    local retries=3
    local count=0
    until "$@"; do
        exit_code=$?
        wait_time=$((2 ** count))
        echo "Command failed with exit code $exit_code. Retrying in $wait_time seconds..."
        sleep $wait_time
        count=$((count + 1))
        if [ $count -ge $retries ]; then
            echo "Command failed after $retries attempts."
            return $exit_code
        fi
    done
    return 0
}

set -xe

# Changing permission to match local user since volume mounts default to root ownership
sudo chown -R appuser:appuser "${HOME}"

# Create the virtual environment if it doesn't exist
if ! [ -f "${HOME}/.venv/bin/python" ]; then
    echo "Creating virtual environment..."
    uv venv "${HOME}/.venv" --python 3.12
    echo "Virtual environment created at ${HOME}/.venv"
else
    echo "Virtual environment already exists at ${HOME}/.venv"
fi

# Set the proper venv path for using make commands
echo "VENV_PATH=${HOME}/.venv" >> ".env.mk"

# Source the venv
source "${HOME}/.venv/bin/activate"
# Change the uv cache dir to reside in the same volume as the venv
# This is to allow uv to hardlink the cache files to the venv for better performance
mkdir -p "${HOME}/.venv/.cache/uv"
export UV_CACHE_DIR="${HOME}/.venv/.cache/uv"

# install the python package plus dev dependencies in editable mode
cd "${HOME}/aiperf" && retry uv pip install -e ".[dev]"

# Pre-commit hooks
cd "${HOME}/aiperf" && pre-commit install && retry pre-commit install-hooks
pre-commit run --all-files || true # don't fail the build if pre-commit hooks fail

# source the venv and setup alias inside bashrc
{
  echo "source \"${HOME}/.venv/bin/activate\"";
  echo "export GPG_TTY=$(tty)";
  echo "alias pip=\"uv pip\"";
  echo "export UV_CACHE_DIR=\"${HOME}/.venv/.cache/uv\"";
} >> ~/.bashrc
