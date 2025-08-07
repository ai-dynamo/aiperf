#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

apt update && apt install -y curl

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv

source .venv/bin/activate

uv pip install /aiperf

bash -x /aiperf/tests/ci/${CI_JOB_NAME}/test.sh
