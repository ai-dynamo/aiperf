#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

apt update && apt install -y curl jq

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv --python 3.10

source .venv/bin/activate

uv pip install "/aiperf[dev]"

echo "AIPerf Installation complete"

pytest -vv -s "/aiperf/tests/ci/test_docs/test_docs.py"

