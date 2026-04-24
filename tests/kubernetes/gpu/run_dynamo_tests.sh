#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run Dynamo GPU Kubernetes integration tests with full debug output.
#
# Usage:
#   ./tests/kubernetes/gpu/run_dynamo_tests.sh
#   GPU_TEST_DYNAMO_ENDPOINT=http://dynamo:8000/v1 ./tests/kubernetes/gpu/run_dynamo_tests.sh
#   GPU_TEST_CONTEXT=my-cluster ./tests/kubernetes/gpu/run_dynamo_tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

exec uv run pytest tests/kubernetes/gpu/dynamo/ \
    -v \
    -s \
    -m dynamo \
    -o "addopts=--strict-markers" \
    --log-cli-level=DEBUG \
    --log-cli-format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s" \
    --log-cli-date-format="%H:%M:%S" \
    --tb=long \
    "$@"
