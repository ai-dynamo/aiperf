#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Forward optional shard parameters from the CI matrix. When SERVER_NAME is
# set, the script runs only that one server (matrix-shard mode). When
# SHARD_TOTAL is also set (>1), the named server's command list is sliced.
# With nothing set, behavior is unchanged: --all-servers, one big sequential run.
ARGS=()
if [ -n "${SERVER_NAME:-}" ]; then
  ARGS+=("--server" "${SERVER_NAME}")
  if [ -n "${SHARD_TOTAL:-}" ]; then
    ARGS+=("--shard-index" "${SHARD_INDEX:-0}" "--shard-total" "${SHARD_TOTAL}")
  fi
else
  ARGS+=("--all-servers")
fi

echo "Running python3 main.py ${ARGS[*]}"
cd ${AIPERF_SOURCE_DIR}/tests/ci/${CI_JOB_NAME}/
python3 main.py "${ARGS[@]}"
