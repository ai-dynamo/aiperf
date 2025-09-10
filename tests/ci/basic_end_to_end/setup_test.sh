#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "Starting dynamo server with model ${MODEL}"

bash -x ${AIPERF_SOURCE_DIR}/tests/ci/${CI_JOB_NAME}/start_server.sh

echo "Running aiperf functionality tests in container"

docker run \
    --rm \
    --gpus all \
    --network host \
    -w /aiperf \
    -v ${AIPERF_SOURCE_DIR}:/aiperf \
    -e CI_JOB_NAME=${CI_JOB_NAME} \
    -e MODEL=${MODEL} \
    ubuntu:24.04 /bin/bash -x /aiperf/tests/ci/common/setup_aiperf.sh
