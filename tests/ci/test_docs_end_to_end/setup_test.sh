#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "Running python3 main.py --all-servers"
cd ${AIPERF_SOURCE_DIR}/tests/ci/${CI_JOB_NAME}
python3 main.py --all-servers