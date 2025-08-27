#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

bash -x "/aiperf/tests/ci/common/setup_aiperf.sh"

bash -x /aiperf/tests/ci/${CI_JOB_NAME}/test.sh

