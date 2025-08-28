#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

. /aiperf/tests/ci/common/setup_aiperf.sh

pytest -vv -s "/aiperf/tests/ci/test_docs/test_docs.py"

