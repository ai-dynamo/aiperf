#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

aiperf profile --model-names ${MODEL} -u localhost:8080 --concurrency 100 --request-count 300
