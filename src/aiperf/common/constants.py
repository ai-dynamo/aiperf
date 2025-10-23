# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MILLIS = 1_000_000
MILLIS_PER_SECOND = 1000
BYTES_PER_MIB = 1024 * 1024

STAT_KEYS = [
    "avg",
    "min",
    "max",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "std",
]

GOOD_REQUEST_COUNT_TAG = "good_request_count"
"""GoodRequestCount metric tag"""
