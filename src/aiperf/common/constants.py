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
    "sum",
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

# Optional percentile stats when failed requests are modeled as unbounded latency.
ADJ_PERCENTILE_STAT_KEYS = ["adj_p50", "adj_p90", "adj_p95", "adj_p99"]

GOOD_REQUEST_COUNT_TAG = "good_request_count"
"""GoodRequestCount metric tag"""
