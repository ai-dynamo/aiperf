# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import platform as _platform

from aiperf.common.enums.enums import CreditPhase

# Platform detection — evaluated once at import time.
IS_WINDOWS: bool = _platform.system() == "Windows"
IS_MACOS: bool = _platform.system() == "Darwin"
IS_LINUX: bool = _platform.system() == "Linux"

NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MILLIS = 1_000_000
MILLIS_PER_SECOND = 1000
BYTES_PER_MIB = 1024 * 1024
WARMUP_SYSTEM_MESSAGE_PREFIX = CreditPhase.WARMUP

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

GOOD_REQUEST_COUNT_TAG = "good_request_count"
"""GoodRequestCount metric tag"""
