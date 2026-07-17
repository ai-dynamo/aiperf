# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import platform as _platform

from aiperf.common.enums.enums import CreditPhase

# Platform detection — evaluated once at import time.
IS_WINDOWS: bool = _platform.system() == "Windows"
IS_MACOS: bool = _platform.system() == "Darwin"
IS_LINUX: bool = _platform.system() == "Linux"
# Windows-on-ARM: several native stacks (pyarrow/datasets wheels, libsndfile,
# kaleido's browser engine) have no working ARM64 build, so features relying on
# them must gate on this rather than crash.
IS_WINDOWS_ARM: bool = IS_WINDOWS and _platform.machine() == "ARM64"

NANOS_PER_SECOND = 1_000_000_000
MICROS_PER_SECOND = 1_000_000
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

STREAMED_REQUEST_TAG = "streamed_request"
"""Hidden per-record streaming predicate tag: its presence in a record's
``MetricRecordDict`` gates every per-record streaming metric. Kept out of
metric modules so non-metric consumers (e.g. gate checks) need not import them."""

STREAMED_REQUEST_COUNT_TAG = "streamed_request_count"
"""Visible aggregate streamed-request-count tag (the streaming denominator
displayed beside Request Count)."""
