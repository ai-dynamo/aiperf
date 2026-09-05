# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import platform as _platform

# Platform detection — evaluated once at import time.
IS_WINDOWS: bool = _platform.system() == "Windows"
IS_MACOS: bool = _platform.system() == "Darwin"
IS_LINUX: bool = _platform.system() == "Linux"
# Windows-on-ARM: several native stacks (pyarrow/datasets wheels, libsndfile,
# kaleido's browser engine) have no working ARM64 build, so features relying on
# them must gate on this rather than crash.
IS_WINDOWS_ARM: bool = IS_WINDOWS and _platform.machine() == "ARM64"

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

GOOD_REQUEST_COUNT_TAG = "good_request_count"
"""GoodRequestCount metric tag"""

SYSTEM_PROMPT_JOIN_SEP = "\n\n"
"""Separator joining a verbatim ``--system-prompt`` to a dataset's own system message.

Shared by the two places that perform the merge -- the composer, for datasets
whose system message was hoisted to ``Conversation.system_message``, and the chat
endpoint, for datasets that leave an unhoisted ``role: system`` in
``turn.raw_messages``. The two must agree: a run can hit either path depending on
the loader, and a drift between them would silently change request bytes and
prefix-cache behavior.
"""
