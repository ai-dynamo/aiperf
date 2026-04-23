# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrency limiting for credit issuance.

Provides three layers of concurrency control:
- DynamicConcurrencyLimit: Semaphore with dynamic limit adjustment and debt tracking
- GlobalPhaseConcurrencyLimiter: Phase-specific limits with global coordination
- ConcurrencyManager: High-level manager for session and prefill concurrency

Used by CreditIssuer and CreditCallbackHandler to control how many concurrent
sessions and prefill requests are active during benchmarking.
"""

from aiperf.timing.concurrency.dynamic_limit import (
    ConcurrencyStats,
    DynamicConcurrencyLimit,
)
from aiperf.timing.concurrency.global_phase_limiter import (
    GlobalPhaseConcurrencyLimiter,
)
from aiperf.timing.concurrency.manager import ConcurrencyManager

__all__ = [
    "ConcurrencyManager",
    "ConcurrencyStats",
    "DynamicConcurrencyLimit",
    "GlobalPhaseConcurrencyLimiter",
]
