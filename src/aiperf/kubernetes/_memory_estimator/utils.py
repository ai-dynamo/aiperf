# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small byte/unit helpers shared across the estimator modules."""

from __future__ import annotations


def _ceil_pow2(n: int) -> int:
    """Next power of 2 >= n (matches GrowableArray doubling behavior)."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _mib(byte_count: float) -> float:
    """Convert bytes to MiB."""
    return byte_count / (1024 * 1024)
