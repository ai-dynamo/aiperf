# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-robin CI sharding of the integration suite (AIPERF_SHARD_INDEX / AIPERF_SHARD_TOTAL)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

import pytest

T = TypeVar("T")

SHARD_INDEX_ENV = "AIPERF_SHARD_INDEX"
SHARD_TOTAL_ENV = "AIPERF_SHARD_TOTAL"


def ci_shard_from_env(environ: Mapping[str, str]) -> tuple[int, int]:
    """Return ``(index, total)``; ``(0, 1)`` (no sharding) when unset.

    Raises ``pytest.UsageError`` for a non-integer value, a total below 1, or an
    index outside ``[0, total)``.
    """
    try:
        total = int(environ.get(SHARD_TOTAL_ENV, "1") or 1)
        index = int(environ.get(SHARD_INDEX_ENV, "0") or 0)
    except ValueError as e:
        raise pytest.UsageError(
            f"{SHARD_INDEX_ENV} and {SHARD_TOTAL_ENV} must be integers: {e}"
        ) from e
    if total < 1:
        raise pytest.UsageError(f"{SHARD_TOTAL_ENV}={total} must be >= 1")
    if total == 1:
        return 0, 1
    if not 0 <= index < total:
        raise pytest.UsageError(f"{SHARD_INDEX_ENV}={index} must be in [0, {total})")
    return index, total


def select_ci_shard(
    items: Sequence[T], index: int, total: int
) -> tuple[list[T], list[T]]:
    """Split ``items`` into ``(keep, drop)`` for shard ``index`` of ``total``.

    Round-robin over the given order, so every xdist worker (same collection
    order) computes the same subset.
    """
    keep: list[T] = []
    drop: list[T] = []
    for i, item in enumerate(items):
        (keep if i % total == index else drop).append(item)
    return keep, drop
