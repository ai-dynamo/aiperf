# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-tag setter closure factories for ``ColumnStore.ingest``.

These closures are resolved on first sighting of each metric tag (via Python
type dispatch) and cached in ``ColumnStore._tag_handlers``. Subsequent records
skip the isinstance ladder and the ``_ensure_*_column`` lookups entirely.

Profiling at 50k records (24 numeric tags + ICL) showed this hoist drops
``ColumnStore.ingest`` wall by ~30% and total ingest function calls by 40%.
The handlers are invalidated by ``_grow()`` because numeric arrays get
reallocated; closures captured the old array references and would write to
garbage. List backends and string lists are unaffected (in-place growth) but
clearing all handlers on grow is simpler and grow runs ~log2(N) times.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator
from aiperf.metrics.ragged_series import RaggedSeries


def make_numeric_handler(
    col: NDArray[np.float64],
    tag: str,
    sums: dict[str, float],
    counts: dict[str, int],
) -> Callable[[int, Any], None]:
    """Closure that writes a numeric metric value at ``idx`` and updates the
    O(1) running sum/count side-channel.

    The ``float()`` cast is intentionally absent: numpy's ``__setitem__``
    coerces Python ``int`` to ``float64`` automatically, and ``+=`` on the
    sum dict promotes the int operand the same way. Saves a Python-level
    function call per numeric metric per record (~5-8% on the scalar path).
    """

    def handler(idx: int, value: Any) -> None:
        col[idx] = value
        sums[tag] = sums[tag] + value
        counts[tag] = counts[tag] + 1

    return handler


def make_string_handler(
    col: list[str | None],
) -> Callable[[int, Any], None]:
    """Closure that writes a string metric value at ``idx``. The list reference
    survives capacity growth (``list.extend`` is in-place)."""

    def handler(idx: int, value: Any) -> None:
        col[idx] = value

    return handler


def make_list_handler(
    backend: RaggedSeries | TDigestListMetricAggregator,
) -> Callable[[int, Any], None]:
    """Closure that hands a list-valued metric to the configured list backend.
    The backend reference is stable across ``ColumnStore._grow`` (list backends
    own their own growth)."""

    def handler(idx: int, value: Any) -> None:
        backend.add_for_record(idx, value)

    return handler
