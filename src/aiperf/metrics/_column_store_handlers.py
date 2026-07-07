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
    O(1) running sum/count side-channel with last-write-wins semantics.

    A record can be re-delivered to the same slot (``idx``). The float64 column
    overwrites the cell in place (last-write-wins, mirroring
    ``GrowableArray.__setitem__``), so the running sum must mirror that too: on a
    re-write the prior slot value is backed out of ``sums[tag]`` before the new
    value is added, and ``counts[tag]`` is NOT incremented — it tracks the number
    of distinct populated slots, which is the deduped column length the read path
    (``accumulator._collect_scalars_and_arrays``) divides the sum by. Without the
    back-out, a re-delivered record double-counts the sum and
    ``avg = sum / dedup_len`` can exceed ``max``.

    A slot is "populated" iff its current cell is non-NaN (NaN is the pre-fill /
    missing sentinel). The ``prior == prior`` test is a NaN check — NaN is the
    only value unequal to itself — avoiding a ``np.isnan`` call on the hot path.
    On a re-write the prior cell is cast to ``float`` so ``sums[tag]`` stays a
    Python float; the first-write branch keeps the cast-free fast path (numpy
    coerces Python ``int`` to ``float64`` and the sum add promotes it the same
    way).
    """

    def handler(idx: int, value: Any) -> None:
        prior = col[idx]
        if prior == prior:  # non-NaN: slot already populated, last-write-wins
            sums[tag] = sums[tag] - float(prior) + value
        else:
            sums[tag] = sums[tag] + value
            counts[tag] = counts[tag] + 1
        col[idx] = value

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
