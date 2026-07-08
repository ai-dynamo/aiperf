# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage for list-valued per-record metrics (e.g., ICL/ITL)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aiperf.common.growable_array import GrowableArray


class RaggedSeries:
    """Storage for list-valued per-record metrics (e.g., ICL).

    Uses offsets array for O(1) per-record lookup, efficient bulk retrieval
    via boolean mask on record indices, and vectorized grouped operations
    (e.g., per-request cumulative sums for ICL-aware throughput sweeps).

    ``SUPPORTS_PER_RECORD_REPLAY = True`` advertises that this backend retains
    enough per-request structure to drive ICL-aware sweep curves in
    ``accumulator_sweeps``. The alternative t-digest backend sets it ``False``
    and the sweep helpers degrade to their request-level fallbacks.
    """

    SUPPORTS_PER_RECORD_REPLAY = True

    __slots__ = ("_values", "_record_indices", "_offsets", "_offsets_capacity")

    def __init__(
        self, initial_capacity: int = 1024, offsets_capacity: int = 256
    ) -> None:
        self._values = GrowableArray(
            initial_capacity=initial_capacity, dtype=np.float64
        )
        self._record_indices = GrowableArray(
            initial_capacity=initial_capacity, dtype=np.int32
        )
        # Per-session_num start offset into _values. -1 means absent.
        self._offsets = np.full(offsets_capacity, -1, dtype=np.int64)
        self._offsets_capacity = offsets_capacity

    @property
    def values(self) -> NDArray[np.float64]:
        """All concatenated values."""
        return self._values.data

    @property
    def record_indices(self) -> NDArray[np.int32]:
        """Session_num per value."""
        return self._record_indices.data

    @property
    def offsets(self) -> NDArray[np.int64]:
        """Per-session_num start offset. -1 if absent."""
        return self._offsets[: self._offsets_capacity]

    def extend(self, idx: int, values: list[float]) -> None:
        """Append values for session_num ``idx``, first-wins on re-delivery.

        A record can be re-delivered to an already-populated ``idx`` (at-least-
        once messaging replays the identical payload). First-wins dedup: the
        re-delivery is skipped entirely, so the slot contributes its values
        EXACTLY ONCE. Without the skip a duplicate block was appended under the
        same ``record_indices`` code, so ``get_values_for_mask`` returned the
        values twice (biasing pooled percentiles) and ``grouped_cumsum``
        resolved the duplicate block against the FIRST block's ``offsets`` start,
        corrupting the per-request cumsum reset for ICL sweeps.

        First-wins (not last-wins) mirrors the observable dedup of the numeric
        handler in ``_column_store_handlers.make_numeric_handler``: re-delivery
        carries an identical payload, so keeping the first equals keeping the
        last. It is also the only semantic implementable on the sibling
        :class:`~aiperf.metrics.list_metric_aggregation.TDigestListMetricAggregator`
        backend, whose sketch has no value-removal op — so both list backends
        agree, and neither disagrees with ``MetricsAccumulator.record_count``.
        """
        n = len(values)
        if n == 0:
            return
        if idx >= self._offsets_capacity:
            self._grow_offsets(idx)
        if self._offsets[idx] >= 0:  # slot already populated: skip re-delivery
            return
        self._offsets[idx] = len(self._values)
        val_arr = np.asarray(values, dtype=np.float64)
        idx_arr = np.full(n, idx, dtype=np.int32)
        self._values.extend(val_arr)
        self._record_indices.extend(idx_arr)

    def add_for_record(self, idx: int, values: list[float]) -> None:
        """Unified backend contract — same signature on the t-digest backend."""
        self.extend(idx, values)

    def get_values_for_mask(
        self, record_mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Return values whose record is selected by the boolean mask."""
        if len(self._record_indices) == 0:
            return np.zeros(0, dtype=np.float64)
        value_mask = record_mask[self._record_indices.data]
        return self._values.data[value_mask]

    def grouped_cumsum(self) -> NDArray[np.float64]:
        """Compute per-request cumulative sum across the flat values array.

        Uses offsets to reset at request boundaries — fully vectorized, no Python loops.
        This is the foundation of ICL-aware throughput sweeps.
        """
        if len(self._values) == 0:
            return np.zeros(0, dtype=np.float64)

        global_cs = np.cumsum(self._values.data)
        rec_idx = self._record_indices.data
        request_offsets = self._offsets[rec_idx]
        start_cs = np.where(request_offsets > 0, global_cs[request_offsets - 1], 0.0)
        return global_cs - start_cs

    def _grow_offsets(self, min_idx: int) -> None:
        """Grow offsets array to accommodate min_idx."""
        new_cap = self._offsets_capacity
        while new_cap <= min_idx:
            new_cap *= 2
        new_offsets = np.full(new_cap, -1, dtype=np.int64)
        new_offsets[: self._offsets_capacity] = self._offsets[: self._offsets_capacity]
        self._offsets = new_offsets
        self._offsets_capacity = new_cap
