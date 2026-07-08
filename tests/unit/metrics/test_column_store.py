# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ColumnStore numeric running-sum/count last-write-wins semantics.

Regression coverage for the RECORD-metric double-count bug: a record can be
re-delivered to the same slot (``idx``); the float64 column overwrites the cell
(last-write-wins), so the O(1) running sum side-channel must too. Without the
back-out, a re-delivered record inflates ``numeric_sum`` while ``numeric()``
stays deduped, and the read path (``metric_result_from_array``) computes
``avg = inflated_sum / dedup_len`` which can exceed ``max``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import param

from aiperf.metrics.column_store import ColumnStore, ListMetricBackendT
from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator
from aiperf.metrics.metric_dicts import metric_result_from_array
from aiperf.metrics.ragged_series import RaggedSeries


def _ingest(store: ColumnStore, idx: int, tag: str, value: float) -> None:
    """Write a single numeric metric value to ``idx`` with dummy timestamps."""
    store.ingest(
        idx,
        record_metrics={tag: value},
        start_ns=1.0,
        end_ns=2.0,
        generation_start_ns=None,
    )


def _ingest_list(store: ColumnStore, idx: int, tag: str, values: list[float]) -> None:
    """Write a single list-valued metric to ``idx`` with dummy timestamps."""
    store.ingest(
        idx,
        record_metrics={tag: values},
        start_ns=1.0,
        end_ns=2.0,
        generation_start_ns=None,
    )


def _list_sample_count(store: ColumnStore, tag: str) -> int:
    """Number of list samples stored for ``tag`` — backend-agnostic."""
    backend = store.ragged(tag)
    if getattr(backend, "SUPPORTS_PER_RECORD_REPLAY", False):
        return len(backend.values)
    return len(backend)


def _list_sample_sum(store: ColumnStore, tag: str) -> float:
    """Sum of list samples stored for ``tag`` — backend-agnostic."""
    backend = store.ragged(tag)
    if getattr(backend, "SUPPORTS_PER_RECORD_REPLAY", False):
        return float(backend.values.sum())
    return float(backend.sum)


class TestColumnStoreNumericReDelivery:
    """Last-write-wins for the numeric running sum/count on slot re-delivery."""

    def test_rewrite_same_slot_last_write_wins(self) -> None:
        """Re-delivering a different value to the same slot replaces, not adds."""
        store = ColumnStore(initial_capacity=8)
        _ingest(store, 0, "m", 64.0)
        _ingest(store, 0, "m", 128.0)  # re-delivery to the SAME slot

        # Sum reflects only the latest value per slot (128, not 64 + 128 = 192).
        assert store.numeric_sum("m") == 128.0
        # Count is the number of distinct populated slots, not write count.
        assert store.numeric_count("m") == 1
        # The deduped column holds only the latest value.
        col = store.numeric("m")
        assert list(col[~np.isnan(col)]) == [128.0]

    def test_rewrite_same_value_does_not_double_count(self) -> None:
        """Re-delivering the SAME value twice keeps sum == value, count == 1.

        Old behavior: sum == 128 over a 1-row deduped column -> avg 128 > max 64.
        """
        store = ColumnStore(initial_capacity=8)
        _ingest(store, 0, "m", 64.0)
        _ingest(store, 0, "m", 64.0)

        assert store.numeric_sum("m") == 64.0
        assert store.numeric_count("m") == 1

    def test_distinct_slots_accumulate_normally(self) -> None:
        """Writes to distinct slots accumulate sum and count as before."""
        store = ColumnStore(initial_capacity=8)
        _ingest(store, 0, "m", 10.0)
        _ingest(store, 1, "m", 20.0)
        _ingest(store, 2, "m", 30.0)

        assert store.numeric_sum("m") == 60.0
        assert store.numeric_count("m") == 3

    def test_multiple_redeliveries_same_slot(self) -> None:
        """Many re-deliveries to one slot leave sum == last value, count == 1."""
        store = ColumnStore(initial_capacity=8)
        for value in (10.0, 20.0, 30.0, 40.0):
            _ingest(store, 0, "m", value)

        assert store.numeric_sum("m") == 40.0
        assert store.numeric_count("m") == 1
        col = store.numeric("m")
        assert list(col[~np.isnan(col)]) == [40.0]

    def test_mixed_distinct_and_redelivery(self) -> None:
        """A rewrite of one slot alongside untouched slots keeps totals exact."""
        store = ColumnStore(initial_capacity=8)
        _ingest(store, 0, "m", 100.0)
        _ingest(store, 1, "m", 25.0)
        _ingest(store, 0, "m", 50.0)  # re-delivery of slot 0

        assert store.numeric_sum("m") == 75.0  # 50 (latest slot0) + 25 (slot1)
        assert store.numeric_count("m") == 2

    def test_redelivery_survives_grow(self) -> None:
        """A grow between writes (which clears cached handlers) still dedups.

        ``_grow`` reallocates numeric columns and drops ``_tag_handlers``; the
        rebuilt handler must still read the (copied) prior cell to back it out.
        """
        store = ColumnStore(initial_capacity=8)
        _ingest(store, 0, "m", 64.0)
        _ingest(store, 2000, "m", 10.0)  # idx beyond capacity -> triggers grow
        _ingest(store, 0, "m", 128.0)  # rewrite slot 0 after the reallocation

        assert store.numeric_sum("m") == 138.0  # 128 (latest slot0) + 10 (slot2000)
        assert store.numeric_count("m") == 2

    def test_readpath_avg_within_min_max_on_redelivery(self) -> None:
        """End-to-end read path: avg stays within [min, max] after a re-delivery.

        Mirrors ``accumulator._collect_scalars_and_arrays`` + ``_build_metric_results``:
        the deduped ``numeric()`` array and the O(1) ``numeric_sum()`` feed
        ``metric_result_from_array``. With the double-count bug this produced
        ``avg = 192 / 1 = 192`` against ``max = 128``.
        """
        store = ColumnStore(initial_capacity=8)
        _ingest(store, 0, "m", 64.0)
        _ingest(store, 0, "m", 128.0)  # re-delivery inflates sum under the bug

        col = store.numeric("m")
        clean = col[~np.isnan(col)]
        result = metric_result_from_array(
            "m", "Test Metric", "ms", clean.copy(), store.numeric_sum("m")
        )

        assert result.min <= result.avg <= result.max
        assert result.avg == 128.0
        assert result.max == 128.0
        assert result.count == 1


@pytest.mark.parametrize(
    "backend_cls",
    [
        param(RaggedSeries, id="ragged"),
        param(TDigestListMetricAggregator, id="tdigest"),
    ],
)  # fmt: skip
class TestColumnStoreListReDelivery:
    """First-wins dedup for list-valued RECORD metrics on slot re-delivery.

    Sibling of the numeric last-write-wins fix: ``make_list_handler`` routes to
    ``backend.add_for_record(idx, values)``. Before the fix a re-delivered
    record's list was appended a second time, so the pooled sample count (and
    thus percentiles) counted one request's chunks twice, disagreeing with the
    deduped numeric store and ``record_count``. First-wins (not last-wins)
    because re-delivery replays an identical payload and the t-digest backend
    has no value-removal op, so both list backends share the semantic.
    """

    def test_redelivered_list_counted_once(
        self, backend_cls: type[ListMetricBackendT]
    ) -> None:
        """The same 3-chunk record delivered twice contributes 3 samples, not 6."""
        store = ColumnStore(initial_capacity=8, list_backend_cls=backend_cls)
        _ingest_list(store, 0, "inter_chunk_latency", [10.0, 20.0, 30.0])
        _ingest_list(store, 0, "inter_chunk_latency", [10.0, 20.0, 30.0])  # re-delivery

        assert _list_sample_count(store, "inter_chunk_latency") == 3
        assert _list_sample_sum(store, "inter_chunk_latency") == 60.0

    def test_distinct_slots_accumulate(
        self, backend_cls: type[ListMetricBackendT]
    ) -> None:
        """Distinct records each contribute their own list samples."""
        store = ColumnStore(initial_capacity=8, list_backend_cls=backend_cls)
        _ingest_list(store, 0, "inter_chunk_latency", [10.0, 20.0, 30.0])
        _ingest_list(store, 1, "inter_chunk_latency", [5.0, 5.0])

        assert _list_sample_count(store, "inter_chunk_latency") == 5
        assert _list_sample_sum(store, "inter_chunk_latency") == 70.0

    def test_redelivery_survives_backend_grow(
        self, backend_cls: type[ListMetricBackendT]
    ) -> None:
        """A slot index beyond the backend's initial dedup capacity still dedups.

        Exercises ``RaggedSeries._grow_offsets`` and
        ``TDigestListMetricAggregator._grow_seen`` (both start at 256).
        """
        store = ColumnStore(initial_capacity=8, list_backend_cls=backend_cls)
        _ingest_list(store, 500, "inter_chunk_latency", [1.0, 2.0])
        _ingest_list(store, 500, "inter_chunk_latency", [1.0, 2.0])  # re-delivery

        assert _list_sample_count(store, "inter_chunk_latency") == 2
        assert _list_sample_sum(store, "inter_chunk_latency") == 3.0
