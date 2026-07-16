# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ColumnStore — session-indexed columnar metric storage."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.metrics.column_store import (
    _BOOL_MISSING,
    _CATEGORICAL_MISSING,
    ColumnStore,
)
from aiperf.metrics.ragged_series import RaggedSeries


def _make_store(initial_capacity: int = 8) -> ColumnStore:
    # Pin the ragged backend to keep tests independent of the env flag.
    return ColumnStore(initial_capacity=initial_capacity, list_backend_cls=RaggedSeries)


def test_init_empty_count_and_columns():
    store = _make_store()
    assert store.count == 0
    assert store.numeric_tags() == []
    assert store.ragged_tags() == []


def test_init_timestamp_columns_filled_with_nan():
    store = _make_store(initial_capacity=4)
    assert np.all(np.isnan(store.start_ns))
    assert np.all(np.isnan(store.end_ns))
    assert np.all(np.isnan(store.generation_start_ns))


def test_ingest_writes_numeric_value_and_timestamps():
    store = _make_store()
    store.ingest(
        0,
        record_metrics={"latency_ns": 100.0},
        start_ns=10.0,
        end_ns=20.0,
        generation_start_ns=15.0,
    )
    assert store.count == 1
    assert store.numeric("latency_ns")[0] == 100.0
    assert store.start_ns[0] == 10.0
    assert store.end_ns[0] == 20.0
    assert store.generation_start_ns[0] == 15.0


def test_ingest_running_sum_invariant_across_records():
    store = _make_store()
    for i, val in enumerate([1.0, 2.0, 3.0, 4.5]):
        store.ingest(
            i,
            record_metrics={"x": val},
            start_ns=float(i),
            end_ns=float(i) + 1.0,
            generation_start_ns=None,
        )
    assert store.numeric_count("x") == 4
    assert store.numeric_sum("x") == pytest.approx(1.0 + 2.0 + 3.0 + 4.5)
    np.testing.assert_array_equal(store.numeric("x"), [1.0, 2.0, 3.0, 4.5])


def test_ingest_running_sum_accepts_int_without_float_cast():
    """Verifies the post-83cb85017 form: no Python-level float() cast.

    numpy's __setitem__ + dict += both auto-coerce int -> float64.
    """
    store = _make_store()
    store.ingest(
        0, record_metrics={"i": 5}, start_ns=0.0, end_ns=1.0, generation_start_ns=None
    )
    store.ingest(
        1, record_metrics={"i": 7}, start_ns=1.0, end_ns=2.0, generation_start_ns=None
    )
    assert store.numeric_sum("i") == 12.0
    assert store.numeric_count("i") == 2
    assert store.numeric("i").dtype == np.float64


def test_numeric_returns_nan_for_unknown_tag():
    store = _make_store()
    store.ingest(
        0, record_metrics={"x": 1.0}, start_ns=0.0, end_ns=1.0, generation_start_ns=None
    )
    out = store.numeric("does_not_exist")
    assert out.shape == (1,)
    assert np.all(np.isnan(out))


def test_numeric_sum_unknown_tag_returns_zero():
    store = _make_store()
    assert store.numeric_sum("missing") == 0.0
    assert store.numeric_count("missing") == 0


def test_ingest_string_value():
    store = _make_store()
    store.ingest(
        0,
        record_metrics={"name": "alice"},
        start_ns=0,
        end_ns=1,
        generation_start_ns=None,
    )
    store.ingest(
        2,
        record_metrics={"name": "bob"},
        start_ns=2,
        end_ns=3,
        generation_start_ns=None,
    )
    col = store.string("name")
    assert col[0] == "alice"
    assert col[1] is None  # uningested slot
    assert col[2] == "bob"


def test_ingest_list_value_routes_to_ragged_backend():
    store = _make_store()
    store.ingest(
        0,
        record_metrics={"icl": [1.0, 2.0, 3.0]},
        start_ns=0,
        end_ns=10,
        generation_start_ns=None,
    )
    store.ingest(
        1,
        record_metrics={"icl": [4.0]},
        start_ns=10,
        end_ns=20,
        generation_start_ns=None,
    )
    backend = store.ragged("icl")
    assert isinstance(backend, RaggedSeries)
    np.testing.assert_array_equal(backend.values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(backend.record_indices, [0, 0, 0, 1])


def test_ingest_out_of_order_count_tracks_max_idx_plus_one():
    store = _make_store()
    store.ingest(
        5, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest(
        2, record_metrics={"x": 2.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    assert store.count == 6
    # Slot 0..1 unfilled
    col = store.numeric("x")
    assert np.isnan(col[0])
    assert np.isnan(col[1])
    assert col[2] == 2.0
    assert col[5] == 1.0


def test_ingest_grows_capacity_when_idx_exceeds_initial():
    store = _make_store(initial_capacity=4)
    # Force grow: idx=10 needs cap >= 16
    store.ingest(
        10, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    assert store.count == 11
    assert store.numeric("x")[10] == 1.0
    assert store.numeric_sum("x") == 1.0


def test_grow_preserves_existing_numeric_values():
    store = _make_store(initial_capacity=4)
    for i in range(3):
        store.ingest(
            i,
            record_metrics={"x": float(i + 1)},
            start_ns=float(i),
            end_ns=float(i) + 1,
            generation_start_ns=None,
        )
    store.ingest(
        20, record_metrics={"x": 99.0}, start_ns=20, end_ns=21, generation_start_ns=None
    )
    col = store.numeric("x")
    assert col[0] == 1.0
    assert col[1] == 2.0
    assert col[2] == 3.0
    assert col[20] == 99.0
    # Running sum survives reallocation
    assert store.numeric_sum("x") == 1.0 + 2.0 + 3.0 + 99.0
    assert store.numeric_count("x") == 4


def test_grow_invalidates_tag_handlers():
    """After _grow, cached numeric closures point at the OLD array; they must
    be cleared so the next ingest rebinds against the new buffer."""
    store = _make_store(initial_capacity=4)
    store.ingest(
        0, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    # Resolve handler for 'x' so it lands in the cache.
    assert "x" in store._tag_handlers
    store.ingest(
        20, record_metrics={"x": 2.0}, start_ns=20, end_ns=21, generation_start_ns=None
    )
    # Post-grow, ingest should still work — handler rebuilt against new array.
    assert store.numeric("x")[20] == 2.0
    assert store.numeric_sum("x") == 3.0


def test_ingest_metadata_numeric_and_string():
    store = _make_store()
    # Metadata accessors slice on _count, which only bumps on ingest(); ingest a
    # placeholder record first so the metadata reads return a populated row.
    store.ingest(0, record_metrics={}, start_ns=0, end_ns=1, generation_start_ns=None)
    store.ingest_metadata(
        0,
        metadata_numeric={"latency_offset_ns": 5.0},
        metadata_string={"worker_id": "w-0"},
    )
    assert store.metadata_numeric("latency_offset_ns")[0] == 5.0
    assert store.metadata_string("worker_id")[0] == "w-0"


def test_metadata_numeric_does_not_appear_in_metric_columns():
    """Metadata columns must NOT be visible to numeric_tags() (which feeds metric compute)."""
    store = _make_store()
    store.ingest_metadata(
        0,
        metadata_numeric={"meta_only": 1.0},
        metadata_string={},
    )
    assert "meta_only" not in store.numeric_tags()


def test_metadata_bool_encoding():
    store = _make_store()
    # _count bumps on ingest() only — write placeholder records so the metadata
    # accessors expose populated rows.
    store.ingest(0, record_metrics={}, start_ns=0, end_ns=1, generation_start_ns=None)
    store.ingest(1, record_metrics={}, start_ns=1, end_ns=2, generation_start_ns=None)
    store.ingest_metadata(
        0,
        metadata_numeric={},
        metadata_string={},
        metadata_bool={"is_streaming": True},
    )
    store.ingest_metadata(
        1,
        metadata_numeric={},
        metadata_string={},
        metadata_bool={"is_streaming": False},
    )
    col = store.metadata_bool("is_streaming")
    assert col[0] == 1
    assert col[1] == 0
    # Slot 2 unfilled (capacity grew implicitly? no — count is 2 here)


def test_metadata_bool_missing_sentinel_for_unfilled_slot():
    store = _make_store()
    store.ingest(
        0, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest(
        2, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest_metadata(0, {}, {}, metadata_bool={"flag": True})
    store.ingest_metadata(2, {}, {}, metadata_bool={"flag": False})
    col = store.metadata_bool("flag")
    assert col[0] == 1
    assert col[1] == _BOOL_MISSING
    assert col[2] == 0


def test_metadata_categorical_intern_and_lookup():
    store = _make_store()
    store.ingest(
        0, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest(
        1, record_metrics={"x": 2.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest(
        2, record_metrics={"x": 3.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest_metadata(0, {}, {}, metadata_categorical={"corr": "conv-A"})
    store.ingest_metadata(1, {}, {}, metadata_categorical={"corr": "conv-B"})
    store.ingest_metadata(2, {}, {}, metadata_categorical={"corr": "conv-A"})

    codes = store.metadata_categorical("corr")
    # First-seen "conv-A" -> 0; "conv-B" -> 1; repeat "conv-A" -> 0
    assert codes[0] == 0
    assert codes[1] == 1
    assert codes[2] == 0

    strings = store.metadata_category_strings("corr")
    assert strings[0] == "conv-A"
    assert strings[1] == "conv-B"


def test_metadata_categorical_missing_sentinel_for_uningested():
    store = _make_store()
    store.ingest(
        0, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest(
        1, record_metrics={"x": 2.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest_metadata(0, {}, {}, metadata_categorical={"corr": "v0"})
    codes = store.metadata_categorical("corr")
    assert codes[0] == 0
    assert codes[1] == _CATEGORICAL_MISSING


def test_unique_categorical_values_lists_seen_strings():
    store = _make_store()
    store.ingest(
        0, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest(
        1, record_metrics={"x": 2.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest_metadata(0, {}, {}, metadata_categorical={"g": "a"})
    store.ingest_metadata(1, {}, {}, metadata_categorical={"g": "b"})
    assert set(store.unique_categorical_values("g")) == {"a", "b"}


def test_mask_for_categorical_selects_matching_records():
    store = _make_store()
    for i in range(3):
        store.ingest(
            i,
            record_metrics={"x": float(i)},
            start_ns=0,
            end_ns=1,
            generation_start_ns=None,
        )
    store.ingest_metadata(0, {}, {}, metadata_categorical={"g": "alpha"})
    store.ingest_metadata(1, {}, {}, metadata_categorical={"g": "beta"})
    store.ingest_metadata(2, {}, {}, metadata_categorical={"g": "alpha"})

    mask = store.mask_for_categorical("g", "alpha")
    np.testing.assert_array_equal(mask, [True, False, True])


def test_mask_for_categorical_unknown_value_returns_all_false():
    store = _make_store()
    store.ingest(
        0, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    store.ingest_metadata(0, {}, {}, metadata_categorical={"g": "a"})
    mask = store.mask_for_categorical("g", "never_seen")
    assert mask.shape == (1,)
    assert not mask.any()


def test_mask_for_categorical_unknown_tag_returns_all_false():
    store = _make_store()
    store.ingest(
        0, record_metrics={"x": 1.0}, start_ns=0, end_ns=1, generation_start_ns=None
    )
    mask = store.mask_for_categorical("never_indexed", "v")
    assert mask.shape == (1,)
    assert not mask.any()


def test_query_time_range_selects_overlapping_records():
    store = _make_store()
    # Record 0: [0, 100], Record 1: [50, 200], Record 2: [300, 400]
    store.ingest(
        0, record_metrics={}, start_ns=0.0, end_ns=100.0, generation_start_ns=None
    )
    store.ingest(
        1, record_metrics={}, start_ns=50.0, end_ns=200.0, generation_start_ns=None
    )
    store.ingest(
        2, record_metrics={}, start_ns=300.0, end_ns=400.0, generation_start_ns=None
    )

    # Query window [75, 250] overlaps records 0 (end=100>=75) and 1 (50..200), not 2.
    mask = store.query_time_range(75.0, 250.0)
    np.testing.assert_array_equal(mask, [True, True, False])


def test_query_time_range_excludes_unfilled_slots():
    store = _make_store()
    store.ingest(
        0, record_metrics={}, start_ns=10.0, end_ns=20.0, generation_start_ns=None
    )
    store.ingest(
        2, record_metrics={}, start_ns=30.0, end_ns=40.0, generation_start_ns=None
    )
    # Slot 1 has NaN start_ns/end_ns — must NOT match any window.
    mask = store.query_time_range(0.0, 100.0)
    assert mask[0]
    assert not mask[1]
    assert mask[2]


def test_query_time_range_empty_store_returns_empty_mask():
    store = _make_store()
    mask = store.query_time_range(0.0, 100.0)
    assert mask.shape == (0,)
    assert mask.dtype == np.bool_


def test_ingest_mixed_numeric_string_list_in_one_record():
    store = _make_store()
    store.ingest(
        0,
        record_metrics={
            "lat_ns": 100.0,
            "model": "gpt-4",
            "icl": [1.0, 2.0],
        },
        start_ns=0,
        end_ns=10,
        generation_start_ns=5,
    )
    assert store.numeric("lat_ns")[0] == 100.0
    assert store.string("model")[0] == "gpt-4"
    np.testing.assert_array_equal(store.ragged("icl").values, [1.0, 2.0])


def test_ingest_skips_unsupported_value_types():
    store = _make_store()
    # dict is neither numeric, str, nor list — should be silently skipped.
    store.ingest(
        0,
        record_metrics={"weird": {"k": "v"}, "ok": 5.0},
        start_ns=0,
        end_ns=1,
        generation_start_ns=None,
    )
    assert store.numeric("ok")[0] == 5.0
    assert "weird" not in store.numeric_tags()
    assert "weird" not in store.ragged_tags()
