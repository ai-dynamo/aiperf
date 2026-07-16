# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RaggedSeries — list-valued per-record metric storage."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import param

from aiperf.metrics.ragged_series import RaggedSeries


def test_init_empty_state_zero_values():
    series = RaggedSeries(initial_capacity=8, offsets_capacity=4)
    assert len(series.values) == 0
    assert len(series.record_indices) == 0
    assert (series.offsets == -1).all()


def test_extend_records_offsets_and_values():
    series = RaggedSeries(initial_capacity=8, offsets_capacity=4)
    series.extend(0, [1.0, 2.0, 3.0])
    series.extend(2, [10.0])

    np.testing.assert_array_equal(series.values, [1.0, 2.0, 3.0, 10.0])
    np.testing.assert_array_equal(series.record_indices, [0, 0, 0, 2])
    assert series.offsets[0] == 0
    assert series.offsets[1] == -1  # absent
    assert series.offsets[2] == 3


def test_extend_empty_list_is_noop():
    series = RaggedSeries(initial_capacity=4, offsets_capacity=4)
    series.extend(0, [])
    assert len(series.values) == 0
    assert series.offsets[0] == -1


def test_add_for_record_alias_matches_extend():
    series_a = RaggedSeries()
    series_b = RaggedSeries()
    series_a.extend(1, [4.0, 5.0])
    series_b.add_for_record(1, [4.0, 5.0])
    np.testing.assert_array_equal(series_a.values, series_b.values)
    np.testing.assert_array_equal(series_a.record_indices, series_b.record_indices)


def test_extend_grows_offsets_when_idx_exceeds_capacity():
    series = RaggedSeries(initial_capacity=8, offsets_capacity=4)
    # idx >= 4 must trigger doubling. Push to idx=10 (requires capacity 16).
    series.extend(10, [7.0])
    assert series.offsets.shape[0] >= 11
    assert series.offsets[10] == 0
    # Earlier slots preserved as -1
    assert (series.offsets[:10] == -1).all()


def test_get_values_for_mask_selects_records():
    series = RaggedSeries(initial_capacity=8, offsets_capacity=4)
    series.extend(0, [1.0, 2.0])
    series.extend(1, [3.0])
    series.extend(2, [4.0, 5.0, 6.0])

    mask = np.array([True, False, True])
    selected = series.get_values_for_mask(mask)
    np.testing.assert_array_equal(np.sort(selected), [1.0, 2.0, 4.0, 5.0, 6.0])


def test_get_values_for_mask_empty_returns_empty():
    series = RaggedSeries()
    out = series.get_values_for_mask(np.zeros(0, dtype=bool))
    assert out.shape == (0,)
    assert out.dtype == np.float64


def test_grouped_cumsum_resets_at_request_boundaries():
    series = RaggedSeries(initial_capacity=8, offsets_capacity=4)
    series.extend(0, [1.0, 2.0, 3.0])
    series.extend(1, [10.0, 20.0])

    cs = series.grouped_cumsum()
    # Within record 0: 1, 1+2, 1+2+3
    # Within record 1: 10, 10+20 (NOT continuing global)
    np.testing.assert_array_equal(cs, [1.0, 3.0, 6.0, 10.0, 30.0])


def test_grouped_cumsum_first_record_at_offset_zero():
    series = RaggedSeries(initial_capacity=4, offsets_capacity=4)
    series.extend(0, [5.0, 7.0])
    cs = series.grouped_cumsum()
    np.testing.assert_array_equal(cs, [5.0, 12.0])


def test_grouped_cumsum_empty_returns_empty():
    series = RaggedSeries()
    cs = series.grouped_cumsum()
    assert cs.shape == (0,)
    assert cs.dtype == np.float64


@pytest.mark.parametrize(
    "extends",
    [
        param([(0, [1.0]), (1, [2.0]), (2, [3.0])], id="three_singletons"),
        param([(0, [1.0, 2.0, 3.0])], id="single_record_three_values"),
        param([(5, [4.0]), (6, [5.0])], id="sparse_record_indices"),
    ],
)
def test_offsets_track_first_value_position(extends):
    series = RaggedSeries(initial_capacity=8, offsets_capacity=8)
    expected_offsets: dict[int, int] = {}
    running_len = 0
    for idx, vals in extends:
        if vals:
            expected_offsets[idx] = running_len
            running_len += len(vals)
        series.extend(idx, vals)

    for idx, off in expected_offsets.items():
        assert series.offsets[idx] == off


def test_supports_per_record_replay_flag_true():
    assert RaggedSeries.SUPPORTS_PER_RECORD_REPLAY is True
