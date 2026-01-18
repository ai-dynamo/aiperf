# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GrowableArray and Growable2DArray."""

import numpy as np
import pytest

from aiperf.common.growable_array import Growable2DArray, GrowableArray


class TestGrowableArray:
    """Tests for GrowableArray."""

    def test_init_default(self) -> None:
        arr = GrowableArray()
        assert len(arr) == 0
        assert arr.capacity == 256
        assert arr.data.dtype == np.float64
        assert arr.sum is None

    def test_init_custom_capacity_and_dtype(self) -> None:
        arr = GrowableArray(initial_capacity=100, dtype=np.int64)
        assert arr.capacity == 100
        assert arr.data.dtype == np.int64

    def test_init_invalid_capacity(self) -> None:
        with pytest.raises(ValueError, match="initial_capacity must be positive"):
            GrowableArray(initial_capacity=0)
        with pytest.raises(ValueError, match="initial_capacity must be positive"):
            GrowableArray(initial_capacity=-1)

    def test_append_single(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.append(1.5)
        assert len(arr) == 1
        assert arr.data[0] == 1.5

    def test_append_multiple_no_growth(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        for i in range(4):
            arr.append(float(i))
        assert len(arr) == 4
        assert arr.capacity == 4
        np.testing.assert_array_equal(arr.data, [0.0, 1.0, 2.0, 3.0])

    def test_append_triggers_growth(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        for i in range(5):
            arr.append(float(i))
        assert len(arr) == 5
        assert arr.capacity == 8  # Doubled
        np.testing.assert_array_equal(arr.data, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_append_multiple_growth_cycles(self) -> None:
        arr = GrowableArray(initial_capacity=2)
        for i in range(100):
            arr.append(float(i))
        assert len(arr) == 100
        assert arr.capacity == 128  # 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128
        np.testing.assert_array_equal(arr.data, np.arange(100, dtype=np.float64))

    def test_extend(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.append(0.0)
        arr.extend(np.array([1.0, 2.0, 3.0]))
        assert len(arr) == 4
        np.testing.assert_array_equal(arr.data, [0.0, 1.0, 2.0, 3.0])

    def test_extend_triggers_growth(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.extend(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert len(arr) == 5
        assert arr.capacity == 8

    def test_extend_empty_array(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.append(1.0)
        arr.extend(np.array([]))
        assert len(arr) == 1

    def test_extend_large_batch(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.extend(np.arange(100, dtype=np.float64))
        assert len(arr) == 100
        assert arr.capacity == 128

    def test_data_returns_view(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.append(1.0)
        arr.append(2.0)
        data = arr.data
        assert len(data) == 2
        # Verify it's a view (not a copy)
        data[0] = 99.0
        assert arr.data[0] == 99.0

    def test_clear(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        for i in range(10):
            arr.append(float(i))
        old_capacity = arr.capacity
        arr.clear()
        assert len(arr) == 0
        assert arr.capacity == old_capacity  # Capacity preserved

    def test_int64_dtype(self) -> None:
        arr = GrowableArray(initial_capacity=4, dtype=np.int64)
        arr.append(42)
        arr.append(100)
        assert arr.data.dtype == np.int64
        np.testing.assert_array_equal(arr.data, [42, 100])


class TestGrowableArraySum:
    """Tests for GrowableArray sum tracking."""

    def test_track_sum_disabled_by_default(self) -> None:
        arr = GrowableArray()
        arr.append(1.0)
        assert arr.sum is None
        assert arr.mean is None

    def test_track_sum_enabled(self) -> None:
        arr = GrowableArray(track_sum=True)
        assert arr.sum == 0.0
        arr.append(1.0)
        arr.append(2.0)
        arr.append(3.0)
        assert arr.sum == 6.0

    def test_mean_calculation(self) -> None:
        arr = GrowableArray(track_sum=True)
        arr.append(2.0)
        arr.append(4.0)
        arr.append(6.0)
        assert arr.mean == 4.0

    def test_mean_empty_array(self) -> None:
        arr = GrowableArray(track_sum=True)
        assert arr.mean is None

    def test_sum_with_extend(self) -> None:
        arr = GrowableArray(track_sum=True)
        arr.append(1.0)
        arr.extend(np.array([2.0, 3.0, 4.0]))
        assert arr.sum == 10.0
        assert arr.mean == 2.5

    def test_clear_resets_sum(self) -> None:
        arr = GrowableArray(track_sum=True)
        arr.append(1.0)
        arr.append(2.0)
        assert arr.sum == 3.0
        arr.clear()
        assert arr.sum == 0.0
        assert arr.mean is None


class TestGrowable2DArray:
    """Tests for Growable2DArray."""

    def test_init_default(self) -> None:
        arr = Growable2DArray(n_columns=3)
        assert len(arr) == 0
        assert arr.capacity == 256
        assert arr.n_columns == 3
        assert arr.data.dtype == np.float64

    def test_init_custom(self) -> None:
        arr = Growable2DArray(n_columns=5, initial_capacity=100, dtype=np.int32)
        assert arr.capacity == 100
        assert arr.n_columns == 5
        assert arr.data.dtype == np.int32

    def test_init_invalid_capacity(self) -> None:
        with pytest.raises(ValueError, match="initial_capacity must be positive"):
            Growable2DArray(n_columns=3, initial_capacity=0)

    def test_init_invalid_columns(self) -> None:
        with pytest.raises(ValueError, match="n_columns must be positive"):
            Growable2DArray(n_columns=0)

    def test_append_single_row(self) -> None:
        arr = Growable2DArray(n_columns=3, initial_capacity=4)
        arr.append(np.array([1.0, 2.0, 3.0]))
        assert len(arr) == 1
        np.testing.assert_array_equal(arr.data[0], [1.0, 2.0, 3.0])

    def test_append_multiple_rows_no_growth(self) -> None:
        arr = Growable2DArray(n_columns=2, initial_capacity=4)
        arr.append(np.array([1.0, 2.0]))
        arr.append(np.array([3.0, 4.0]))
        arr.append(np.array([5.0, 6.0]))
        assert len(arr) == 3
        assert arr.capacity == 4
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(arr.data, expected)

    def test_append_triggers_growth(self) -> None:
        arr = Growable2DArray(n_columns=2, initial_capacity=2)
        arr.append(np.array([1.0, 2.0]))
        arr.append(np.array([3.0, 4.0]))
        arr.append(np.array([5.0, 6.0]))  # Triggers growth
        assert len(arr) == 3
        assert arr.capacity == 4
        assert arr.n_columns == 2  # Columns unchanged

    def test_data_returns_view(self) -> None:
        arr = Growable2DArray(n_columns=2, initial_capacity=4)
        arr.append(np.array([1.0, 2.0]))
        data = arr.data
        data[0, 0] = 99.0
        assert arr.data[0, 0] == 99.0

    def test_clear(self) -> None:
        arr = Growable2DArray(n_columns=3, initial_capacity=4)
        for i in range(10):
            arr.append(np.array([float(i), float(i + 1), float(i + 2)]))
        old_capacity = arr.capacity
        arr.clear()
        assert len(arr) == 0
        assert arr.capacity == old_capacity
        assert arr.n_columns == 3


class TestGrowableArrayIndexing:
    """Tests for GrowableArray indexing."""

    def test_getitem_positive_index(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.extend(np.array([10.0, 20.0, 30.0]))
        assert arr[0] == 10.0
        assert arr[1] == 20.0
        assert arr[2] == 30.0

    def test_getitem_negative_index(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.extend(np.array([10.0, 20.0, 30.0]))
        assert arr[-1] == 30.0
        assert arr[-2] == 20.0
        assert arr[-3] == 10.0

    def test_getitem_slice(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.extend(np.array([10.0, 20.0, 30.0, 40.0]))
        np.testing.assert_array_equal(arr[1:3], [20.0, 30.0])
        np.testing.assert_array_equal(arr[:2], [10.0, 20.0])
        np.testing.assert_array_equal(arr[::2], [10.0, 30.0])

    def test_setitem_positive_index(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.extend(np.array([10.0, 20.0, 30.0]))
        arr[1] = 99.0
        assert arr[1] == 99.0

    def test_setitem_negative_index(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.extend(np.array([10.0, 20.0, 30.0]))
        arr[-1] = 99.0
        assert arr[2] == 99.0

    def test_setitem_updates_sum(self) -> None:
        arr = GrowableArray(initial_capacity=4, track_sum=True)
        arr.extend(np.array([10.0, 20.0, 30.0]))
        assert arr.sum == 60.0
        arr[1] = 50.0  # Replace 20 with 50
        assert arr.sum == 90.0  # 10 + 50 + 30

    def test_setitem_out_of_range(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        arr.append(1.0)
        with pytest.raises(IndexError):
            arr[5] = 99.0
        with pytest.raises(IndexError):
            arr[-5] = 99.0


class TestGrowable2DArrayIndexing:
    """Tests for Growable2DArray indexing."""

    def test_getitem_positive_index(self) -> None:
        arr = Growable2DArray(n_columns=2, initial_capacity=4)
        arr.append(np.array([1.0, 2.0]))
        arr.append(np.array([3.0, 4.0]))
        np.testing.assert_array_equal(arr[0], [1.0, 2.0])
        np.testing.assert_array_equal(arr[1], [3.0, 4.0])

    def test_getitem_negative_index(self) -> None:
        arr = Growable2DArray(n_columns=2, initial_capacity=4)
        arr.append(np.array([1.0, 2.0]))
        arr.append(np.array([3.0, 4.0]))
        np.testing.assert_array_equal(arr[-1], [3.0, 4.0])

    def test_getitem_slice(self) -> None:
        arr = Growable2DArray(n_columns=2, initial_capacity=4)
        arr.append(np.array([1.0, 2.0]))
        arr.append(np.array([3.0, 4.0]))
        arr.append(np.array([5.0, 6.0]))
        result = arr[1:]
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_setitem(self) -> None:
        arr = Growable2DArray(n_columns=2, initial_capacity=4)
        arr.append(np.array([1.0, 2.0]))
        arr.append(np.array([3.0, 4.0]))
        arr[0] = np.array([99.0, 98.0])
        np.testing.assert_array_equal(arr[0], [99.0, 98.0])

    def test_setitem_out_of_range(self) -> None:
        arr = Growable2DArray(n_columns=2, initial_capacity=4)
        arr.append(np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            arr[5] = np.array([99.0, 98.0])


class TestGrowableArrayEdgeCases:
    """Edge case tests."""

    def test_very_small_initial_capacity(self) -> None:
        arr = GrowableArray(initial_capacity=1)
        for i in range(1000):
            arr.append(float(i))
        assert len(arr) == 1000
        np.testing.assert_array_equal(arr.data, np.arange(1000, dtype=np.float64))

    def test_data_shape_consistency(self) -> None:
        arr = GrowableArray(initial_capacity=4)
        assert arr.data.shape == (0,)
        arr.append(1.0)
        assert arr.data.shape == (1,)
        arr.append(2.0)
        assert arr.data.shape == (2,)

    def test_2d_data_shape_consistency(self) -> None:
        arr = Growable2DArray(n_columns=3, initial_capacity=4)
        assert arr.data.shape == (0, 3)
        arr.append(np.array([1.0, 2.0, 3.0]))
        assert arr.data.shape == (1, 3)
