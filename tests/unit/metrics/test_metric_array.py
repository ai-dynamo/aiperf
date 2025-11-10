# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from aiperf.metrics.metric_dicts import MetricArray


@pytest.fixture
def test_array():
    """Create an empty MetricArray for testing."""
    return MetricArray(initial_capacity=5)


def assert_array_equal(array: MetricArray, expected_values: list) -> None:
    """Assert that an array contains expected values."""
    assert array._size == len(expected_values)
    assert array._sum == sum(expected_values)
    np.testing.assert_array_equal(array.data, expected_values)


class TestMetricArray:
    """Test cases for MetricArray class."""

    def test_initialization_default_capacity(self):
        """Test array initialization with default capacity."""
        array = MetricArray()
        assert array._capacity == 10000
        assert array._size == 0
        assert array._sum == 0

    def test_initialization_custom_capacity(self):
        """Test array initialization with custom capacity."""
        array = MetricArray(initial_capacity=100)
        assert array._capacity == 100
        assert array._size == 0

    def test_append(self, test_array: MetricArray):
        """Test appending single values."""
        test_array.append(1.0)
        assert_array_equal(test_array, [1.0])

        test_array.append(2.5)
        assert_array_equal(test_array, [1.0, 2.5])

    def test_extend(self, test_array: MetricArray):
        """Test extending with multiple values."""
        test_array.extend([1.0, 2.0, 3.0])
        assert_array_equal(test_array, [1.0, 2.0, 3.0])

        test_array.extend([4.0, 5.0])
        assert_array_equal(test_array, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_mixed_operations(self, test_array: MetricArray):
        """Test mixing append and extend operations."""
        test_array.append(1.0)
        test_array.extend([2.0, 3.0])
        test_array.append(4.0)

        assert_array_equal(test_array, [1.0, 2.0, 3.0, 4.0])

    def test_resize_on_capacity_exceeded(self):
        """Test that array resizes when capacity is exceeded."""
        array = MetricArray(initial_capacity=2)
        array.extend([1.0, 2.0])
        assert array._capacity == 2

        # This should trigger resize
        array.append(3.0)
        assert array._capacity == 4
        assert_array_equal(array, [1.0, 2.0, 3.0])

    def test_properties(self, test_array: MetricArray):
        """Test sum and data properties."""
        values = [1.0, 2.0, 3.0, 4.0]
        test_array.extend(values)
        assert test_array.sum == 10.0
        assert len(test_array.data) == 4
        np.testing.assert_array_equal(test_array.data, values)

    @pytest.mark.parametrize(
        "values,expected_min,expected_max,expected_avg,p50",
        [
            ([5.0], 5.0, 5.0, 5.0, 5.0),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 1.0, 5.0, 3.0, 3.0),
            ([10.0, 20.0, 30.0, 40.0, 50.0], 10.0, 50.0, 30.0, 30.0),
        ],
    )
    def test_to_result_statistics(
        self,
        values: list[float],
        expected_min: float,
        expected_max: float,
        expected_avg: float,
        p50: float,
    ):
        """Test to_result method computes statistics correctly."""
        array = MetricArray()
        array.extend(values)

        result = array.to_result("test", "Test Metric", "ms")

        assert result.tag == "test"
        assert result.header == "Test Metric"
        assert result.unit == "ms"
        assert result.min == expected_min
        assert result.max == expected_max
        assert result.avg == expected_avg
        assert result.p50 == p50
        assert result.count == len(values)


class TestMetricArrayEdgeCases:
    """Test edge cases for MetricArray."""

    def test_array_to_result_raises_error(self):
        """Test that to_result raises error for empty array."""
        with pytest.raises(IndexError):
            MetricArray().to_result("empty", "Empty Metric", "units")

    @pytest.mark.parametrize("invalid_capacity", [-1, 0])
    def test_invalid_initial_capacity(self, invalid_capacity: int):
        """Test behavior with invalid initial capacity."""
        with pytest.raises(ValueError):
            MetricArray(initial_capacity=invalid_capacity)

    def test_large_batch_resize(self):
        """Test resize behavior when adding large batch."""
        array = MetricArray(initial_capacity=2)
        large_batch = list(range(1, 11))

        array.extend(large_batch)

        assert array._capacity == 10
        assert_array_equal(array, large_batch)
