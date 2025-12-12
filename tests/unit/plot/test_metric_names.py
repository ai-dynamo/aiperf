# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for metric names utility module.

Tests for metric display name functions in aiperf.plot.metric_names module.
"""

from aiperf.plot.metric_names import (
    get_aggregated_metrics,
    get_all_metric_display_names,
    get_gpu_metric_unit,
    get_gpu_metrics,
    get_metric_display_name,
    get_metric_display_name_with_unit,
    get_request_metrics,
    get_timeslice_metrics,
)


class TestGetAllMetricDisplayNames:
    """Tests for get_all_metric_display_names function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_all_metric_display_names()
        assert isinstance(result, dict)

    def test_dict_is_not_empty(self):
        """Test that returned dictionary contains entries."""
        result = get_all_metric_display_names()
        assert len(result) > 0

    def test_contains_known_metrics(self):
        """Test that dictionary contains known standard metrics."""
        result = get_all_metric_display_names()
        assert "time_to_first_token" in result
        assert "request_latency" in result

    def test_contains_gpu_metrics(self):
        """Test that dictionary contains GPU telemetry metrics."""
        result = get_all_metric_display_names()
        gpu_metrics = [key for key in result if "gpu" in key.lower()]
        assert len(gpu_metrics) > 0

    def test_contains_derived_metrics(self):
        """Test that dictionary contains derived metrics."""
        result = get_all_metric_display_names()
        assert "output_token_throughput_per_gpu" in result

    def test_all_values_are_strings(self):
        """Test that all display names are strings."""
        result = get_all_metric_display_names()
        for value in result.values():
            assert isinstance(value, str)
            assert len(value) > 0


class TestGetMetricDisplayName:
    """Tests for get_metric_display_name function."""

    def test_returns_known_metric_display_name(self):
        """Test that function returns correct display name for known metric."""
        result = get_metric_display_name("time_to_first_token")
        assert result == "Time to First Token"

    def test_returns_title_cased_fallback_for_unknown_metric(self):
        """Test that function returns title-cased tag for unknown metric."""
        result = get_metric_display_name("unknown_metric_name")
        assert result == "Unknown Metric Name"

    def test_handles_underscore_replacement_in_fallback(self):
        """Test that underscores are replaced with spaces in fallback."""
        result = get_metric_display_name("my_custom_metric")
        assert "_" not in result
        assert " " in result
        assert result[0].isupper()

    def test_preserves_known_metric_formatting(self):
        """Test that known metrics preserve their registered formatting."""
        result = get_metric_display_name("request_latency")
        assert result == "Request Latency"

    def test_handles_empty_string(self):
        """Test that function handles empty string input."""
        result = get_metric_display_name("")
        assert isinstance(result, str)

    def test_handles_single_word(self):
        """Test that function handles single word without underscores."""
        result = get_metric_display_name("latency")
        assert result == "Latency"


class TestGetAggregatedMetrics:
    """Tests for get_aggregated_metrics function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_aggregated_metrics()
        assert isinstance(result, list)

    def test_list_is_not_empty(self):
        """Test that returned list contains entries."""
        result = get_aggregated_metrics()
        assert len(result) > 0

    def test_contains_expected_metrics(self):
        """Test that list contains expected aggregated metrics."""
        result = get_aggregated_metrics()
        assert "request_latency" in result

    def test_contains_derived_metrics(self):
        """Test that list contains derived metrics."""
        result = get_aggregated_metrics()
        assert "output_token_throughput_per_gpu" in result

    def test_all_items_are_strings(self):
        """Test that all metric tags are strings."""
        result = get_aggregated_metrics()
        for item in result:
            assert isinstance(item, str)
            assert len(item) > 0


class TestGetRequestMetrics:
    """Tests for get_request_metrics function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_request_metrics()
        assert isinstance(result, list)

    def test_list_is_not_empty(self):
        """Test that returned list contains entries."""
        result = get_request_metrics()
        assert len(result) > 0

    def test_contains_expected_metrics(self):
        """Test that list contains expected per-request metrics."""
        result = get_request_metrics()
        assert "request_number" in result
        assert "timestamp" in result
        assert "request_latency" in result

    def test_contains_computed_columns(self):
        """Test that list contains computed columns."""
        result = get_request_metrics()
        assert "timestamp_s" in result
        assert "throughput_tokens_per_sec" in result
        assert "active_requests" in result

    def test_all_items_are_strings(self):
        """Test that all metric/column names are strings."""
        result = get_request_metrics()
        for item in result:
            assert isinstance(item, str)
            assert len(item) > 0


class TestGetTimesliceMetrics:
    """Tests for get_timeslice_metrics function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_timeslice_metrics()
        assert isinstance(result, list)

    def test_returns_exactly_expected_values(self):
        """Test that function returns exactly the expected timeslice metrics."""
        result = get_timeslice_metrics()
        expected = [
            "Timeslice",
            "Time to First Token",
            "Inter Token Latency",
            "Request Throughput",
            "Request Latency",
        ]
        assert result == expected

    def test_all_items_are_display_names(self):
        """Test that all items are properly formatted display names."""
        result = get_timeslice_metrics()
        for item in result:
            assert isinstance(item, str)
            assert len(item) > 0
            assert item[0].isupper()


class TestGetGpuMetrics:
    """Tests for get_gpu_metrics function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_gpu_metrics()
        assert isinstance(result, list)

    def test_all_items_are_strings(self):
        """Test that all GPU metric field names are strings."""
        result = get_gpu_metrics()
        for item in result:
            assert isinstance(item, str)
            assert len(item) > 0

    def test_contains_gpu_related_metrics(self):
        """Test that list contains GPU-related field names."""
        result = get_gpu_metrics()
        gpu_related = [m for m in result if "gpu" in m.lower()]
        assert len(gpu_related) > 0


class TestGetGpuMetricUnit:
    """Tests for get_gpu_metric_unit function."""

    def test_returns_unit_string_for_valid_gpu_metric(self):
        """Test that function returns unit string for valid GPU metric."""
        gpu_metrics = get_gpu_metrics()
        if gpu_metrics:
            result = get_gpu_metric_unit(gpu_metrics[0])
            assert result is None or isinstance(result, str)

    def test_returns_none_for_invalid_metric(self):
        """Test that function returns None for non-GPU metric."""
        result = get_gpu_metric_unit("invalid_metric_name")
        assert result is None

    def test_returns_none_for_standard_metric(self):
        """Test that function returns None for standard non-GPU metric."""
        result = get_gpu_metric_unit("request_latency")
        assert result is None

    def test_handles_empty_string(self):
        """Test that function handles empty string input."""
        result = get_gpu_metric_unit("")
        assert result is None


class TestGetMetricDisplayNameWithUnit:
    """Tests for get_metric_display_name_with_unit function."""

    def test_returns_display_name_with_unit_for_gpu_metric(self):
        """Test that function returns display name with unit in parentheses for GPU metric."""
        gpu_metrics = get_gpu_metrics()
        if gpu_metrics:
            result = get_metric_display_name_with_unit(gpu_metrics[0])
            assert isinstance(result, str)
            assert len(result) > 0

    def test_returns_display_name_without_unit_for_non_gpu_metric(self):
        """Test that function returns display name without unit for non-GPU metric."""
        result = get_metric_display_name_with_unit("request_latency")
        assert result == "Request Latency"
        assert "(" not in result

    def test_adds_percentage_for_utilization_metrics(self):
        """Test that function adds (%) for metrics containing 'utilization'."""
        result = get_metric_display_name_with_unit("some_utilization_metric")
        assert "(%)" in result

    def test_handles_gpu_utilization_metric(self):
        """Test that GPU utilization metrics get percentage unit."""
        result = get_metric_display_name_with_unit("gpu_utilization")
        assert "(%)" in result or result.endswith("%)")

    def test_unit_in_parentheses_format(self):
        """Test that unit is formatted in parentheses."""
        gpu_metrics = get_gpu_metrics()
        for metric in gpu_metrics:
            result = get_metric_display_name_with_unit(metric)
            if "(" in result:
                assert result.count("(") == 1
                assert result.count(")") == 1
                assert result.endswith(")")

    def test_handles_unknown_metric(self):
        """Test that function handles unknown metric with fallback."""
        result = get_metric_display_name_with_unit("unknown_metric")
        assert result == "Unknown Metric"

    def test_returns_string(self):
        """Test that function always returns a string."""
        result = get_metric_display_name_with_unit("any_metric")
        assert isinstance(result, str)
        assert len(result) > 0
