# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import MetricResult, ProfileResults
from aiperf.common.models.export_models import JsonMetricResult


class TestProfileResults:
    """Test cases for ProfileResults model."""

    def test_profile_results_with_timeslice_metric_results(self):
        """Test ProfileResults can store timeslice metric results."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        timeslice_results = {
            0: [metric_result],
            1: [metric_result],
        }

        profile_results = ProfileResults(
            records=[metric_result],
            timeslice_metric_results=timeslice_results,
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert 0 in profile_results.timeslice_metric_results
        assert 1 in profile_results.timeslice_metric_results
        assert len(profile_results.timeslice_metric_results[0]) == 1
        assert len(profile_results.timeslice_metric_results[1]) == 1

    def test_profile_results_without_timeslice_metric_results(self):
        """Test ProfileResults works without timeslice metric results."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        profile_results = ProfileResults(
            records=[metric_result],
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is None

    def test_profile_results_with_empty_timeslice_dict(self):
        """Test ProfileResults with empty timeslice results dict."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        profile_results = ProfileResults(
            records=[metric_result],
            timeslice_metric_results={},
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert len(profile_results.timeslice_metric_results) == 0

    def test_profile_results_with_multiple_timeslices_and_metrics(self):
        """Test ProfileResults with multiple timeslices containing multiple metrics."""
        latency_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        throughput_result = MetricResult(
            tag="request_throughput",
            header="Request Throughput",
            unit="requests/sec",
            avg=50.0,
            count=1,
        )

        timeslice_results = {
            0: [latency_result, throughput_result],
            1: [latency_result, throughput_result],
            2: [latency_result, throughput_result],
        }

        profile_results = ProfileResults(
            records=[latency_result, throughput_result],
            timeslice_metric_results=timeslice_results,
            completed=2,
            start_ns=1000000000,
            end_ns=3000000000,
        )

        assert profile_results.timeslice_metric_results is not None
        assert len(profile_results.timeslice_metric_results) == 3
        for i in range(3):
            assert i in profile_results.timeslice_metric_results
            assert len(profile_results.timeslice_metric_results[i]) == 2


class TestMetricResultSumField:
    """Test the sum field on MetricResult."""

    def test_sum_field_stored(self) -> None:
        result = MetricResult(
            tag="total_tokens",
            header="Total Tokens",
            unit="tokens",
            avg=100.0,
            sum=5000.0,
            count=50,
        )
        assert result.sum == 5000.0

    def test_sum_field_defaults_to_none(self) -> None:
        result = MetricResult(tag="latency", header="Latency", unit="ms", avg=10.0)
        assert result.sum is None

    @pytest.mark.parametrize(
        "sum_value",
        [0, 42, 1_000_000, 3.14, -1.0],
        ids=["zero", "int", "large", "float", "negative"],
    )
    def test_sum_accepts_numeric_types(self, sum_value) -> None:
        result = MetricResult(tag="metric", header="M", unit="u", sum=sum_value)
        assert result.sum == sum_value


class TestMetricResultToJsonResult:
    """Test that to_json_result excludes sum."""

    def test_sum_excluded_from_json_result(self) -> None:
        result = MetricResult(
            tag="throughput",
            header="Throughput",
            unit="req/s",
            avg=50.0,
            min=10.0,
            max=90.0,
            sum=5000.0,
            count=100,
        )
        json_result = result.to_json_result()

        assert isinstance(json_result, JsonMetricResult)
        assert not hasattr(json_result, "sum")
        assert json_result.avg == 50.0
        assert json_result.min == 10.0
        assert json_result.max == 90.0

    def test_stat_keys_preserved_in_json_result(self) -> None:
        result = MetricResult(
            tag="latency",
            header="Latency",
            unit="ms",
            avg=100.0,
            p50=95.0,
            p99=200.0,
            min=10.0,
            max=300.0,
            std=25.0,
        )
        json_result = result.to_json_result()

        assert json_result.avg == 100.0
        assert json_result.p50 == 95.0
        assert json_result.p99 == 200.0
        assert json_result.min == 10.0
        assert json_result.max == 300.0
        assert json_result.std == 25.0
