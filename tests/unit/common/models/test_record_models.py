# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel, Field, SerializeAsAny

from aiperf.common.enums import SSEFieldType
from aiperf.common.models import MetricResult, ProfileResults, SSEMessage
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


class TestSSEMessageDataclass:
    """Test that SSEMessage dataclass works correctly."""

    def test_parse_produces_valid_message(self) -> None:
        """parse() produces a fully usable SSEMessage."""
        msg = SSEMessage.parse("data: hello\nevent: message", perf_ns=42)
        assert msg.perf_ns == 42
        assert len(msg.packets) == 2
        assert msg.packets[0].name == SSEFieldType.DATA
        assert msg.packets[0].value == "hello"
        assert msg.packets[1].name == SSEFieldType.EVENT
        assert msg.packets[1].value == "message"

    def test_parse_returns_sse_message_instance(self) -> None:
        """parse() returns an SSEMessage instance."""
        msg = SSEMessage.parse("data: test", perf_ns=1)
        assert isinstance(msg, SSEMessage)

    def test_parse_empty_produces_no_packets(self) -> None:
        """Empty input yields zero packets."""
        msg = SSEMessage.parse("", perf_ns=0)
        assert msg.packets == []

    def test_parse_bytes_input(self) -> None:
        """parse() handles bytes input."""
        msg = SSEMessage.parse(b"data: from_bytes", perf_ns=99)
        assert msg.packets[0].value == "from_bytes"

    def test_pydantic_serialization_roundtrip(self) -> None:
        """SSEMessage roundtrips through Pydantic when inside a model field."""

        class Wrapper(BaseModel):
            responses: SerializeAsAny[list[SSEMessage]] = Field(default_factory=list)

        msg = SSEMessage.parse("data: roundtrip\nevent: test", perf_ns=123)
        wrapper = Wrapper(responses=[msg])
        json_bytes = wrapper.model_dump_json().encode()
        restored = Wrapper.model_validate_json(json_bytes)
        assert restored.responses[0].perf_ns == 123
        assert len(restored.responses[0].packets) == 2
        assert restored.responses[0].packets[0].value == "roundtrip"

    def test_parse_get_text_and_get_json(self) -> None:
        """Protocol methods work on dataclass instances."""
        msg = SSEMessage.parse('data: {"key": "value"}', perf_ns=1)
        assert msg.get_text() == '{"key": "value"}'
        json_obj = msg.get_json()
        assert json_obj == {"key": "value"}


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
    """Test that to_json_result preserves all stats including sum and count."""

    def test_record_metric_includes_count_and_sum(self) -> None:
        result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=50.0,
            min=10.0,
            max=90.0,
            sum=5000.0,
            count=100,
        )
        json_result = result.to_json_result()

        assert isinstance(json_result, JsonMetricResult)
        assert json_result.sum == 5000.0
        assert json_result.count == 100
        assert json_result.avg == 50.0
        assert json_result.min == 10.0
        assert json_result.max == 90.0

    def test_derived_metric_omits_count(self) -> None:
        """Derived/aggregate scalars get count=1 trivially; we suppress it."""
        result = MetricResult(
            tag="request_throughput",
            header="Request Throughput",
            unit="requests/sec",
            avg=1.5,
            sum=1.5,
            count=1,
        )
        json_result = result.to_json_result()

        assert json_result.count is None
        assert json_result.sum == 1.5
        assert json_result.avg == 1.5

    def test_aggregate_metric_omits_count(self) -> None:
        result = MetricResult(
            tag="request_count",
            header="Request Count",
            unit="requests",
            avg=20.0,
            count=1,
        )
        json_result = result.to_json_result()

        assert json_result.count is None
        assert json_result.avg == 20.0

    def test_unknown_tag_keeps_count(self) -> None:
        """Tags from other registries (e.g. GPU telemetry) keep count as-is."""
        result = MetricResult(
            tag="gpu_power_usage",
            header="GPU Power Usage",
            unit="W",
            avg=250.0,
            count=42,
        )
        json_result = result.to_json_result()

        assert json_result.count == 42

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
