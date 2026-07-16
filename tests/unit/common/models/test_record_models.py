# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel, Field, SerializeAsAny

from aiperf.common.enums import SSEFieldType
from aiperf.common.models import (
    MetricResult,
    ProfileResults,
    SSEMessage,
    TimesliceResult,
)
from aiperf.common.models.export_models import JsonMetricResult


class TestProfileResults:
    """Test cases for ProfileResults model."""

    def test_profile_results_timeslices_preserves_metric_lookup(self) -> None:
        """Test ProfileResults stores accumulator-backed timeslices."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )
        timeslices = [
            TimesliceResult(
                start_ns=1_000_000_000,
                end_ns=2_000_000_000,
                metric_results=[metric_result],
            ),
            TimesliceResult(
                start_ns=2_000_000_000,
                end_ns=3_000_000_000,
                metric_results=[metric_result],
            ),
        ]

        profile_results = ProfileResults(
            records=[metric_result],
            timeslices=timeslices,
            completed=1,
            start_ns=1_000_000_000,
            end_ns=3_000_000_000,
        )

        assert profile_results.timeslices is not None
        assert len(profile_results.timeslices) == 2
        assert (
            profile_results.timeslices[0].metric_results["request_latency"]
            is metric_result
        )
        assert (
            profile_results.timeslices[1].metric_results["request_latency"]
            is metric_result
        )

    def test_profile_results_no_timeslices_defaults_to_none(self) -> None:
        """Test ProfileResults works without timeslice results."""
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
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )

        assert profile_results.timeslices is None

    def test_profile_results_empty_timeslices_preserves_empty_list(self) -> None:
        """Test ProfileResults with empty timeslice list."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        profile_results = ProfileResults(
            records=[metric_result],
            timeslices=[],
            completed=1,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )

        assert profile_results.timeslices == []

    def test_profile_results_multiple_timeslices_maps_metrics_by_tag(self) -> None:
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

        timeslices = [
            TimesliceResult(
                start_ns=1_000_000_000 + i * 1_000_000_000,
                end_ns=2_000_000_000 + i * 1_000_000_000,
                metric_results=[latency_result, throughput_result],
            )
            for i in range(3)
        ]

        profile_results = ProfileResults(
            records=[latency_result, throughput_result],
            timeslices=timeslices,
            completed=2,
            start_ns=1_000_000_000,
            end_ns=4_000_000_000,
        )

        assert profile_results.timeslices is not None
        assert len(profile_results.timeslices) == 3
        for timeslice in profile_results.timeslices:
            assert set(timeslice.metric_results) == {
                "request_latency",
                "request_throughput",
            }


class TestRecordDataStrictRouting:
    """RecordData routes by record_type and raises on an unregistered value.

    ``strict_routing=True`` replaces AutoRoutedModel's base-class fallback: the
    base RecordData has no standalone shape, so an unknown record_type must fail
    loudly instead of degrading to a bare instance with every typed field dropped.
    """

    def test_unknown_record_type_raises(self) -> None:
        from aiperf.common.models.record_models import RecordData

        with pytest.raises(ValueError, match="Unknown record_type 'does_not_exist'"):
            RecordData.from_json({"record_type": "does_not_exist"})

    def test_registered_record_type_routes_to_subclass(self) -> None:
        # Importing the module registers the subclass via __init_subclass__.
        from aiperf.accuracy.models import AccuracyRecordsData
        from aiperf.common.enums import CreditPhase
        from aiperf.common.models.record_models import RecordData

        original = AccuracyRecordsData(
            session_num=0,
            worker_id="w1",
            benchmark_phase=CreditPhase.PROFILING,
            timestamp_ns=1_000,
            grader_name="multiple_choice",
            passed=True,
            confidence=1.0,
            expected="A",
            actual="A",
            explanation="ok",
        )

        routed = RecordData.from_json(original.model_dump())

        assert isinstance(routed, AccuracyRecordsData)
        assert routed.expected == "A"
        assert routed.grader_name == "multiple_choice"

    def test_strict_routing_raises_with_empty_lookup_table(self) -> None:
        # Even when NO subclass is registered (producing module not imported yet),
        # a strict hierarchy must raise rather than silently degrade to the base.
        from aiperf.common.models.auto_routed_model import AutoRoutedModel

        class _StrictRoot(AutoRoutedModel):
            discriminator_field = "kind"
            strict_routing = True
            kind: str

        assert _StrictRoot._model_lookup_table == {}  # nothing registered
        with pytest.raises(ValueError, match="Unknown kind 'nope'"):
            _StrictRoot.from_json({"kind": "nope"})


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
