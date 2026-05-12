# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
from queue import Full
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.models import CreditPhaseStats
from aiperf.common.optional_dependencies import OTEL_METRICS_STREAMING_FEATURE
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.otel_metrics_results_processor import (
    OTelMetricsResultsProcessor,
)
from tests.unit.post_processors.conftest import create_metric_records_message


@pytest.fixture
def user_config_otel(tmp_artifact_dir) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        ),
        output=OutputConfig(
            artifact_directory=tmp_artifact_dir,
        ),
        otel_url="collector:4318",
    )


@pytest.fixture
def user_config_otel_mlflow(tmp_artifact_dir) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        ),
        output=OutputConfig(
            artifact_directory=tmp_artifact_dir,
        ),
        otel_url="collector:4318",
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment="aiperf-tests",
    )


@pytest.fixture
def user_config_mlflow_only(tmp_artifact_dir) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        ),
        output=OutputConfig(
            artifact_directory=tmp_artifact_dir,
        ),
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment="aiperf-tests",
    )


_ORIGINAL_IMPORT = builtins.__import__


def _import_side_effect_for_otel(name: str, *args: Any, **kwargs: Any) -> Any:
    """Raise ImportError for opentelemetry imports, delegate all others."""
    if name.startswith("opentelemetry"):
        raise ImportError("opentelemetry intentionally unavailable in test")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


class _FakeQueue:
    """Fake multiprocessing queue that captures events for test assertions."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.closed = False

    def put_nowait(self, event: dict[str, object]) -> None:
        self.events.append(event)

    def get_nowait(self) -> dict[str, object]:
        return self.events.pop(0)

    def close(self) -> None:
        self.closed = True


def _setup_fanout_processor(
    processor: OTelMetricsResultsProcessor,
) -> _FakeQueue:
    """Configure processor to use a fake fanout queue for testing."""
    fake_queue = _FakeQueue()
    processor._streaming_ready = True
    processor._fanout_queue = fake_queue
    return fake_queue


class TestOTelMetricsResultsProcessor:
    def test_disabled_without_otel_or_mlflow(
        self, service_config: ServiceConfig
    ) -> None:
        user_config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
            )
        )
        with pytest.raises(PostProcessorDisabled):
            OTelMetricsResultsProcessor(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config,
            )

    def test_enabled_with_mlflow_without_otel_url(
        self,
        service_config: ServiceConfig,
        user_config_mlflow_only: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_mlflow_only,
        )
        assert processor._otel_metrics_url is None
        assert processor._mlflow_live_enabled is True

    def test_mlflow_only_does_not_require_otel_imports(
        self,
        service_config: ServiceConfig,
        user_config_mlflow_only: UserConfig,
    ) -> None:
        with patch("builtins.__import__", side_effect=_import_side_effect_for_otel):
            processor = OTelMetricsResultsProcessor(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config_mlflow_only,
            )
        assert processor._mlflow_live_enabled is True

    def test_init_dependency_failure_raises_post_processor_disabled(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        with (
            patch("builtins.__import__", side_effect=_import_side_effect_for_otel),
            pytest.raises(PostProcessorDisabled) as exc_info,
        ):
            OTelMetricsResultsProcessor(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config_otel,
            )
        assert OTEL_METRICS_STREAMING_FEATURE in str(exc_info.value)

    def test_init_otel_import_failure_falls_back_to_mlflow_only(
        self,
        service_config: ServiceConfig,
        user_config_otel_mlflow: UserConfig,
    ) -> None:
        """When both sinks are configured but OTel imports fail, MLflow live
        streaming must still be constructed. Regression: the parent-side OTel
        import check previously disabled the entire fanout processor, dropping
        MLflow live metrics even though MLflow was independently usable.
        """
        with patch("builtins.__import__", side_effect=_import_side_effect_for_otel):
            processor = OTelMetricsResultsProcessor(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config_otel_mlflow,
            )
        assert processor._otel_metrics_url is None
        assert processor._mlflow_live_enabled is True

    @pytest.mark.asyncio
    async def test_process_result_records_histogram_values_by_metric(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        fake_queue = _setup_fanout_processor(processor)

        metric_record = create_metric_records_message(
            results=[
                {
                    "request_latency_ns": 123_000_000,
                    "request_count": 1,
                    "tokens_per_response": [1, 2, 3],
                }
            ]
        ).to_data()
        await processor.process_result(metric_record)

        histogram_events = [
            e for e in fake_queue.events if e.get("type") == "histogram_record"
        ]
        # Should have emitted histograms for all numeric metrics
        # (exact names depend on GenAI semconv translation)
        assert len(histogram_events) >= 3
        # Verify events contain expected structure
        for event in histogram_events:
            assert "metric_name" in event["payload"]  # type: ignore[operator]
            assert "value" in event["payload"]  # type: ignore[operator]
            assert "attributes" in event["payload"]  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_process_result_skips_metrics_when_metrics_telemetry_disabled(
        self,
        service_config: ServiceConfig,
        tmp_artifact_dir,
    ) -> None:
        user_config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
            ),
            output=OutputConfig(
                artifact_directory=tmp_artifact_dir,
            ),
            otel_url="collector:4318",
            stream="timing",
        )
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config,
        )
        fake_queue = _setup_fanout_processor(processor)

        metric_record = create_metric_records_message(
            results=[{"request_latency_ns": 123_000_000}]
        ).to_data()
        await processor.process_result(metric_record)

        histogram_events = [
            e for e in fake_queue.events if e.get("type") == "histogram_record"
        ]
        assert histogram_events == []

    @pytest.mark.asyncio
    async def test_process_result_skips_timing_when_timing_telemetry_disabled(
        self,
        service_config: ServiceConfig,
        tmp_artifact_dir,
    ) -> None:
        user_config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
            ),
            output=OutputConfig(
                artifact_directory=tmp_artifact_dir,
            ),
            otel_url="collector:4318",
            stream="metrics",
        )
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config,
        )
        fake_queue = _setup_fanout_processor(processor)

        timing_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            start_ns=1_000_000_000,
            requests_end_ns=2_000_000_000,
            requests_sent=1,
            requests_completed=1,
            requests_cancelled=0,
            request_errors=0,
            sent_sessions=1,
            completed_sessions=1,
            cancelled_sessions=0,
            total_session_turns=1,
        )
        await processor.process_result(timing_stats)

        counter_events = [
            e for e in fake_queue.events if e.get("type") == "counter_add"
        ]
        up_down_events = [
            e for e in fake_queue.events if e.get("type") == "up_down_counter_add"
        ]
        assert counter_events == []
        assert up_down_events == []

    @pytest.mark.asyncio
    async def test_process_result_records_timing_counters_and_gauge_like_metrics(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        fake_queue = _setup_fanout_processor(processor)

        timing_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            start_ns=1_000_000_000,
            requests_end_ns=6_000_000_000,
            requests_sent=10,
            requests_completed=8,
            requests_cancelled=1,
            request_errors=2,
            sent_sessions=4,
            completed_sessions=2,
            cancelled_sessions=1,
            total_session_turns=9,
            timeout_triggered=False,
            grace_period_timeout_triggered=False,
            was_cancelled=False,
        )
        await processor.process_result(timing_stats)

        counter_events = [
            e for e in fake_queue.events if e.get("type") == "counter_add"
        ]
        up_down_events = [
            e for e in fake_queue.events if e.get("type") == "up_down_counter_add"
        ]

        counter_by_name = {}
        for e in counter_events:
            name = e["payload"]["metric_name"]  # type: ignore[index]
            counter_by_name[name] = e["payload"]["value"]  # type: ignore[index]

        assert counter_by_name["aiperf.timing.requests.sent"] == 10
        assert counter_by_name["aiperf.timing.requests.completed"] == 8
        assert counter_by_name["aiperf.timing.requests.cancelled"] == 1
        assert counter_by_name["aiperf.timing.requests.errors"] == 2
        assert counter_by_name["aiperf.timing.sessions.sent"] == 4
        assert counter_by_name["aiperf.timing.sessions.completed"] == 2
        assert counter_by_name["aiperf.timing.sessions.cancelled"] == 1
        assert counter_by_name["aiperf.timing.sessions.turns_total"] == 9

        up_down_by_name = {}
        for e in up_down_events:
            name = e["payload"]["metric_name"]  # type: ignore[index]
            up_down_by_name[name] = e["payload"]["value"]  # type: ignore[index]

        assert up_down_by_name["aiperf.timing.requests.in_flight"] == 1.0
        assert up_down_by_name["aiperf.timing.sessions.in_flight"] == 1.0
        assert up_down_by_name["aiperf.timing.phase.elapsed_sec"] == 5.0
        # First false boolean snapshots emit zero delta and are skipped.
        assert "aiperf.timing.phase.timeout_triggered" not in up_down_by_name
        assert "aiperf.timing.phase.grace_timeout_triggered" not in up_down_by_name
        assert "aiperf.timing.phase.was_cancelled" not in up_down_by_name

    @pytest.mark.asyncio
    async def test_process_result_timing_uses_delta_values_for_cumulative_counters(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        fake_queue = _setup_fanout_processor(processor)

        first_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            start_ns=1_000_000_000,
            requests_end_ns=2_000_000_000,
            requests_sent=10,
            requests_completed=8,
            requests_cancelled=1,
            request_errors=1,
            sent_sessions=4,
            completed_sessions=3,
            cancelled_sessions=0,
            total_session_turns=10,
            timeout_triggered=False,
            grace_period_timeout_triggered=False,
            was_cancelled=False,
        )
        second_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            start_ns=1_000_000_000,
            requests_end_ns=3_000_000_000,
            requests_sent=15,
            requests_completed=12,
            requests_cancelled=1,
            request_errors=2,
            sent_sessions=6,
            completed_sessions=4,
            cancelled_sessions=1,
            total_session_turns=16,
            timeout_triggered=True,
            grace_period_timeout_triggered=False,
            was_cancelled=False,
        )
        await processor.process_result(first_stats)
        await processor.process_result(second_stats)

        counter_events = [
            e for e in fake_queue.events if e.get("type") == "counter_add"
        ]

        # Collect all counter adds by metric name
        counter_adds: dict[str, list[float]] = {}
        for e in counter_events:
            name = e["payload"]["metric_name"]  # type: ignore[index]
            counter_adds.setdefault(name, []).append(e["payload"]["value"])  # type: ignore[index]

        # Second snapshot deltas: 15-10=5 sent, 12-8=4 completed, 1-1=0 errors delta
        assert counter_adds["aiperf.timing.requests.sent"][-1] == 5
        assert counter_adds["aiperf.timing.requests.completed"][-1] == 4
        assert counter_adds["aiperf.timing.requests.errors"][-1] == 1
        assert counter_adds["aiperf.timing.sessions.turns_total"][-1] == 6

        # No new cancellations in second snapshot (still 1), so only first emission
        assert len(counter_adds["aiperf.timing.requests.cancelled"]) == 1

        up_down_events = [
            e for e in fake_queue.events if e.get("type") == "up_down_counter_add"
        ]
        up_down_adds: dict[str, list[float]] = {}
        for e in up_down_events:
            name = e["payload"]["metric_name"]  # type: ignore[index]
            up_down_adds.setdefault(name, []).append(e["payload"]["value"])  # type: ignore[index]

        # In-flight requests: first=1, second=2, so two emissions
        assert len(up_down_adds["aiperf.timing.requests.in_flight"]) == 2
        # timeout_triggered went from False(0) to True(1), delta=1.0
        assert up_down_adds["aiperf.timing.phase.timeout_triggered"][-1] == 1.0

    @pytest.mark.asyncio
    async def test_flush_emits_flush_event_to_fanout_queue(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        fake_queue = _setup_fanout_processor(processor)

        await processor.flush(force=True)

        flush_events = [e for e in fake_queue.events if e.get("type") == "flush"]
        assert len(flush_events) == 1

    @pytest.mark.asyncio
    async def test_initialize_uses_fanout_by_default(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        processor._start_fanout_process = AsyncMock()

        await processor._initialize_meter_provider()

        processor._start_fanout_process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_uses_fanout_for_mlflow_only(
        self,
        service_config: ServiceConfig,
        user_config_mlflow_only: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_mlflow_only,
        )
        processor._start_fanout_process = AsyncMock()

        await processor._initialize_meter_provider()

        processor._start_fanout_process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_fanout_failure_disables_streaming(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        class FakeQueue:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class FakeProcess:
            def start(self) -> None:
                raise RuntimeError("fanout start failed")

        class FakeContext:
            def __init__(self) -> None:
                self.queue = FakeQueue()
                self.process = FakeProcess()

            def Queue(self, maxsize: int):  # noqa: N802
                return self.queue

            def Process(  # noqa: N802
                self, target: object, args: tuple[object, ...], name: str, daemon: bool
            ):
                return self.process

        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        fake_context = FakeContext()

        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.mp.get_context",
            return_value=fake_context,
        ):
            await processor._start_fanout_process()

        assert fake_context.queue.closed is True
        assert processor._fanout_queue is None
        assert processor._fanout_process is None
        assert processor._streaming_ready is False

    @pytest.mark.asyncio
    async def test_start_fanout_failure_disables_streaming_for_mlflow_only(
        self,
        service_config: ServiceConfig,
        user_config_mlflow_only: UserConfig,
    ) -> None:
        class FakeContext:
            def Queue(self, maxsize: int):  # noqa: N802
                raise RuntimeError("queue creation failed")

        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_mlflow_only,
        )

        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.mp.get_context",
            return_value=FakeContext(),
        ):
            await processor._start_fanout_process()

        assert processor._streaming_ready is False
        assert processor._fanout_queue is None
        assert processor._fanout_process is None

    @pytest.mark.asyncio
    async def test_process_result_fanout_emits_metric_and_timing_events(
        self,
        service_config: ServiceConfig,
        user_config_otel_mlflow: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel_mlflow,
        )
        fake_queue = _setup_fanout_processor(processor)

        metric_record = create_metric_records_message(
            results=[{"request_latency_ns": 123_000_000, "request_count": 1}]
        ).to_data()
        await processor.process_result(metric_record)

        timing_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            start_ns=1_000_000_000,
            requests_end_ns=3_000_000_000,
            requests_sent=10,
            requests_completed=8,
            requests_cancelled=1,
            request_errors=0,
            sent_sessions=4,
            completed_sessions=3,
            cancelled_sessions=0,
            total_session_turns=9,
        )
        await processor.process_result(timing_stats)

        event_types = [str(event.get("type")) for event in fake_queue.events]
        assert "histogram_record" in event_types
        assert "counter_add" in event_types
        assert "up_down_counter_add" in event_types
        # Verify at least one histogram has a metric related to request latency
        # (exact name depends on GenAI semconv translation)
        histogram_names = {
            event.get("payload", {}).get("metric_name")
            for event in fake_queue.events
            if event.get("type") == "histogram_record"
        }
        assert len(histogram_names) >= 1

    def test_queue_fanout_event_drops_oldest_when_queue_is_full(
        self,
        service_config: ServiceConfig,
        user_config_mlflow_only: UserConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class FullFakeQueue:
            def __init__(self, events: list[dict[str, object]], maxsize: int) -> None:
                self.events = list(events)
                self.maxsize = maxsize

            def put_nowait(self, event: dict[str, object]) -> None:
                if len(self.events) >= self.maxsize:
                    raise Full
                self.events.append(event)

            def get_nowait(self) -> dict[str, object]:
                return self.events.pop(0)

            def close(self) -> None:
                return

        oldest_event = {"type": "histogram_record", "payload": {"metric_name": "old"}}
        newest_queued_event = {
            "type": "histogram_record",
            "payload": {"metric_name": "newer"},
        }
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_mlflow_only,
        )
        processor._fanout_queue = FullFakeQueue(
            events=[oldest_event, newest_queued_event],
            maxsize=2,
        )

        with caplog.at_level("WARNING"):
            processor._queue_fanout_event("flush", {})

        assert processor._fanout_dropped_events == 1
        assert processor._fanout_sent_events == 1
        assert processor._fanout_queue.events == [
            newest_queued_event,
            {"type": "flush", "payload": {}},
        ]
        assert "dropping oldest event" in caplog.text

    @pytest.mark.asyncio
    async def test_flush_and_stop_emit_fanout_control_events(
        self,
        service_config: ServiceConfig,
        user_config_otel_mlflow: UserConfig,
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.join_calls: list[float] = []
                self.terminate_called = False

            def join(self, timeout: float) -> None:
                self.join_calls.append(timeout)

            def is_alive(self) -> bool:
                return False

            def terminate(self) -> None:
                self.terminate_called = True

        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel_mlflow,
        )
        fake_queue = _setup_fanout_processor(processor)
        processor._fanout_process = FakeProcess()

        await processor.flush(force=True)
        await processor._flush_and_shutdown()

        event_types = [str(event.get("type")) for event in fake_queue.events]
        assert "flush" in event_types
        assert "shutdown" in event_types
        assert fake_queue.closed is True

    @pytest.mark.asyncio
    async def test_on_stop_flushes_and_stops_fanout(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        fake_queue = _setup_fanout_processor(processor)
        processor._fanout_process = None

        await processor._flush_and_shutdown()

        event_types = [str(event.get("type")) for event in fake_queue.events]
        assert "flush" in event_types
        assert "shutdown" in event_types
        assert processor._streaming_ready is False

    def test_build_record_attributes(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        metric_record = create_metric_records_message(
            results=[{"request_latency_ns": 123_000_000}]
        ).to_data()

        attributes = processor.build_record_attributes(metric_record)
        assert attributes["aiperf.worker.id"] == metric_record.metadata.worker_id
        assert (
            attributes["aiperf.record_processor.id"]
            == metric_record.metadata.record_processor_id
        )
        assert attributes["aiperf.benchmark_phase"] == str(
            metric_record.metadata.benchmark_phase
        )
        assert attributes["aiperf.has_error"] is False
        # Verify high-cardinality attributes are NOT included
        assert "aiperf.session_num" not in attributes
        assert "aiperf.turn_index" not in attributes

    def test_coerce_metric_values_handling(
        self,
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        assert processor.coerce_metric_values("test", 123) == [123.0]
        assert processor.coerce_metric_values("test", 123.5) == [123.5]
        assert processor.coerce_metric_values("test", [1, 2.5, "invalid", True]) == [
            1.0,
            2.5,
        ]
        assert processor.coerce_metric_values("test", True) == []
        assert processor.coerce_metric_values("test", {"key": "value"}) == []
        assert processor.coerce_metric_values("test", None) == []

    def test_build_resource_attributes_populates_model_name(
        self, service_config: ServiceConfig, user_config_otel: UserConfig
    ) -> None:
        """Happy path: model_names[0] populates aiperf.model.name."""
        processor = OTelMetricsResultsProcessor(
            service_id="test-service",
            user_config=user_config_otel,
            service_config=service_config,
        )
        attrs = processor._build_resource_attributes()
        assert attrs["aiperf.model.name"] == "test-model"

    def test_build_resource_attributes_empty_model_names_does_not_raise(
        self, service_config: ServiceConfig, user_config_otel: UserConfig
    ) -> None:
        """Regression: empty model_names must skip aiperf.model.name instead of
        raising IndexError. EndpointConfig.model_names has no min_length=1, so
        a programmatic caller can construct an empty list — the OTel resource
        attributes builder must not crash the fanout in that case.
        """
        processor = OTelMetricsResultsProcessor(
            service_id="test-service",
            user_config=user_config_otel,
            service_config=service_config,
        )
        # Mutate after construction to bypass any Field validator.
        user_config_otel.endpoint.model_names = []
        attrs = processor._build_resource_attributes()
        assert "aiperf.model.name" not in attrs
        # Other required resource attrs should still be present.
        assert attrs["service.name"] == "aiperf"
        assert attrs["aiperf.endpoint.type"] == str(user_config_otel.endpoint.type)
