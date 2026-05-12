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
from aiperf.common.models import CreditPhaseStats, MetricResult
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
        assert processor._use_fanout_process is True

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

    @pytest.mark.asyncio
    async def test_initialize_otel_imports_failure_raises_post_processor_disabled(
        self,
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        processor._use_fanout_process = False
        with (
            patch("builtins.__import__", side_effect=_import_side_effect_for_otel),
            pytest.raises(PostProcessorDisabled) as exc_info,
        ):
            await processor._initialize_meter_provider()
        assert OTEL_METRICS_STREAMING_FEATURE in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_result_records_histogram_values_by_metric(
        self,
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        processor._use_fanout_process = False
        await processor._initialize_meter_provider()

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

        provider_cls = fake_otel["MeterProvider"]
        provider = provider_cls.instances[-1]
        histograms = provider.meter.histograms
        assert set(histograms) == {
            "aiperf.request_latency_ns",
            "aiperf.request_count",
            "aiperf.tokens_per_response",
        }
        assert [
            value for value, _ in histograms["aiperf.request_latency_ns"].records
        ] == [123_000_000.0]
        assert [value for value, _ in histograms["aiperf.request_count"].records] == [
            1.0
        ]
        assert [
            value for value, _ in histograms["aiperf.tokens_per_response"].records
        ] == [
            1.0,
            2.0,
            3.0,
        ]

    @pytest.mark.asyncio
    async def test_process_result_skips_metrics_when_metrics_telemetry_disabled(
        self,
        fake_otel: dict[str, object],
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
        processor._use_fanout_process = False
        await processor._initialize_meter_provider()

        metric_record = create_metric_records_message(
            results=[{"request_latency_ns": 123_000_000}]
        ).to_data()
        await processor.process_result(metric_record)

        provider_cls = fake_otel["MeterProvider"]
        provider = provider_cls.instances[-1]
        assert provider.meter.histograms == {}

    @pytest.mark.asyncio
    async def test_process_result_skips_timing_when_timing_telemetry_disabled(
        self,
        fake_otel: dict[str, object],
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
        processor._use_fanout_process = False
        await processor._initialize_meter_provider()

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

        provider_cls = fake_otel["MeterProvider"]
        provider = provider_cls.instances[-1]
        assert provider.meter.counters == {}
        assert provider.meter.up_down_counters == {}

    @pytest.mark.asyncio
    async def test_process_result_records_timing_counters_and_gauge_like_metrics(
        self,
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        processor._use_fanout_process = False
        await processor._initialize_meter_provider()

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

        provider_cls = fake_otel["MeterProvider"]
        provider = provider_cls.instances[-1]
        counters = provider.meter.counters
        up_down_counters = provider.meter.up_down_counters

        assert counters["aiperf.timing.requests.sent"].adds[-1][0] == 10
        assert counters["aiperf.timing.requests.completed"].adds[-1][0] == 8
        assert counters["aiperf.timing.requests.cancelled"].adds[-1][0] == 1
        assert counters["aiperf.timing.requests.errors"].adds[-1][0] == 2
        assert counters["aiperf.timing.sessions.sent"].adds[-1][0] == 4
        assert counters["aiperf.timing.sessions.completed"].adds[-1][0] == 2
        assert counters["aiperf.timing.sessions.cancelled"].adds[-1][0] == 1
        assert counters["aiperf.timing.sessions.turns_total"].adds[-1][0] == 9

        assert up_down_counters["aiperf.timing.requests.in_flight"].adds[-1][0] == 1.0
        assert up_down_counters["aiperf.timing.sessions.in_flight"].adds[-1][0] == 1.0
        # First false boolean snapshots emit zero delta and are intentionally skipped.
        assert "aiperf.timing.phase.timeout_triggered" not in up_down_counters
        assert "aiperf.timing.phase.grace_timeout_triggered" not in up_down_counters
        assert "aiperf.timing.phase.was_cancelled" not in up_down_counters
        assert up_down_counters["aiperf.timing.phase.elapsed_sec"].adds[-1][0] == 5.0

    @pytest.mark.asyncio
    async def test_process_result_timing_uses_delta_values_for_cumulative_counters(
        self,
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        processor._use_fanout_process = False
        await processor._initialize_meter_provider()

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

        provider_cls = fake_otel["MeterProvider"]
        provider = provider_cls.instances[-1]
        counters = provider.meter.counters
        up_down_counters = provider.meter.up_down_counters

        assert counters["aiperf.timing.requests.sent"].adds[-1][0] == 5
        assert counters["aiperf.timing.requests.completed"].adds[-1][0] == 4
        assert counters["aiperf.timing.requests.errors"].adds[-1][0] == 1
        assert counters["aiperf.timing.sessions.turns_total"].adds[-1][0] == 6

        # No new cancellations in second snapshot, so the delta counter should not emit.
        assert len(counters["aiperf.timing.requests.cancelled"].adds) == 1

        # In-flight requests moved from 1 to 2, so one additional delta update is emitted.
        assert len(up_down_counters["aiperf.timing.requests.in_flight"].adds) == 2
        assert (
            up_down_counters["aiperf.timing.phase.timeout_triggered"].adds[-1][0] == 1.0
        )

    @pytest.mark.asyncio
    async def test_flush_calls_meter_provider_force_flush(
        self,
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        processor._use_fanout_process = False
        await processor._initialize_meter_provider()

        provider_cls = fake_otel["MeterProvider"]
        provider = provider_cls.instances[-1]
        await processor.flush(force=True)
        assert len(provider.force_flush_calls) == 1

    @pytest.mark.asyncio
    async def test_initialize_uses_fanout_by_default(
        self,
        fake_otel: dict[str, object],
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
        fake_otel: dict[str, object],
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
    async def test_start_fanout_failure_falls_back_to_in_process_for_otel(
        self,
        fake_otel: dict[str, object],
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
        processor._initialize_in_process_meter_provider = AsyncMock()

        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.mp.get_context",
            return_value=fake_context,
        ):
            await processor._start_fanout_process()

        processor._initialize_in_process_meter_provider.assert_awaited_once()
        assert processor._use_fanout_process is False
        assert fake_context.queue.closed is True
        assert processor._fanout_queue is None
        assert processor._fanout_process is None

    @pytest.mark.asyncio
    async def test_start_fanout_failure_disables_streaming_for_mlflow_only(
        self,
        fake_otel: dict[str, object],
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
        processor._initialize_in_process_meter_provider = AsyncMock()

        with patch(
            "aiperf.post_processors.otel_metrics_results_processor.mp.get_context",
            return_value=FakeContext(),
        ):
            await processor._start_fanout_process()

        processor._initialize_in_process_meter_provider.assert_not_awaited()
        assert processor._use_fanout_process is True
        assert processor._streaming_ready is False
        assert processor._fanout_queue is None
        assert processor._fanout_process is None

    @pytest.mark.asyncio
    async def test_process_result_fanout_emits_metric_and_timing_events(
        self,
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel_mlflow: UserConfig,
    ) -> None:
        class FakeQueue:
            def __init__(self) -> None:
                self.events: list[dict[str, object]] = []

            def put_nowait(self, event: dict[str, object]) -> None:
                self.events.append(event)

            def close(self) -> None:
                return

        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel_mlflow,
        )
        fake_queue = FakeQueue()
        processor._use_fanout_process = True
        processor._streaming_ready = True
        processor._fanout_queue = fake_queue

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
        assert any(
            event.get("payload", {}).get("metric_name") == "aiperf.request_latency_ns"
            for event in fake_queue.events
        )

    def test_queue_fanout_event_drops_oldest_when_queue_is_full(
        self,
        service_config: ServiceConfig,
        user_config_mlflow_only: UserConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class FakeQueue:
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
        processor._fanout_queue = FakeQueue(
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
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel_mlflow: UserConfig,
    ) -> None:
        class FakeQueue:
            def __init__(self) -> None:
                self.events: list[dict[str, object]] = []
                self.closed = False

            def put_nowait(self, event: dict[str, object]) -> None:
                self.events.append(event)

            def close(self) -> None:
                self.closed = True

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
        fake_queue = FakeQueue()
        processor._use_fanout_process = True
        processor._streaming_ready = True
        processor._fanout_queue = fake_queue
        processor._fanout_process = FakeProcess()

        await processor.flush(force=True)
        await processor._flush_and_shutdown()

        event_types = [str(event.get("type")) for event in fake_queue.events]
        assert "flush" in event_types
        assert "shutdown" in event_types
        assert fake_queue.closed is True

    @pytest.mark.asyncio
    async def test_on_stop_flushes_and_shuts_down_provider(
        self,
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel: UserConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel,
        )
        processor._use_fanout_process = False
        await processor._initialize_meter_provider()
        provider_cls = fake_otel["MeterProvider"]
        provider = provider_cls.instances[-1]

        processor.flush = AsyncMock()
        await processor._flush_and_shutdown()

        processor.flush.assert_awaited_once_with(force=True)
        assert provider.shutdown_calls == 1
        assert processor._meter_provider is None
        assert processor._meter is None

    def test_build_record_attributes(
        self,
        fake_otel: dict[str, object],
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
        assert "aiperf.session_num" not in attributes
        assert "aiperf.turn_index" not in attributes

    @pytest.mark.asyncio
    async def test_process_realtime_metrics_emits_mlflow_snapshot_events(
        self,
        fake_otel: dict[str, object],
        service_config: ServiceConfig,
        user_config_otel_mlflow: UserConfig,
    ) -> None:
        class FakeQueue:
            def __init__(self) -> None:
                self.events: list[dict[str, object]] = []

            def put_nowait(self, event: dict[str, object]) -> None:
                self.events.append(event)

        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_otel_mlflow,
        )
        fake_queue = FakeQueue()
        processor._use_fanout_process = True
        processor._streaming_ready = True
        processor._fanout_queue = fake_queue

        await processor.process_realtime_metrics(
            [
                MetricResult(
                    tag="request_count",
                    header="Request Count",
                    unit="requests",
                    avg=60,
                ),
                MetricResult(
                    tag="empty_metric",
                    header="Empty",
                    unit="1",
                    avg=None,
                ),
            ]
        )

        assert fake_queue.events == [
            {
                "type": "mlflow_metric_snapshot",
                "payload": {
                    "metric_name": "aiperf.request_count",
                    "value": 60.0,
                },
            }
        ]

    def test_coerce_metric_values_handling(
        self,
        fake_otel: dict[str, object],
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
        assert processor.coerce_metric_values("request_latency", 200_000_000) == [200.0]
        assert processor._metric_unit("request_latency") == "ms"
