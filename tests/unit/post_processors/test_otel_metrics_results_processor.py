# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
from queue import Full
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.models import CreditPhaseStats
from aiperf.common.optional_dependencies import OTEL_METRICS_STREAMING_FEATURE
from aiperf.config import (
    BenchmarkConfig,
)
from aiperf.config.flags import CLIConfig
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.otel_metrics_results_processor import (
    OTelMetricsResultsProcessor,
)
from tests.unit.conftest import make_run_from_cli
from tests.unit.post_processors.conftest import create_metric_records_message


@pytest.fixture
def cfg_otel(tmp_artifact_dir):
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            artifact_directory=tmp_artifact_dir,
        )
    )
    run.cfg.otel.metrics_url = "collector:4318"
    run.cfg.mlflow.tracking_uri = "http://mlflow:5000"
    run.cfg.mlflow.experiment = "aiperf-tests"
    return run


@pytest.fixture
def cfg_otel_mlflow(tmp_artifact_dir):
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            artifact_directory=tmp_artifact_dir,
        )
    )
    run.cfg.otel.metrics_url = "collector:4318"
    run.cfg.mlflow.tracking_uri = "http://mlflow:5000"
    run.cfg.mlflow.experiment = "aiperf-tests"
    return run


@pytest.fixture
def cfg_mlflow_only(tmp_artifact_dir):
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            artifact_directory=tmp_artifact_dir,
        )
    )
    run.cfg.mlflow.tracking_uri = "http://mlflow:5000"
    run.cfg.mlflow.experiment = "aiperf-tests"
    return run


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
    def test_disabled_without_otel_or_mlflow(self) -> None:
        cfg = make_run_from_cli(
            CLIConfig(model_names=["test-model"], endpoint_type=EndpointType.CHAT)
        )
        with pytest.raises(PostProcessorDisabled):
            OTelMetricsResultsProcessor(
                service_id="records-manager",
                run=cfg,
            )

    def test_enabled_with_mlflow_without_otel_url(
        self,
        cfg_mlflow_only: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_mlflow_only,
        )
        assert processor._otel_metrics_url is None
        assert processor._mlflow_live_enabled is True

    def test_mlflow_only_does_not_require_otel_imports(
        self,
        cfg_mlflow_only: BenchmarkConfig,
    ) -> None:
        with patch("builtins.__import__", side_effect=_import_side_effect_for_otel):
            processor = OTelMetricsResultsProcessor(
                service_id="records-manager",
                run=cfg_mlflow_only,
            )
        assert processor._mlflow_live_enabled is True

    def test_init_dependency_failure_raises_post_processor_disabled(
        self,
        tmp_artifact_dir,
    ) -> None:
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                endpoint_type=EndpointType.CHAT,
                artifact_directory=tmp_artifact_dir,
            )
        )
        run.cfg.otel.metrics_url = "collector:4318"
        with (
            patch("builtins.__import__", side_effect=_import_side_effect_for_otel),
            pytest.raises(PostProcessorDisabled) as exc_info,
        ):
            OTelMetricsResultsProcessor(
                service_id="records-manager",
                run=run,
            )
        assert OTEL_METRICS_STREAMING_FEATURE in str(exc_info.value)

    def test_init_otel_import_failure_falls_back_to_mlflow_only(
        self,
        cfg_otel_mlflow: BenchmarkConfig,
    ) -> None:
        """When both sinks are configured but OTel imports fail, MLflow live
        streaming must still be constructed. Regression: the parent-side OTel
        import check previously disabled the entire fanout processor, dropping
        MLflow live metrics even though MLflow was independently usable.
        """
        with patch("builtins.__import__", side_effect=_import_side_effect_for_otel):
            processor = OTelMetricsResultsProcessor(
                service_id="records-manager",
                run=cfg_otel_mlflow,
            )
        assert processor._otel_metrics_url is None
        assert processor._mlflow_live_enabled is True

    @pytest.mark.asyncio
    async def test_process_result_records_histogram_values_by_metric(
        self,
        cfg_otel: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel,
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
        tmp_artifact_dir,
    ) -> None:
        cfg = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                endpoint_type=EndpointType.CHAT,
                artifact_directory=tmp_artifact_dir,
            )
        )
        cfg.cfg.otel.metrics_url = "collector:4318"
        cfg.cfg.mlflow.tracking_uri = "http://mlflow:5000"
        cfg.cfg.mlflow.experiment = "aiperf-tests"
        cfg.cfg.otel.stream_metrics_enabled = False
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg,
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
        tmp_artifact_dir,
    ) -> None:
        cfg = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                endpoint_type=EndpointType.CHAT,
                artifact_directory=tmp_artifact_dir,
            )
        )
        cfg.cfg.otel.metrics_url = "collector:4318"
        cfg.cfg.mlflow.tracking_uri = "http://mlflow:5000"
        cfg.cfg.mlflow.experiment = "aiperf-tests"
        cfg.cfg.otel.stream_timing_enabled = False
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg,
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
        cfg_otel: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel,
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
        cfg_otel: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel,
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
        cfg_otel: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel,
        )
        fake_queue = _setup_fanout_processor(processor)

        await processor.flush(force=True)

        flush_events = [e for e in fake_queue.events if e.get("type") == "flush"]
        assert len(flush_events) == 1

    @pytest.mark.asyncio
    async def test_initialize_uses_fanout_by_default(
        self,
        cfg_otel: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel,
        )
        processor._start_fanout_process = AsyncMock()

        await processor._initialize_meter_provider()

        processor._start_fanout_process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_uses_fanout_for_mlflow_only(
        self,
        cfg_mlflow_only: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_mlflow_only,
        )
        processor._start_fanout_process = AsyncMock()

        await processor._initialize_meter_provider()

        processor._start_fanout_process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_fanout_failure_disables_streaming(
        self,
        cfg_otel: BenchmarkConfig,
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
            run=cfg_otel,
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
        cfg_mlflow_only: BenchmarkConfig,
    ) -> None:
        class FakeContext:
            def Queue(self, maxsize: int):  # noqa: N802
                raise RuntimeError("queue creation failed")

        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_mlflow_only,
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
        cfg_otel_mlflow: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel_mlflow,
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
        cfg_mlflow_only: BenchmarkConfig,
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
            run=cfg_mlflow_only,
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
        cfg_otel_mlflow: BenchmarkConfig,
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
            run=cfg_otel_mlflow,
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
        cfg_otel: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel,
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
        cfg_otel: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel,
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
        cfg_otel: BenchmarkConfig,
    ) -> None:
        processor = OTelMetricsResultsProcessor(
            service_id="records-manager",
            run=cfg_otel,
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

    def test_build_resource_attributes_populates_model_name(self, cfg_otel) -> None:
        """Happy path: model_names[0] populates aiperf.model.name."""
        processor = OTelMetricsResultsProcessor(
            service_id="test-service",
            run=cfg_otel,
        )
        attrs = processor._build_resource_attributes()
        assert attrs["aiperf.model.name"] == "test-model"

    def test_build_resource_attributes_empty_model_names_does_not_raise(
        self, cfg_otel
    ) -> None:
        """Regression: empty model_names must skip aiperf.model.name instead of
        raising IndexError. EndpointConfig.model_names has no min_length=1, so
        a programmatic caller can construct an empty list — the OTel resource
        attributes builder must not crash the fanout in that case.
        """
        processor = OTelMetricsResultsProcessor(
            service_id="test-service",
            run=cfg_otel,
        )
        # Mutate after construction to bypass any Field validator.
        cfg_otel.cfg.models.items = []
        attrs = processor._build_resource_attributes()
        assert "aiperf.model.name" not in attrs
        # Other required resource attrs should still be present.
        assert attrs["service.name"] == "aiperf"
        assert attrs["aiperf.endpoint.type"] == str(cfg_otel.cfg.endpoint.type)
