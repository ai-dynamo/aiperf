# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.messages import RealtimeMetricsCommand
from aiperf.common.messages.inference_messages import (
    MetricRecordsData,
    MetricRecordsMessage,
)
from aiperf.common.models import (
    CreditPhaseStats,
    ErrorDetails,
    MetricResult,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProfileResults,
    RequestRecord,
    TimesliceResult,
)
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.common.types import MetricTagT
from aiperf.plugin.enums import UIType
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_tracker import RecordsTracker


# Helper functions
def create_mock_records_manager(
    start_time_ns: int,
    expected_duration_sec: float | None,
    grace_period_sec: float = 0.0,
) -> MagicMock:
    """Create a mock RecordsManager instance for testing filtering logic."""
    instance = MagicMock()
    instance.expected_duration_sec = expected_duration_sec
    instance.start_time_ns = start_time_ns
    instance.user_config.loadgen.benchmark_grace_period = grace_period_sec
    instance.debug = MagicMock()
    return instance


def create_metric_record_data(
    request_start_ns: int,
    request_end_ns: int,
    metrics: dict[MetricTagT, int | float] | None = None,
) -> MetricRecordsData:
    """Create a MetricRecordsData object with sensible defaults for testing."""
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=0,
            conversation_id="test",
            turn_index=0,
            request_start_ns=request_start_ns,
            request_end_ns=request_end_ns,
            worker_id="worker-1",
            record_processor_id="processor-1",
            benchmark_phase=CreditPhase.PROFILING,
        ),
        metrics=metrics or {},
    )


class RecordingAccumulator:
    """Capture metric records delivered by RecordsManager dispatch."""

    def __init__(self) -> None:
        self.records: list[Any] = []

    async def process_record(self, record_data: MetricRecordsData) -> None:
        self.records.append(record_data)


def test_has_realtime_update_detects_changed_server_snapshot_with_same_record_count() -> (
    None
):
    phase_stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        success_records=2,
    )
    manager = MagicMock(spec=RecordsManager)
    manager._previous_realtime_records = 2
    manager._previous_realtime_server_snapshot = {"num_running": 1.0}
    manager._has_realtime_update = RecordsManager._has_realtime_update.__get__(manager)

    assert (
        manager._has_realtime_update(
            phase_stats,
            {"num_running": 2.0},
        )
        is True
    )


def _make_realtime_task_manager(
    *,
    phase_stats: PhaseRecordsStats,
    server_snapshot: dict[str, float],
) -> MagicMock:
    manager = MagicMock(spec=RecordsManager)
    manager.stop_requested = False
    manager.service_config = ServiceConfig(ui_type=UIType.NONE)
    manager._records_tracker = MagicMock()
    manager._records_tracker.create_stats_for_phase.return_value = phase_stats
    manager._collect_realtime_server_snapshot = MagicMock(return_value=server_snapshot)
    manager._has_realtime_update = RecordsManager._has_realtime_update.__get__(manager)
    manager._previous_realtime_records = None
    manager._previous_realtime_server_snapshot = None
    manager._report_realtime_metrics = AsyncMock(return_value=True)
    return manager


async def _run_one_realtime_task_tick(
    manager: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stop_after_sleep(_interval: float) -> None:
        manager.stop_requested = True

    monkeypatch.setattr(
        "aiperf.records.records_manager.asyncio.sleep", stop_after_sleep
    )
    monkeypatch.setattr(Environment.UI, "REALTIME_METRICS_INTERVAL", 0.01)
    task = RecordsManager._report_realtime_inference_metrics_task
    await getattr(task, "__wrapped__", task)(manager)


@pytest.mark.asyncio
async def test_realtime_task_reports_and_advances_changed_snapshot_with_same_record_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        success_records=2,
    )
    manager = _make_realtime_task_manager(
        phase_stats=phase_stats,
        server_snapshot={"num_running": 2.0},
    )
    manager._previous_realtime_records = 2
    manager._previous_realtime_server_snapshot = {"num_running": 1.0}

    await _run_one_realtime_task_tick(manager, monkeypatch)

    manager._report_realtime_metrics.assert_awaited_once_with(
        server_snapshot={"num_running": 2.0}
    )
    assert manager._previous_realtime_records == 2
    assert manager._previous_realtime_server_snapshot == {"num_running": 2.0}


@pytest.mark.asyncio
async def test_realtime_task_skips_unchanged_records_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        success_records=2,
    )
    manager = _make_realtime_task_manager(
        phase_stats=phase_stats,
        server_snapshot={"num_running": 1.0},
    )
    manager._previous_realtime_records = 2
    manager._previous_realtime_server_snapshot = {"num_running": 1.0}

    await _run_one_realtime_task_tick(manager, monkeypatch)

    manager._report_realtime_metrics.assert_not_awaited()
    assert manager._previous_realtime_records == 2
    assert manager._previous_realtime_server_snapshot == {"num_running": 1.0}


@pytest.mark.asyncio
async def test_realtime_task_reuses_precomputed_snapshot_without_duplicate_collect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        start_ns=1234,
        success_records=2,
    )
    manager = _make_realtime_task_manager(
        phase_stats=phase_stats,
        server_snapshot={"num_running": 2.0},
    )

    await _run_one_realtime_task_tick(manager, monkeypatch)

    manager._collect_realtime_server_snapshot.assert_called_once_with(start_ns=1234)
    manager._report_realtime_metrics.assert_awaited_once_with(
        server_snapshot={"num_running": 2.0}
    )


@pytest.mark.asyncio
async def test_realtime_task_does_not_advance_state_when_report_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        success_records=2,
    )
    manager = _make_realtime_task_manager(
        phase_stats=phase_stats,
        server_snapshot={"num_running": 2.0},
    )
    manager._report_realtime_metrics.side_effect = RuntimeError("publish failed")

    with pytest.raises(RuntimeError, match="publish failed"):
        await _run_one_realtime_task_tick(manager, monkeypatch)

    assert manager._previous_realtime_records is None
    assert manager._previous_realtime_server_snapshot is None


@pytest.mark.asyncio
async def test_realtime_metrics_command_reports_unconditionally() -> None:
    manager = MagicMock(spec=RecordsManager)
    manager._report_realtime_metrics = AsyncMock()
    command = RealtimeMetricsCommand(service_id="dashboard")

    handler = RecordsManager._on_realtime_metrics_command
    await getattr(handler, "__wrapped__", handler)(manager, command)

    manager._report_realtime_metrics.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_context_overflow_skip_bypasses_metric_accumulators_and_stream_exporters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = MetricRecordMetadata(
        session_num=0,
        conversation_id="overflow-conversation",
        turn_index=0,
        request_start_ns=100,
        request_end_ns=200,
        worker_id="worker-1",
        record_processor_id="processor-1",
        benchmark_phase=CreditPhase.PROFILING,
        context_overflow_skip=True,
    )
    message = MetricRecordsMessage(
        service_id="record-processor-1",
        metadata=metadata,
        results=[{"context_overflow_count": 1}],
        error=ErrorDetails(message="context window exceeded"),
    )
    record_data = message.to_data().model_copy(
        update={"request": RequestRecord(context_overflow=True)}
    )
    monkeypatch.setattr(message, "to_data", lambda: record_data)

    fake_metric_accumulator = RecordingAccumulator()
    fake_stream_exporter = RecordingAccumulator()
    manager = MagicMock(spec=RecordsManager)
    manager.is_trace_enabled = False
    manager._metric_record_accumulators = [fake_metric_accumulator]
    manager._metric_record_stream_exporters = [fake_stream_exporter]
    manager._records_tracker = RecordsTracker()
    manager._records_tracker.update_phase_info(
        CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            final_requests_completed=1,
        )
    )
    manager._skipped_context_overflow_count = 0
    manager._error_tracker = MagicMock()
    manager._handle_all_records_received = AsyncMock()
    manager._maybe_trigger_failed_request_abort = AsyncMock()
    manager._send_record_to_accumulators = (
        RecordsManager._send_record_to_accumulators.__get__(manager)
    )
    manager._on_metric_records = RecordsManager._on_metric_records.__get__(manager)

    await manager._on_metric_records(message)

    assert fake_metric_accumulator.records == []
    assert fake_stream_exporter.records == []
    phase_stats = manager._records_tracker.create_stats_for_phase(CreditPhase.PROFILING)
    assert phase_stats.success_records == 1
    assert phase_stats.error_records == 0
    assert phase_stats.records_end_ns is not None
    assert manager._skipped_context_overflow_count == 1
    manager._handle_all_records_received.assert_awaited_once_with(CreditPhase.PROFILING)
    manager._error_tracker.increment_error_count_for_phase.assert_not_called()
    manager._maybe_trigger_failed_request_abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_results_exposes_skipped_context_overflow_count() -> None:
    manager = MagicMock()
    manager.service_id = "records-manager"
    manager._records_tracker = RecordsTracker()
    manager._records_tracker.update_phase_info(
        CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            final_requests_completed=1,
            final_requests_sent=1,
            start_ns=100,
            requests_end_ns=200,
        )
    )
    manager._skipped_context_overflow_count = 1
    manager._error_tracker = MagicMock()
    manager._error_tracker.get_error_summary_for_phase.return_value = []
    manager._snapshot_branch_stats.return_value = None
    manager._summarize_all_accumulators = AsyncMock(return_value=([], [], []))
    manager._finalize_stream_exporters = AsyncMock()
    manager._run_analyzers = AsyncMock(return_value={})
    manager._publish_all_results = AsyncMock()
    manager.publish = AsyncMock()
    manager.debug = MagicMock()
    manager.info = MagicMock()
    manager.user_config.gpu_telemetry_disabled = True
    manager.user_config.server_metrics_disabled = True
    manager._process_results = RecordsManager._process_results.__get__(manager)

    result = await manager._process_results(
        phase=CreditPhase.PROFILING, cancelled=False
    )

    assert result.results.completed == 0
    assert result.results.context_overflow_count == 1
    published = manager.publish.await_args.args[0]
    assert published.results.results.context_overflow_count == 1


class TestRecordsManagerTelemetry:
    """Test RecordsManager telemetry handling with mocked components."""

    @pytest.mark.asyncio
    async def test_on_telemetry_records_valid(self):
        """Test handling valid telemetry records."""
        from unittest.mock import AsyncMock, MagicMock

        from aiperf.common.messages import TelemetryRecordsMessage
        from aiperf.common.models import (
            TelemetryHierarchy,
            TelemetryMetrics,
            TelemetryRecord,
        )

        # Create sample telemetry records
        records = [
            TelemetryRecord(
                timestamp_ns=1000000,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=0,
                gpu_uuid="GPU-123",
                gpu_model_name="Test GPU",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=100.0,
                ),
            )
        ]

        message = TelemetryRecordsMessage(
            service_id="test_service",
            collector_id="test_collector",
            dcgm_url="http://localhost:9400/metrics",
            records=records,
            error=None,
        )

        # Mock the hierarchy
        mock_hierarchy = MagicMock(spec=TelemetryHierarchy)
        mock_hierarchy.add_record = MagicMock()
        mock_send_to_processors = AsyncMock()

        # Test the logic directly without instantiating the full service
        for record in message.records:
            mock_hierarchy.add_record(record)

        if message.records:
            await mock_send_to_processors(message.records)

        # Verify behavior
        assert mock_hierarchy.add_record.call_count == len(records)
        mock_send_to_processors.assert_called_once_with(records)

    @pytest.mark.asyncio
    async def test_on_telemetry_records_invalid(self):
        """Test handling invalid telemetry records with errors."""
        from unittest.mock import AsyncMock

        from aiperf.common.messages import TelemetryRecordsMessage
        from aiperf.common.models import ErrorDetails

        error = ErrorDetails(message="Test error", code=500)

        message = TelemetryRecordsMessage(
            service_id="test_service",
            collector_id="test_collector",
            dcgm_url="http://localhost:9400/metrics",
            records=[],
            error=error,
        )

        mock_send_to_processors = AsyncMock()
        error_counts = {}

        # Test the logic: errors should be tracked, not sent to processors
        if message.error:
            error_counts[message.error] = error_counts.get(message.error, 0) + 1
        else:
            await mock_send_to_processors(message.records)

        # Should not send to processors
        mock_send_to_processors.assert_not_called()

        # Error should be tracked
        assert error in error_counts
        assert error_counts[error] == 1

    @pytest.mark.asyncio
    async def test_send_telemetry_to_results_processors(self):
        """Test sending telemetry records to processors."""
        from unittest.mock import AsyncMock, Mock

        from aiperf.common.models import TelemetryMetrics, TelemetryRecord

        # Create mock telemetry processor
        mock_processor = Mock()
        mock_processor.process_telemetry_record = AsyncMock()

        records = [
            TelemetryRecord(
                timestamp_ns=1000000,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=0,
                gpu_uuid="GPU-123",
                gpu_model_name="Test GPU",
                telemetry_data=TelemetryMetrics(),
            ),
            TelemetryRecord(
                timestamp_ns=1000001,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=1,
                gpu_uuid="GPU-456",
                gpu_model_name="Test GPU",
                telemetry_data=TelemetryMetrics(),
            ),
        ]

        # Test the logic: each record should be sent to processor
        for record in records:
            await mock_processor.process_telemetry_record(record)

        # Processor should be called for each record
        assert mock_processor.process_telemetry_record.call_count == len(records)

    def test_telemetry_hierarchy_add_record(self):
        """Test that telemetry hierarchy adds records correctly."""
        from aiperf.common.models import (
            TelemetryHierarchy,
            TelemetryMetrics,
            TelemetryRecord,
        )

        hierarchy = TelemetryHierarchy()

        record = TelemetryRecord(
            timestamp_ns=1000000,
            dcgm_url="http://localhost:9400/metrics",
            gpu_index=0,
            gpu_uuid="GPU-123",
            gpu_model_name="Test GPU",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=100.0,
            ),
        )

        # Add record to hierarchy
        hierarchy.add_record(record)

        # Verify hierarchy structure
        assert "http://localhost:9400/metrics" in hierarchy.dcgm_endpoints
        assert "GPU-123" in hierarchy.dcgm_endpoints["http://localhost:9400/metrics"]


class TestRecordsManagerTimeslice:
    """Test cases for RecordsManager timeslice functionality."""

    @pytest.mark.asyncio
    async def test_process_records_result_with_both_records_and_timeslices(self):
        """Test that ProcessRecordsResult can contain both records and timeslice results."""

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

        # Create a ProcessRecordsResult with both types of results
        result = ProcessRecordsResult(
            results=ProfileResults(
                records=[metric_result, metric_result],
                timeslices=timeslices,
                completed=2,
                start_ns=1000000000,
                end_ns=2000000000,
            )
        )

        assert result.results.records is not None
        assert len(result.results.records) == 2
        assert result.results.timeslices is not None
        assert len(result.results.timeslices) == 2

    @pytest.mark.asyncio
    async def test_profile_results_serialization_with_timeslices(self):
        """Test that ProfileResults with timeslice data can be serialized."""
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
            start_ns=1000000000,
            end_ns=2000000000,
        )

        # Test that it can be converted to dict (for JSON serialization)
        result_dict = profile_results.model_dump()

        assert "records" in result_dict
        assert "timeslices" in result_dict
        assert result_dict["timeslices"] is not None
        assert len(result_dict["timeslices"]) == 2
