# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase, ProfileCancelReason
from aiperf.common.environment import Environment
from aiperf.common.messages import BaseServiceErrorMessage, ProfileCancelCommand
from aiperf.common.messages.inference_messages import (
    MetricRecordsData,
    RecordsMessage,
)
from aiperf.common.messages.telemetry_messages import TelemetryRecordsMessage
from aiperf.common.models import (
    BranchStats,
    CreditPhaseStats,
    MetricResult,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProfileResults,
    TelemetryMetrics,
    TelemetryRecord,
    TimesliceResult,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.common.types import MetricTagT
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.cache_reporting_hint import CACHE_REPORTING_HINT
from aiperf.plugin.enums import AccumulatorType, TimingMode, UIType
from aiperf.records import records_manager as records_manager_module
from aiperf.records import records_manager_processing
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_manager import ErrorTrackingState, RecordsManager
from aiperf.records.records_manager_processing import LoadedAnalyzer
from aiperf.records.records_tracker import RecordsTracker
from aiperf.timing.config import CreditPhaseConfig

# Helper functions


def test_orphan_phase_tracker_does_not_block_aggregate_completion() -> None:
    tracker = RecordsTracker()
    tracker.update_phase_info(
        CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            profiling_index=0,
            phase_name="load",
            phase_kind="profiling",
            start_ns=100,
            requests_end_ns=300,
            baseline_start_ns=90,
            baseline_end_ns=310,
            final_requests_completed=1,
        )
    )
    orphan_tracker = tracker._get_phase_tracker(CreditPhase.PROFILING, None)
    orphan_tracker.increment_error_records()
    tracker.update_from_request(
        MetricRecordMetadata(
            session_num=1,
            request_start_ns=101,
            request_end_ns=110,
            worker_id="worker",
            record_processor_id="processor",
            benchmark_phase=CreditPhase.PROFILING,
            phase_index=1,
        ),
        None,
    )

    aggregate = tracker.create_aggregate_stats_for_phase(CreditPhase.PROFILING)

    assert aggregate.success_records == 1
    assert aggregate.error_records == 1
    assert aggregate.baseline_start_ns == 90
    assert aggregate.baseline_end_ns == 310
    assert tracker.check_and_set_all_records_received_for_phase(CreditPhase.PROFILING)


def test_live_phase_counters_use_indexed_tracker_without_creating_orphan() -> None:
    tracker = RecordsTracker()
    tracker.update_phase_info(
        CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            phase_index=0,
            profiling_index=0,
            phase_name="profiling",
            phase_kind="profiling",
            start_ns=100,
        )
    )
    metadata = MetricRecordMetadata(
        session_num=1,
        request_start_ns=101,
        request_end_ns=110,
        worker_id="worker",
        record_processor_id="processor",
        benchmark_phase=CreditPhase.PROFILING,
        phase_index=0,
    )
    tracker.update_from_request(metadata, None)
    tracker.update_from_request(
        metadata, ErrorDetails(type="ClientConnectorError", message="refused")
    )

    assert tracker.total_records_for_phase(CreditPhase.PROFILING, 0) == 2
    assert tracker.error_records_for_phase(CreditPhase.PROFILING, 0) == 1
    assert tracker.total_records_for_phase(CreditPhase.PROFILING) == 2
    assert tracker.error_records_for_phase(CreditPhase.PROFILING) == 1
    assert (CreditPhase.PROFILING, None) not in tracker._phase_trackers


@pytest.mark.asyncio
async def test_failed_request_threshold_checks_current_phase_index() -> None:
    manager = RecordsManager.__new__(RecordsManager)
    manager._failed_request_threshold = 0.10
    manager._failed_request_grace_floor = 10
    manager._failed_request_abort_triggered = False
    manager._records_tracker = RecordsTracker()
    manager.warning = MagicMock()
    manager.publish = AsyncMock()
    manager._on_profile_cancel_command = AsyncMock()
    manager.service_id = "records-manager"

    error = ErrorDetails(type="ClientConnectorError", message="refused")
    for phase_index in (0, 1):
        manager._records_tracker.update_phase_info(
            CreditPhaseStats(
                phase=CreditPhase.PROFILING,
                phase_index=phase_index,
                profiling_index=phase_index,
                phase_name=f"profiling-{phase_index}",
                phase_kind="profiling",
                start_ns=100 + phase_index,
            )
        )
        metadata = MetricRecordMetadata(
            session_num=phase_index,
            request_start_ns=101,
            request_end_ns=110,
            worker_id="worker",
            record_processor_id="processor",
            benchmark_phase=CreditPhase.PROFILING,
            phase_index=phase_index,
        )
        for request_index in range(10):
            request_error = error if phase_index == 0 and request_index > 0 else None
            manager._records_tracker.update_from_request(metadata, request_error)

    await manager._maybe_trigger_failed_request_abort(
        CreditPhase.PROFILING, phase_index=1
    )
    manager.publish.assert_not_awaited()

    await manager._maybe_trigger_failed_request_abort(
        CreditPhase.PROFILING, phase_index=0
    )

    manager.publish.assert_awaited_once()
    command = manager.publish.await_args.args[0]
    assert isinstance(command, ProfileCancelCommand)
    assert command.reason == ProfileCancelReason.FAILED_REQUEST_THRESHOLD
    manager._on_profile_cancel_command.assert_awaited_once_with(command)
    assert manager._failed_request_abort_triggered


@pytest.mark.asyncio
async def test_failed_request_threshold_publish_failure_retries_later() -> None:
    manager = RecordsManager.__new__(RecordsManager)
    manager._failed_request_threshold = 0.10
    manager._failed_request_grace_floor = 10
    manager._failed_request_abort_triggered = False
    manager._records_tracker = MagicMock()
    manager._records_tracker.total_records_for_phase.return_value = 10
    manager._records_tracker.error_records_for_phase.return_value = 10
    manager.warning = MagicMock()
    manager.publish = AsyncMock(side_effect=RuntimeError("message bus unavailable"))
    manager._on_profile_cancel_command = AsyncMock()
    manager.service_id = "records-manager"

    await manager._maybe_trigger_failed_request_abort(
        CreditPhase.PROFILING, phase_index=0
    )

    assert not manager._failed_request_abort_triggered
    manager._on_profile_cancel_command.assert_not_awaited()


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


def _telemetry_record(gpu_index: int = 0) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp_ns=1_000_000 + gpu_index,
        telemetry_source_url="http://localhost:9400/metrics",
        gpu_index=gpu_index,
        gpu_uuid=f"GPU-{gpu_index}",
        gpu_model_name="Test GPU",
        telemetry_data=TelemetryMetrics(gpu_power_usage=100.0),
    )


class TestRecordsManagerTelemetry:
    """Telemetry records route through the unified record dispatcher."""

    @pytest.mark.asyncio
    async def test_on_telemetry_records_valid_dispatches_each_record(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._telemetry_state = ErrorTrackingState()
        manager._dispatch_record = AsyncMock(return_value=[])
        records = [_telemetry_record(0), _telemetry_record(1)]
        message = TelemetryRecordsMessage(
            service_id="test_service",
            collector_id="test_collector",
            telemetry_source_url="http://localhost:9400/metrics",
            records=records,
            error=None,
        )

        await manager._on_telemetry_records(message)

        assert manager._dispatch_record.await_args_list == [
            ((records[0],),),
            ((records[1],),),
        ]
        assert manager._telemetry_state.error_counts == {}

    @pytest.mark.asyncio
    async def test_on_telemetry_dispatch_errors_are_tracked(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._telemetry_state = ErrorTrackingState()
        dispatch_error = RuntimeError("telemetry writer failed")
        manager._dispatch_record = AsyncMock(return_value=[dispatch_error])

        await manager._on_telemetry_records(
            TelemetryRecordsMessage(
                service_id="test_service",
                collector_id="test_collector",
                telemetry_source_url="http://localhost:9400/metrics",
                records=[_telemetry_record()],
                error=None,
            )
        )

        tracked = ErrorDetails.from_exception(dispatch_error)
        assert manager._telemetry_state.error_counts[tracked] == 1

    @pytest.mark.asyncio
    async def test_on_telemetry_records_invalid_tracks_error(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._telemetry_state = ErrorTrackingState()
        manager._dispatch_record = AsyncMock(return_value=[])
        error = ErrorDetails(message="Test error", code=500)

        await manager._on_telemetry_records(
            TelemetryRecordsMessage(
                service_id="test_service",
                collector_id="test_collector",
                telemetry_source_url="http://localhost:9400/metrics",
                records=[],
                error=error,
            )
        )

        assert manager._telemetry_state.error_counts[error] == 1
        manager._dispatch_record.assert_not_awaited()


class TestRecordsManagerMetricRecordDispatchErrors:
    """Metric-handler failures must surface in the phase error summary rather
    than being silently dropped while the record is marked processed."""

    def _make_manager(self) -> RecordsManager:
        manager = RecordsManager.__new__(RecordsManager)
        manager.debug = MagicMock()
        manager.error = MagicMock()
        manager.trace = MagicMock()
        manager.is_enabled_for = MagicMock(return_value=False)
        manager._dataset_configured_event = asyncio.Event()
        manager._dataset_configured_event.set()
        manager._records_tracker = MagicMock()
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = False
        manager._error_tracker = ErrorTracker()
        manager._complete_credit_phases = set()
        manager._warned_missing_cache_reporting = False
        manager._failed_request_threshold = None
        manager._failed_request_abort_triggered = False
        manager._skipped_context_overflow_counts_by_phase = {
            CreditPhase.WARMUP: 0,
            CreditPhase.PROFILING: 0,
        }
        return manager

    def _records_message(self) -> RecordsMessage:
        record = create_metric_record_data(1_000, 2_000)
        return RecordsMessage(
            service_id="rp", metadata=record.metadata, records=[record]
        )

    @pytest.mark.asyncio
    async def test_metric_dispatch_error_recorded_in_phase_error_summary(self) -> None:
        manager = self._make_manager()
        dispatch_error = RuntimeError("metric accumulator failed")
        manager._dispatch_record = AsyncMock(return_value=[dispatch_error])

        await manager._on_records(self._records_message())

        # Record is still counted, but the handler failure is not swallowed.
        manager._records_tracker.update_from_request.assert_called_once()
        summary = manager._error_tracker.get_error_summary_for_phase(
            CreditPhase.PROFILING
        )
        tracked = ErrorDetails.from_exception(dispatch_error)
        assert any(e.error_details == tracked for e in summary)

    @pytest.mark.asyncio
    async def test_on_records_tracks_errors_by_phase_index(self) -> None:
        manager = self._make_manager()
        dispatch_error = RuntimeError("metric accumulator failed")
        request_error = ErrorDetails(
            code=499, type="RequestCancellationError", message="cancelled"
        )
        manager._dispatch_record = AsyncMock(return_value=[dispatch_error])
        message = self._records_message()
        message.metadata.phase_index = 2
        message.error = request_error

        await manager._on_records(message)

        indexed_summary = manager._error_tracker.get_error_summary_for_phase(
            CreditPhase.PROFILING, phase_index=2
        )
        indexed_errors = {item.error_details: item.count for item in indexed_summary}
        assert indexed_errors[request_error] == 1
        assert indexed_errors[ErrorDetails.from_exception(dispatch_error)] == 1

    @pytest.mark.asyncio
    async def test_successful_metric_dispatch_records_no_phase_error(self) -> None:
        manager = self._make_manager()
        manager._dispatch_record = AsyncMock(return_value=[])

        await manager._on_records(self._records_message())

        assert (
            manager._error_tracker.get_error_summary_for_phase(CreditPhase.PROFILING)
            == []
        )

    @pytest.mark.asyncio
    async def test_context_overflow_skip_counts_as_success_not_error(self) -> None:
        """AGENTIC_REPLAY overflow skips must advance the success counter and
        must NOT inflate error_records (which would trip --failed-request-threshold).
        """
        manager = self._make_manager()
        manager._dispatch_record = AsyncMock(return_value=[])
        record = create_metric_record_data(1_000, 2_000)
        record.metadata.context_overflow_skip = True
        message = RecordsMessage(
            service_id="rp",
            metadata=record.metadata,
            records=[record],
            error=ErrorDetails(code=400, type="ContextOverflow", message="overflow"),
        )

        await manager._on_records(message)

        assert (
            manager._skipped_context_overflow_counts_by_phase[CreditPhase.PROFILING]
            == 1
        )
        manager._records_tracker.update_from_request.assert_called_once_with(
            message.metadata, None
        )
        manager._dispatch_record.assert_not_called()
        assert (
            manager._error_tracker.get_error_summary_for_phase(CreditPhase.PROFILING)
            == []
        )

    @pytest.mark.asyncio
    async def test_warmup_plus_single_profiling_builds_phase_results(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._records_tracker = MagicMock()
        warmup_tracker = MagicMock()
        warmup_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.WARMUP,
            phase_index=0,
            phase_name="warmup",
            phase_kind="warmup",
            start_ns=1_000,
            requests_end_ns=2_000,
        )
        profile_tracker = MagicMock()
        profile_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            profiling_index=0,
            phase_name="load",
            phase_kind="profiling",
            start_ns=3_000,
            requests_end_ns=4_000,
        )
        manager._records_tracker._phase_trackers = {
            (CreditPhase.WARMUP, 0): warmup_tracker,
            (CreditPhase.PROFILING, 1): profile_tracker,
        }
        manager._accumulators = {}
        manager._metric_record_accumulators = []
        manager._telemetry_state = ErrorTrackingState()
        manager._server_metrics_state = ErrorTrackingState()
        manager._gpu_telemetry_accumulator = None
        manager._server_metrics_accumulator = None
        manager._error_tracker = ErrorTracker()

        results = await RecordsManager._build_phase_profile_results(
            manager, CreditPhase.PROFILING, cancelled=False
        )

        assert results is not None
        assert [r.phase_name for r in results] == ["warmup", "load"]

    @staticmethod
    def _warmup_summary_manager(
        warmup_overflow: int, base_metrics: list[MetricResult]
    ) -> RecordsManager:
        manager = RecordsManager.__new__(RecordsManager)
        manager.error = MagicMock()
        manager._records_tracker = MagicMock()
        manager._records_tracker.was_phase_cancelled = MagicMock(return_value=False)
        manager._has_records_for_phase = MagicMock(return_value=True)
        manager._summarize_metric_record_accumulators = AsyncMock(
            return_value=(list(base_metrics), None, [], None)
        )
        manager._skipped_context_overflow_counts_by_phase = {
            CreditPhase.WARMUP: warmup_overflow,
            CreditPhase.PROFILING: 7,
        }
        return manager

    @pytest.mark.asyncio
    async def test_warmup_summary_injects_context_overflow_metric(self) -> None:
        base = [MetricResult(tag="request_latency", header="h", unit="ms", avg=1.0)]
        manager = self._warmup_summary_manager(warmup_overflow=3, base_metrics=base)

        results = await RecordsManager._summarize_warmup_metric_records(manager)

        assert results is not None
        overflow = [r for r in results if r.tag == "context_overflow_count"]
        assert len(overflow) == 1
        # WARMUP count (3) is surfaced, not the PROFILING count (7).
        assert overflow[0].avg == 3.0
        assert overflow[0].count == 1
        assert overflow[0].unit == "requests"

    @pytest.mark.asyncio
    async def test_warmup_summary_omits_metric_when_no_overflow(self) -> None:
        base = [MetricResult(tag="request_latency", header="h", unit="ms", avg=1.0)]
        manager = self._warmup_summary_manager(warmup_overflow=0, base_metrics=base)

        results = await RecordsManager._summarize_warmup_metric_records(manager)

        assert results is not None
        assert all(r.tag != "context_overflow_count" for r in results)

    @pytest.mark.asyncio
    async def test_warmup_summary_surfaces_overflow_when_all_requests_skipped(
        self,
    ) -> None:
        """Even at 100% warmup overflow the count is still surfaced.

        Context-overflow skips are recorded as success via
        ``update_from_request(metadata, None)``, so ``_has_records_for_phase``
        (which reads the records tracker's ``total_records``, not the metric
        accumulator) returns True and the injection block is reached even when
        the metric accumulator collected nothing.
        """
        manager = RecordsManager.__new__(RecordsManager)
        manager.error = MagicMock()

        tracker = RecordsTracker()
        for i in range(3):
            tracker.update_from_request(
                MetricRecordMetadata(
                    session_num=i,
                    request_start_ns=1,
                    request_end_ns=2,
                    worker_id="w0",
                    record_processor_id="rp0",
                    benchmark_phase=CreditPhase.WARMUP,
                    phase_index=0,
                    context_overflow_skip=True,
                ),
                None,
            )
        manager._records_tracker = tracker
        # Metric accumulator collected nothing (every request was skipped).
        manager._summarize_metric_record_accumulators = AsyncMock(
            return_value=([], None, [], None)
        )
        manager._skipped_context_overflow_counts_by_phase = {
            CreditPhase.WARMUP: 3,
            CreditPhase.PROFILING: 0,
        }

        results = await RecordsManager._summarize_warmup_metric_records(manager)

        assert results is not None
        overflow = [r for r in results if r.tag == "context_overflow_count"]
        assert len(overflow) == 1
        assert overflow[0].avg == 3.0

    @pytest.mark.asyncio
    async def test_single_profiling_phase_does_not_build_duplicate_phase_results(
        self,
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._records_tracker = MagicMock()
        profile_tracker = MagicMock()
        profile_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.PROFILING,
            phase_index=0,
            profiling_index=0,
            phase_name="profiling",
            phase_kind="profiling",
            start_ns=1_000,
            requests_end_ns=2_000,
        )
        manager._records_tracker._phase_trackers = {
            (CreditPhase.PROFILING, 0): profile_tracker,
        }

        results = await RecordsManager._build_phase_profile_results(
            manager, CreditPhase.PROFILING, cancelled=False
        )

        assert results is None

    @pytest.mark.asyncio
    async def test_phase_telemetry_exports_use_baseline_window(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._records_tracker = MagicMock()
        warmup_tracker = MagicMock()
        warmup_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.WARMUP,
            phase_index=0,
            phase_name="warmup",
            phase_kind="warmup",
            start_ns=100,
            requests_end_ns=200,
        )
        profile_tracker = MagicMock()
        profile_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            profiling_index=0,
            phase_name="load",
            phase_kind="profiling",
            start_ns=1_000,
            requests_end_ns=2_000,
            baseline_start_ns=900,
            baseline_end_ns=2_200,
        )
        manager._records_tracker._phase_trackers = {
            (CreditPhase.WARMUP, 0): warmup_tracker,
            (CreditPhase.PROFILING, 1): profile_tracker,
        }
        manager._accumulators = {}
        manager._metric_record_accumulators = []
        manager._telemetry_state = ErrorTrackingState()
        manager._server_metrics_state = ErrorTrackingState()
        manager._error_tracker = ErrorTracker()

        class _CaptureAccumulator:
            def __init__(self, result) -> None:
                self.result = result
                self.contexts: list[ExportContext] = []

            async def export_results(self, ctx: ExportContext):
                self.contexts.append(ctx)
                return self.result

        gpu_accumulator = _CaptureAccumulator(SimpleNamespace(endpoints={}))
        server_accumulator = _CaptureAccumulator(SimpleNamespace(endpoint_summaries={}))
        manager._gpu_telemetry_accumulator = gpu_accumulator
        manager._server_metrics_accumulator = server_accumulator

        results = await RecordsManager._build_phase_profile_results(
            manager, CreditPhase.PROFILING, cancelled=False
        )

        assert results is not None
        load_result = next(result for result in results if result.phase_name == "load")
        assert load_result.baseline_start_ns == 900
        assert load_result.baseline_end_ns == 2_200
        assert gpu_accumulator.contexts[1].start_ns == 900
        assert gpu_accumulator.contexts[1].end_ns == 2_200
        assert gpu_accumulator.contexts[1].is_phase_scoped is True
        assert server_accumulator.contexts[1].start_ns == 900
        assert server_accumulator.contexts[1].end_ns == 2_200
        assert server_accumulator.contexts[1].is_phase_scoped is True
        assert load_result.telemetry_results is None
        assert load_result.server_metrics_results is None
        assert load_result.telemetry_warnings == []
        assert load_result.server_metrics_warnings == []

    @pytest.mark.asyncio
    async def test_root_telemetry_export_uses_bounded_profiling_window(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager.debug = MagicMock()
        manager._telemetry_state = ErrorTrackingState()
        manager._records_tracker = MagicMock()
        manager._records_tracker.create_aggregate_stats_for_phase.return_value = (
            PhaseRecordsStats(
                phase=CreditPhase.PROFILING,
                start_ns=1_000,
                requests_end_ns=2_000,
            )
        )
        manager._gpu_telemetry_accumulator = MagicMock()
        manager._gpu_telemetry_accumulator.export_results = AsyncMock(return_value=None)

        result = await RecordsManager._process_telemetry_results(manager)

        assert result.results is None
        ctx = manager._gpu_telemetry_accumulator.export_results.await_args.args[0]
        assert ctx.start_ns == 1_000
        assert ctx.end_ns == 2_000 + Environment.GPU.FINAL_SCRAPE_GRACE_NS
        assert ctx.phase == CreditPhase.PROFILING

    def test_multi_profiling_aggregate_rates_use_active_phase_duration(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        first_tracker = MagicMock()
        first_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.PROFILING,
            phase_index=0,
            profiling_index=0,
            phase_name="low",
            phase_kind="profiling",
            start_ns=1_000_000_000,
            requests_end_ns=2_000_000_000,
        )
        second_tracker = MagicMock()
        second_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.PROFILING,
            phase_index=2,
            profiling_index=1,
            phase_name="storm",
            phase_kind="profiling",
            start_ns=7_000_000_000,
            requests_end_ns=9_000_000_000,
        )
        manager._has_multiple_phase_instances = lambda phase: (
            phase == CreditPhase.PROFILING
        )
        manager._records_tracker = MagicMock()
        manager._records_tracker._phase_trackers = {
            (CreditPhase.PROFILING, 0): first_tracker,
            (CreditPhase.PROFILING, 2): second_tracker,
        }
        records = [
            MetricResult(
                tag="benchmark_duration",
                header="Benchmark Duration",
                unit="sec",
                avg=8,
            ),
            MetricResult(
                tag="request_count", header="Request Count", unit="requests", avg=60
            ),
            MetricResult(
                tag="request_throughput",
                header="Request Throughput",
                unit="requests/sec",
                avg=7.5,
            ),
            MetricResult(tag="total_isl", header="Total ISL", unit="tokens", avg=120),
            MetricResult(tag="total_osl", header="Total OSL", unit="tokens", avg=30),
            MetricResult(
                tag="input_token_throughput",
                header="Input Token Throughput",
                unit="tokens/sec",
                avg=15,
            ),
            MetricResult(
                tag="output_token_throughput",
                header="Output Token Throughput",
                unit="tokens/sec",
                avg=3.75,
            ),
            MetricResult(
                tag="total_token_throughput",
                header="Total Token Throughput",
                unit="tokens/sec",
                avg=18.75,
            ),
        ]

        RecordsManager._adjust_multi_phase_aggregate_rates(
            manager, CreditPhase.PROFILING, records
        )

        by_tag = {result.tag: result for result in records}
        assert by_tag["benchmark_duration"].avg == 3
        assert by_tag["request_throughput"].avg == 20
        assert by_tag["input_token_throughput"].avg == 40
        assert by_tag["output_token_throughput"].avg == 10
        assert by_tag["total_token_throughput"].avg == 50

    def test_multi_warmup_aggregate_rates_use_active_phase_duration(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        first_tracker = MagicMock()
        first_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.WARMUP,
            phase_index=0,
            phase_name="prime",
            phase_kind="warmup",
            start_ns=1_000_000_000,
            requests_end_ns=2_000_000_000,
        )
        second_tracker = MagicMock()
        second_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.WARMUP,
            phase_index=2,
            phase_name="settle",
            phase_kind="warmup",
            start_ns=7_000_000_000,
            requests_end_ns=9_000_000_000,
        )
        manager._has_multiple_phase_instances = lambda phase: (
            phase == CreditPhase.WARMUP
        )
        manager._records_tracker = MagicMock()
        manager._records_tracker._phase_trackers = {
            (CreditPhase.WARMUP, 0): first_tracker,
            (CreditPhase.WARMUP, 2): second_tracker,
        }
        records = [
            MetricResult(
                tag="benchmark_duration",
                header="Benchmark Duration",
                unit="sec",
                avg=8,
            ),
            MetricResult(
                tag="request_count", header="Request Count", unit="requests", avg=24
            ),
            MetricResult(
                tag="request_throughput",
                header="Request Throughput",
                unit="requests/sec",
                avg=3,
            ),
            MetricResult(tag="total_isl", header="Total ISL", unit="tokens", avg=96),
            MetricResult(tag="total_osl", header="Total OSL", unit="tokens", avg=48),
            MetricResult(
                tag="input_token_throughput",
                header="Input Token Throughput",
                unit="tokens/sec",
                avg=12,
            ),
            MetricResult(
                tag="output_token_throughput",
                header="Output Token Throughput",
                unit="tokens/sec",
                avg=6,
            ),
            MetricResult(
                tag="total_token_throughput",
                header="Total Token Throughput",
                unit="tokens/sec",
                avg=18,
            ),
        ]

        RecordsManager._adjust_multi_phase_aggregate_rates(
            manager, CreditPhase.WARMUP, records
        )

        by_tag = {result.tag: result for result in records}
        assert by_tag["benchmark_duration"].avg == 3
        assert by_tag["request_throughput"].avg == 8
        assert by_tag["input_token_throughput"].avg == 32
        assert by_tag["output_token_throughput"].avg == 16
        assert by_tag["total_token_throughput"].avg == 48

    @pytest.mark.asyncio
    async def test_realtime_delta_resets_when_phase_index_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._records_tracker = MagicMock()
        manager._records_tracker.create_stats_for_phase.return_value = (
            PhaseRecordsStats(
                phase=CreditPhase.PROFILING,
                phase_index=2,
                phase_name="storm",
                phase_kind="profiling",
                start_ns=0,
                records_end_ns=1_000_000_000,
                success_records=1,
            )
        )
        manager._metric_record_accumulators = []
        manager._prev_realtime_snapshot = (10, 5.0)
        manager._prev_realtime_phase_index = 1
        manager._server_metrics_accumulator = None
        manager.publish = AsyncMock()
        manager.service_id = "records_manager"
        manager.run = SimpleNamespace(cfg=SimpleNamespace(ui_type=UIType.NONE))

        async def fake_generate_realtime_metrics(*args, **kwargs):
            return [
                MetricResult(
                    tag="request_throughput",
                    header="Request Throughput",
                    unit="requests/sec",
                    avg=1.0,
                )
            ]

        captured: dict[str, object] = {}

        def fake_render(
            metric_results, phase_stats, prev_snapshot, server_snapshot=None
        ):
            captured["prev_snapshot"] = prev_snapshot
            return "rendered"

        monkeypatch.setattr(
            records_manager_module,
            "generate_realtime_metrics",
            fake_generate_realtime_metrics,
        )
        monkeypatch.setattr(
            records_manager_processing,
            "filter_display_metrics",
            lambda metrics: metrics,
        )
        monkeypatch.setattr(
            records_manager_module, "_render_realtime_block", fake_render
        )

        await manager._report_realtime_metrics(emit_log_block=False)

        assert captured["prev_snapshot"] is None
        assert manager._prev_realtime_snapshot == (1, 1.0)
        assert manager._prev_realtime_phase_index == 2

    @pytest.mark.asyncio
    async def test_disabled_observability_skips_phase_exports(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager.run = SimpleNamespace(
            cfg=SimpleNamespace(
                gpu_telemetry_disabled=True,
                server_metrics_disabled=True,
            )
        )
        manager._records_tracker = MagicMock()
        warmup_tracker = MagicMock()
        warmup_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.WARMUP,
            phase_index=0,
            phase_name="warmup",
            phase_kind="warmup",
            start_ns=100,
            requests_end_ns=200,
        )
        profile_tracker = MagicMock()
        profile_tracker.create_stats.return_value = PhaseRecordsStats(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            profiling_index=0,
            phase_name="load",
            phase_kind="profiling",
            start_ns=1_000,
            requests_end_ns=2_000,
            baseline_start_ns=900,
            baseline_end_ns=2_200,
        )
        manager._records_tracker._phase_trackers = {
            (CreditPhase.WARMUP, 0): warmup_tracker,
            (CreditPhase.PROFILING, 1): profile_tracker,
        }
        manager._accumulators = {}
        manager._metric_record_accumulators = []
        manager._telemetry_state = ErrorTrackingState()
        manager._server_metrics_state = ErrorTrackingState()
        manager._error_tracker = ErrorTracker()

        class _UnexpectedAccumulator:
            async def export_results(self, ctx: ExportContext):
                raise AssertionError("disabled observability should not export")

        manager._gpu_telemetry_accumulator = _UnexpectedAccumulator()
        manager._server_metrics_accumulator = _UnexpectedAccumulator()

        results = await RecordsManager._build_phase_profile_results(
            manager, CreditPhase.PROFILING, cancelled=False
        )

        assert results is not None
        assert results[0].telemetry_results is None
        assert results[0].server_metrics_results is None
        assert results[0].telemetry_warnings == []
        assert results[0].server_metrics_warnings == []


class TestRecordsManagerTimeslice:
    """ProfileResults stores accumulator-backed timeslices."""

    def _timeslices(self, metric_result: MetricResult) -> list[TimesliceResult]:
        return [
            TimesliceResult(
                start_ns=1_000_000_000,
                end_ns=2_000_000_000,
                metric_results={metric_result.tag: metric_result},
            ),
            TimesliceResult(
                start_ns=2_000_000_000,
                end_ns=3_000_000_000,
                metric_results={metric_result.tag: metric_result},
            ),
        ]

    def test_process_records_result_with_both_records_and_timeslices(self) -> None:
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        result = ProcessRecordsResult(
            results=ProfileResults(
                records=[metric_result, metric_result],
                timeslices=self._timeslices(metric_result),
                completed=2,
                start_ns=1_000_000_000,
                end_ns=2_000_000_000,
            )
        )

        assert result.results.records is not None
        assert len(result.results.records) == 2
        assert result.results.timeslices is not None
        assert len(result.results.timeslices) == 2

    def test_profile_results_serialization_with_timeslices(self) -> None:
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )
        profile_results = ProfileResults(
            records=[metric_result],
            timeslices=self._timeslices(metric_result),
            completed=1,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )

        result_dict = profile_results.model_dump()

        assert "records" in result_dict
        assert "timeslices" in result_dict
        assert "timeslice_metric_results" not in result_dict
        assert len(result_dict["timeslices"]) == 2


def _create_credit_phase_stats() -> CreditPhaseStats:
    return CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        start_ns=1_000_000_000,
        sent_end_ns=2_000_000_000,
        requests_end_ns=3_000_000_000,
        total_expected_requests=64,
        expected_duration_sec=60.0,
        expected_grace_period_sec=30.0,
        requests_sent=64,
        requests_completed=64,
        requests_cancelled=0,
        request_errors=0,
        sent_sessions=64,
        completed_sessions=64,
        cancelled_sessions=0,
        total_session_turns=64,
    )


def _create_manager_for_timing_dispatch() -> RecordsManager:
    manager = RecordsManager.__new__(RecordsManager)
    manager._dataset_configured_event = asyncio.Event()
    manager._dataset_configured_event.set()
    manager._records_tracker = MagicMock()
    manager._error_tracker = MagicMock()
    manager._complete_credit_phases = set()
    manager._phase_branch_stats = {}
    manager._latest_branch_stats = None
    manager._dispatch_record = AsyncMock(return_value=[])
    manager.info = MagicMock()
    manager.notice = MagicMock()
    manager.debug = MagicMock()
    manager.trace = MagicMock()
    manager.is_enabled_for = MagicMock(return_value=False)
    manager._handle_all_records_received = AsyncMock()
    manager._credits_complete_received = False
    manager._all_records_received_phases = set()
    manager._warned_missing_cache_reporting = False
    manager._failed_request_threshold = None
    manager._failed_request_abort_triggered = False
    manager._skipped_context_overflow_counts_by_phase = {
        CreditPhase.WARMUP: 0,
        CreditPhase.PROFILING: 0,
    }
    return manager


def _metric_records_message(
    phase: CreditPhase = CreditPhase.PROFILING,
) -> RecordsMessage:
    metadata = MetricRecordMetadata(
        session_num=17,
        conversation_id="conv-2026-05-14-race",
        turn_index=0,
        request_start_ns=1_000_000_000,
        request_end_ns=1_250_000_000,
        worker_id="worker-a100-03",
        record_processor_id="record-processor-rp-7f2a",
        benchmark_phase=phase,
    )
    return RecordsMessage(
        service_id="record-processor-rp-7f2a",
        metadata=metadata,
        records=[
            MetricRecordsData(
                metadata=metadata, metrics={"request_latency": 250_000_000}
            )
        ],
    )


class TestRecordsManagerTimingDispatch:
    @pytest.mark.asyncio
    async def test_on_credit_phase_start_dispatches_timing_snapshot(self) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats()
        message = CreditPhaseStartMessage(
            service_id="timing-manager",
            stats=stats,
            config=CreditPhaseConfig(
                phase=CreditPhase.PROFILING,
                timing_mode=TimingMode.REQUEST_RATE,
            ),
        )

        await manager._on_credit_phase_start(message)

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._dispatch_record.assert_awaited_once_with(stats, warn_if_unrouted=False)

    @pytest.mark.asyncio
    async def test_on_credit_phase_progress_dispatches_timing_snapshot(self) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats()

        await manager._on_credit_phase_progress(
            CreditPhaseProgressMessage(service_id="timing-manager", stats=stats)
        )

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._dispatch_record.assert_awaited_once_with(stats, warn_if_unrouted=False)

    @pytest.mark.asyncio
    async def test_on_credit_phase_sending_complete_dispatches_timing_snapshot(
        self,
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats().model_copy(
            update={"final_requests_sent": 64}
        )

        await manager._on_credit_phase_sending_complete(
            CreditPhaseSendingCompleteMessage(
                service_id="timing-manager",
                stats=stats,
            )
        )

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._dispatch_record.assert_awaited_once_with(stats, warn_if_unrouted=False)

    @pytest.mark.asyncio
    async def test_on_credit_phase_complete_dispatches_timing_snapshot(self) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats().model_copy(
            update={"final_requests_completed": 64}
        )
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = False
        manager._records_tracker.create_stats_for_phase.return_value = MagicMock(
            total_records=64,
            final_requests_completed=64,
        )

        await manager._on_credit_phase_complete(
            CreditPhaseCompleteMessage(service_id="timing-manager", stats=stats)
        )

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._dispatch_record.assert_awaited_once_with(stats, warn_if_unrouted=False)

    @pytest.mark.asyncio
    async def test_on_credit_phase_complete_with_pending_final_count_does_not_raise(
        self,
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats().model_copy(
            update={"final_requests_completed": None}
        )
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = False
        manager._records_tracker.create_stats_for_phase.return_value = MagicMock(
            total_records=10,
            final_requests_completed=None,
        )

        await manager._on_credit_phase_complete(
            CreditPhaseCompleteMessage(service_id="timing-manager", stats=stats)
        )

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._dispatch_record.assert_awaited_once_with(stats, warn_if_unrouted=False)
        manager._handle_all_records_received.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_metric_records_records_complete_before_phase_complete_defers_finalization(
        self,
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = True
        manager._records_tracker.create_stats_for_phase.return_value = MagicMock(
            total_records=64,
            final_requests_completed=64,
        )

        await manager._on_records(_metric_records_message())

        manager._records_tracker.update_from_request.assert_called_once()
        manager._records_tracker.check_and_set_all_records_received_for_phase.assert_not_called()
        manager._handle_all_records_received.assert_not_awaited()

        await manager._on_credit_phase_complete(
            CreditPhaseCompleteMessage(
                service_id="timing-manager",
                stats=_create_credit_phase_stats().model_copy(
                    update={"final_requests_completed": 64}
                ),
            )
        )

        manager._records_tracker.check_and_set_all_records_received_for_phase.assert_called_once_with(
            CreditPhase.PROFILING
        )
        manager._handle_all_records_received.assert_awaited_once_with(
            CreditPhase.PROFILING
        )

    @pytest.mark.asyncio
    async def test_on_credits_complete_before_phase_complete_defers_finalization(
        self,
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = True
        manager._records_tracker.create_stats_for_phase.return_value = MagicMock(
            total_records=64,
            final_requests_completed=64,
        )

        await manager._on_credits_complete(
            CreditsCompleteMessage(service_id="timing-manager")
        )

        manager._records_tracker.check_and_set_all_records_received_for_phase.assert_not_called()
        manager._handle_all_records_received.assert_not_awaited()

        await manager._on_credit_phase_complete(
            CreditPhaseCompleteMessage(
                service_id="timing-manager",
                stats=_create_credit_phase_stats().model_copy(
                    update={"final_requests_completed": 64}
                ),
            )
        )

        manager._records_tracker.check_and_set_all_records_received_for_phase.assert_called_once_with(
            CreditPhase.PROFILING
        )
        manager._handle_all_records_received.assert_awaited_once_with(
            CreditPhase.PROFILING
        )

    @pytest.mark.asyncio
    async def test_on_metric_records_after_phase_complete_finalization_observes_branch_stats(
        self,
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        branch_stats = BranchStats(children_spawned=3, parents_resumed=1)
        observed_branch_stats: list[BranchStats | None] = []

        async def _record_branch_stats_at_finalization(phase: CreditPhase) -> None:
            assert phase == CreditPhase.PROFILING
            observed_branch_stats.append(manager._latest_branch_stats)

        manager._handle_all_records_received = AsyncMock(
            side_effect=_record_branch_stats_at_finalization
        )
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = False
        manager._records_tracker.create_stats_for_phase.return_value = MagicMock(
            total_records=63,
            final_requests_completed=64,
        )

        await manager._on_credit_phase_complete(
            CreditPhaseCompleteMessage(
                service_id="timing-manager",
                stats=_create_credit_phase_stats().model_copy(
                    update={"final_requests_completed": 64}
                ),
                branch_stats=branch_stats,
            )
        )

        assert manager._latest_branch_stats is branch_stats
        manager._handle_all_records_received.assert_not_awaited()

        manager._records_tracker.check_and_set_all_records_received_for_phase.reset_mock()
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = True

        await manager._on_records(_metric_records_message())

        manager._records_tracker.check_and_set_all_records_received_for_phase.assert_called_once_with(
            CreditPhase.PROFILING
        )
        manager._handle_all_records_received.assert_awaited_once_with(
            CreditPhase.PROFILING
        )
        assert observed_branch_stats == [branch_stats]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_order",
        [
            ("phase_complete", "metric_record", "credits_complete"),
            ("phase_complete", "credits_complete", "metric_record"),
            ("metric_record", "phase_complete", "credits_complete"),
            ("metric_record", "credits_complete", "phase_complete"),
            ("credits_complete", "phase_complete", "metric_record"),
            ("credits_complete", "metric_record", "phase_complete"),
        ],
    )
    async def test_finalization_runs_once_for_all_terminal_event_orders(
        self, event_order: tuple[str, str, str]
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        manager._records_tracker = RecordsTracker()
        phase_complete = CreditPhaseCompleteMessage(
            service_id="timing-manager",
            stats=_create_credit_phase_stats().model_copy(
                update={"final_requests_completed": 1}
            ),
        )
        credits_complete = CreditsCompleteMessage(service_id="timing-manager")
        metric_record = _metric_records_message()

        for event in event_order:
            if event == "phase_complete":
                await manager._on_credit_phase_complete(phase_complete)
            elif event == "credits_complete":
                await manager._on_credits_complete(credits_complete)
            else:
                await manager._on_records(metric_record)

        manager._handle_all_records_received.assert_awaited_once_with(
            CreditPhase.PROFILING
        )

    @pytest.mark.asyncio
    async def test_finalization_runs_when_final_record_arrives_during_phase_complete_dispatch(
        self,
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        manager._records_tracker = RecordsTracker()
        timing_dispatch_started = asyncio.Event()
        release_timing_dispatch = asyncio.Event()

        async def _block_timing_dispatch(record, **_kwargs) -> list[BaseException]:
            if isinstance(record, CreditPhaseStats):
                timing_dispatch_started.set()
                await release_timing_dispatch.wait()
            return []

        manager._dispatch_record = AsyncMock(side_effect=_block_timing_dispatch)
        phase_complete_task = asyncio.create_task(
            manager._on_credit_phase_complete(
                CreditPhaseCompleteMessage(
                    service_id="timing-manager",
                    stats=_create_credit_phase_stats().model_copy(
                        update={"final_requests_completed": 1}
                    ),
                )
            )
        )
        await timing_dispatch_started.wait()

        await manager._on_records(_metric_records_message())
        manager._handle_all_records_received.assert_not_awaited()

        release_timing_dispatch.set()
        await phase_complete_task

        manager._handle_all_records_received.assert_awaited_once_with(
            CreditPhase.PROFILING
        )

    @pytest.mark.asyncio
    async def test_dispatch_errors_still_update_tracker_and_converge_barrier(
        self,
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        manager._dispatch_record = AsyncMock(
            return_value=[RuntimeError("handler boom")]
        )
        manager._complete_credit_phases = {CreditPhase.PROFILING}
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = True

        await manager._on_records(_metric_records_message())

        manager._records_tracker.update_from_request.assert_called_once()
        manager._records_tracker.check_and_set_all_records_received_for_phase.assert_called_once_with(
            CreditPhase.PROFILING
        )
        manager._handle_all_records_received.assert_awaited_once_with(
            CreditPhase.PROFILING
        )

    @pytest.mark.asyncio
    async def test_on_records_defers_finalization_for_multi_profiling_until_credits_complete(
        self,
    ) -> None:
        """Multi-profiling runs must not finalize PROFILING before CREDITS_COMPLETE."""
        manager = _create_manager_for_timing_dispatch()
        manager._complete_credit_phases = {CreditPhase.PROFILING}
        manager._has_multiple_profiling_phases = MagicMock(return_value=True)
        manager._credits_complete_received = False
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = True

        await manager._on_records(_metric_records_message())

        manager._records_tracker.check_and_set_all_records_received_for_phase.assert_not_called()
        manager._handle_all_records_received.assert_not_awaited()

        manager._credits_complete_received = True
        await manager._on_records(_metric_records_message())

        manager._records_tracker.check_and_set_all_records_received_for_phase.assert_called_once_with(
            CreditPhase.PROFILING
        )
        manager._handle_all_records_received.assert_awaited_once_with(
            CreditPhase.PROFILING
        )


class TestRecordsManagerAnalyzerMetrics:
    """Pin the invariant that `completed` counts request-derived records only,
    and that analyzer-injected metrics are merged after the snapshot."""

    @pytest.mark.asyncio
    async def test_completed_excludes_analyzer_metrics(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)

        manager.debug = MagicMock()
        manager.info = MagicMock()
        manager.error = MagicMock()
        manager.exception = MagicMock()
        manager.service_id = "records-manager-test"
        manager._latest_branch_stats = None
        manager.publish = AsyncMock()
        manager._skipped_context_overflow_counts_by_phase = {
            CreditPhase.WARMUP: 0,
            CreditPhase.PROFILING: 0,
        }

        manager.run = MagicMock()
        manager.run.cfg.gpu_telemetry_disabled = True
        manager.run.cfg.server_metrics_disabled = True
        manager.run.cfg.network_latency.enabled = False

        request_records = [
            MetricResult(tag="request_latency", header="h", unit="ms", avg=1.0),
            MetricResult(tag="output_token_count", header="h", unit="tokens", avg=2.0),
        ]
        metric_accumulator = MagicMock()
        metric_accumulator.summarize = AsyncMock(
            return_value=AccumulatorMetricsSummary(
                results={r.tag: r for r in request_records},
            )
        )
        manager._accumulators = {AccumulatorType.METRIC_RESULTS: metric_accumulator}
        manager._metric_record_accumulators = [metric_accumulator]
        manager._stream_exporters = {}
        manager._gpu_telemetry_accumulator = None
        manager._server_metrics_accumulator = None

        # An analyzer contributes derived aggregates that must NOT inflate
        # `completed` (which counts request-derived records only).
        analyzer_metrics = [
            MetricResult(tag="total_gpu_power", header="h", unit="W", avg=200.0),
            MetricResult(tag="total_gpu_energy", header="h", unit="J", avg=1000.0),
            MetricResult(
                tag="output_tokens_per_joule", header="h", unit="tokens/J", avg=0.002
            ),
        ]
        stub_analyzer = MagicMock()
        stub_analyzer.analyze = AsyncMock(return_value=analyzer_metrics)
        manager._analyzers = [
            LoadedAnalyzer(
                analyzer=stub_analyzer,
                required_accumulators=[],
                required_summaries=[],
            )
        ]
        manager._run_analyzers = RecordsManager._run_analyzers.__get__(manager)

        manager._records_tracker = MagicMock()
        manager._records_tracker.create_stats_for_phase.return_value = MagicMock(
            start_ns=1_000_000_000,
            requests_end_ns=2_000_000_000,
            success_records=2,
            error_records=0,
        )
        manager._error_tracker = MagicMock()
        manager._error_tracker.get_error_summary_for_phase.return_value = []

        manager._process_results_lock = asyncio.Lock()
        manager._processed_results = {}

        result = await manager._process_results(CreditPhase.PROFILING, cancelled=False)

        assert result.results.completed == len(request_records)
        assert len(result.results.records) == len(request_records) + len(
            analyzer_metrics
        )
        assert {r.tag for r in result.results.records} == {
            "request_latency",
            "output_token_count",
            "total_gpu_power",
            "total_gpu_energy",
            "output_tokens_per_joule",
        }
        stub_analyzer.analyze.assert_awaited_once()


class TestMidRunCacheReportingHint:
    """MetricsAccumulator warns once when usage lacks prompt-cache read tokens."""

    def _accumulator(self) -> MetricsAccumulator:
        accumulator = MetricsAccumulator.__new__(MetricsAccumulator)
        accumulator.warning = MagicMock()
        accumulator._warned_missing_cache_reporting = False
        return accumulator

    def test_warns_once_on_first_qualifying_record(self) -> None:
        accumulator = self._accumulator()
        record_data = SimpleNamespace(metrics={"usage_prompt_tokens": 1024})
        accumulator._maybe_hint_missing_cache_reporting(record_data)
        accumulator._maybe_hint_missing_cache_reporting(record_data)
        accumulator.warning.assert_called_once_with(CACHE_REPORTING_HINT)

    def test_no_warning_when_cache_reported(self) -> None:
        accumulator = self._accumulator()
        record_data = SimpleNamespace(
            metrics={"usage_prompt_tokens": 1024, "usage_prompt_cache_read_tokens": 0}
        )
        accumulator._maybe_hint_missing_cache_reporting(record_data)
        accumulator.warning.assert_not_called()

    def test_no_warning_when_usage_absent(self) -> None:
        accumulator = self._accumulator()
        record_data = SimpleNamespace(metrics={"output_sequence_length": 32})
        accumulator._maybe_hint_missing_cache_reporting(record_data)
        accumulator.warning.assert_not_called()


class TestRealtimeUpdateGate:
    def _manager(self) -> RecordsManager:
        manager = RecordsManager.__new__(RecordsManager)
        manager._previous_realtime_records = None
        manager._previous_realtime_server_snapshot = None
        return manager

    def test_first_tick_is_an_update(self) -> None:
        m = self._manager()
        assert m._has_realtime_update(0, {}) is True

    def test_record_count_change_triggers_update(self) -> None:
        m = self._manager()
        m._previous_realtime_records = 10
        m._previous_realtime_server_snapshot = {"kv_cache_usage_pct": 50.0}
        assert m._has_realtime_update(11, {"kv_cache_usage_pct": 50.0}) is True

    def test_server_metric_change_triggers_update_even_with_static_records(
        self,
    ) -> None:
        m = self._manager()
        m._previous_realtime_records = 10
        m._previous_realtime_server_snapshot = {"kv_cache_usage_pct": 50.0}
        assert m._has_realtime_update(10, {"kv_cache_usage_pct": 72.0}) is True

    def test_no_change_skips_update(self) -> None:
        m = self._manager()
        m._previous_realtime_records = 10
        m._previous_realtime_server_snapshot = {"kv_cache_usage_pct": 50.0}
        assert m._has_realtime_update(10, {"kv_cache_usage_pct": 50.0}) is False


class _DatasetAwareHandler:
    def __init__(self) -> None:
        self.metadata = None

    def on_dataset_configured(self, metadata) -> None:
        self.metadata = metadata


class TestRecordsManagerDatasetConfiguredBarrier:
    @pytest.mark.asyncio
    async def test_on_dataset_configured_sets_event_and_notifies_handlers(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._dataset_configured_event = asyncio.Event()
        acc = _DatasetAwareHandler()
        exp = _DatasetAwareHandler()
        manager._accumulators = {AccumulatorType.METRIC_RESULTS: acc}
        manager._stream_exporters = {MagicMock(): exp}
        message = MagicMock()
        message.metadata = {"task": "accuracy"}

        await manager._on_dataset_configured(message)

        assert manager._dataset_configured_event.is_set()
        assert acc.metadata == message.metadata
        assert exp.metadata == message.metadata

    @pytest.mark.asyncio
    async def test_on_metric_records_waits_for_dataset_configured(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._dataset_configured_event = asyncio.Event()
        manager.is_enabled_for = MagicMock(return_value=False)
        manager._records_tracker = MagicMock()
        manager._error_tracker = MagicMock()
        manager._complete_credit_phases = set()
        manager._dispatch_record = AsyncMock(
            side_effect=RuntimeError("REACHED_PROCESSING")
        )
        manager._warned_missing_cache_reporting = False
        message = _metric_records_message()

        task = asyncio.create_task(manager._on_records(message))
        for _ in range(3):
            await asyncio.sleep(0)

        assert not task.done()
        manager._dispatch_record.assert_not_called()

        manager._dataset_configured_event.set()
        with pytest.raises(RuntimeError, match="REACHED_PROCESSING"):
            await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_on_metric_records_fails_run_on_config_timeout(
        self, monkeypatch
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager.service_id = "rm-test"
        manager._dataset_configured_event = asyncio.Event()
        manager.is_enabled_for = MagicMock(return_value=False)
        manager.publish = AsyncMock()
        manager._kill = AsyncMock()
        manager._dispatch_record = AsyncMock()
        message = _metric_records_message()

        async def _raise_timeout(coro, *args, **kwargs):
            coro.close()
            raise TimeoutError

        monkeypatch.setattr(
            "aiperf.records.dataset_gate.asyncio.wait_for", _raise_timeout
        )

        await manager._on_records(message)

        manager._kill.assert_awaited_once()
        published = manager.publish.await_args.args[0]
        assert isinstance(published, BaseServiceErrorMessage)
        # ... and the record is not processed.
        manager._dispatch_record.assert_not_called()


class TestProcessServerMetricsResults:
    """Warmup-bound plumbing for the server metrics export path."""

    def _manager_with_accumulator(self) -> RecordsManager:
        manager = _create_manager_for_timing_dispatch()
        manager._server_metrics_state = MagicMock()
        manager._server_metrics_state.error_counts = {}
        return manager

    @pytest.mark.asyncio
    async def test_export_receives_profiling_and_warmup_bounds(self) -> None:
        manager = self._manager_with_accumulator()
        profiling_stats = MagicMock(
            start_ns=10_000_000_000, requests_end_ns=20_000_000_000
        )
        warmup_stats = MagicMock(start_ns=1_000_000_000, requests_end_ns=5_000_000_000)
        manager._records_tracker.create_stats_for_phase.side_effect = (
            lambda phase: profiling_stats
            if phase == CreditPhase.PROFILING
            else warmup_stats
        )
        exported = ServerMetricsResults(
            start_ns=10_000_000_000,
            end_ns=20_000_000_000,
            warmup_start_ns=1_000_000_000,
            warmup_end_ns=5_000_000_000,
        )
        accumulator = MagicMock()
        accumulator.export_results = AsyncMock(return_value=exported)
        manager._server_metrics_accumulator = accumulator

        result = await manager._process_server_metrics_results()

        ctx = accumulator.export_results.await_args.args[0]
        assert ctx.start_ns == 10_000_000_000
        assert ctx.end_ns == 20_000_000_000
        assert ctx.warmup_start_ns == 1_000_000_000
        assert ctx.warmup_end_ns == 5_000_000_000
        assert result.results is exported

    @pytest.mark.asyncio
    async def test_no_accumulator_returns_empty_results(self) -> None:
        manager = self._manager_with_accumulator()
        manager._server_metrics_accumulator = None

        result = await manager._process_server_metrics_results()

        assert result.results is None
