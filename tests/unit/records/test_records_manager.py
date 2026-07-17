# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.common.messages.inference_messages import (
    MetricRecordsData,
    RecordsMessage,
)
from aiperf.common.messages.telemetry_messages import TelemetryRecordsMessage
from aiperf.common.models import (
    BranchStats,
    CreditPhaseStats,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
    TelemetryMetrics,
    TelemetryRecord,
    TimesliceResult,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.record_models import MetricRecordMetadata
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
from aiperf.plugin.enums import AccumulatorType, TimingMode
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_manager import ErrorTrackingState, RecordsManager
from aiperf.records.records_manager_processing import LoadedAnalyzer
from aiperf.records.records_tracker import RecordsTracker
from aiperf.timing.config import CreditPhaseConfig

# Helper functions


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
        dcgm_url="http://localhost:9400/metrics",
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
            dcgm_url="http://localhost:9400/metrics",
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
                dcgm_url="http://localhost:9400/metrics",
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
                dcgm_url="http://localhost:9400/metrics",
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
    async def test_successful_metric_dispatch_records_no_phase_error(self) -> None:
        manager = self._make_manager()
        manager._dispatch_record = AsyncMock(return_value=[])

        await manager._on_records(self._records_message())

        assert (
            manager._error_tracker.get_error_summary_for_phase(CreditPhase.PROFILING)
            == []
        )


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
        manager._dispatch_record.assert_not_called()


# ---------------------------------------------------------------------------
# context-overflow record metric-exclusion (graph-IR)
# ---------------------------------------------------------------------------


def _overflow_skip_message(
    overflow_count: int = 1,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> RecordsMessage:
    metadata = MetricRecordMetadata(
        session_num=3,
        conversation_id="conv-overflow",
        turn_index=2,
        request_start_ns=1_000_000_000,
        request_end_ns=1_010_000_000,
        worker_id="worker-overflow",
        record_processor_id="rp-1",
        benchmark_phase=phase,
        context_overflow_skip=True,
    )
    return RecordsMessage(
        service_id="rp-1",
        metadata=metadata,
        records=[
            MetricRecordsData(
                metadata=metadata,
                metrics={
                    "context_overflow_count": overflow_count,
                    "request_latency": 9_999_000_000,
                },
            )
        ],
        error=ErrorDetails(message="context_length_exceeded", code=400),
    )


def _create_real_tracker_manager() -> RecordsManager:
    manager = RecordsManager.__new__(RecordsManager)
    manager._records_tracker = RecordsTracker()
    manager._error_tracker = ErrorTracker()
    manager._skipped_context_overflow_count = 0
    manager._complete_credit_phases = set()
    manager._dataset_configured_event = asyncio.Event()
    manager._dataset_configured_event.set()
    manager.info = MagicMock()
    manager.debug = MagicMock()
    manager.trace = MagicMock()
    manager.warning = MagicMock()
    manager.is_enabled_for = MagicMock(return_value=False)
    manager._handle_all_records_received = AsyncMock()
    forwarded: list[MetricRecordsData] = []

    async def _capture(record, **kwargs) -> list[BaseException]:
        forwarded.append(record)
        return []

    manager._dispatch_record = AsyncMock(side_effect=_capture)
    manager._forwarded = forwarded  # type: ignore[attr-defined]
    return manager


class TestRecordsManagerOverflowExclusion:
    """Graph-IR context-overflow records skip perf accumulation + the error
    tracker, but still advance the success counter and forward ONLY the
    context_overflow_count metric (submission-rate gate)."""

    @pytest.mark.asyncio
    async def test_overflow_skip_counts_success_not_error(self) -> None:
        manager = _create_real_tracker_manager()
        await manager._on_records(_overflow_skip_message())

        tracker = manager._records_tracker._get_phase_tracker(CreditPhase.PROFILING)
        assert tracker._success_records == 1
        assert tracker._error_records == 0
        assert manager._skipped_context_overflow_count == 1
        summary = manager._error_tracker.get_error_summary_for_phase(
            CreditPhase.PROFILING
        )
        assert sum(e.count for e in summary) == 0

    @pytest.mark.asyncio
    async def test_overflow_skip_forwards_only_overflow_count_metric(self) -> None:
        manager = _create_real_tracker_manager()
        await manager._on_records(_overflow_skip_message(overflow_count=1))

        forwarded = manager._forwarded  # type: ignore[attr-defined]
        assert len(forwarded) == 1
        trimmed = forwarded[0]
        assert trimmed.metrics == {"context_overflow_count": 1}
        assert "request_latency" not in trimmed.metrics
        assert trimmed.error is None
        assert trimmed.metadata.context_overflow_skip is True
        assert trimmed.valid is True

    @pytest.mark.asyncio
    async def test_non_overflow_record_unchanged(self) -> None:
        manager = _create_real_tracker_manager()
        await manager._on_records(_metric_records_message())

        forwarded = manager._forwarded  # type: ignore[attr-defined]
        assert len(forwarded) == 1
        assert forwarded[0].metrics == {"request_latency": 250_000_000}
        assert manager._skipped_context_overflow_count == 0
        tracker = manager._records_tracker._get_phase_tracker(CreditPhase.PROFILING)
        assert tracker._success_records == 1
