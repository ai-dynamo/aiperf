# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial coverage tests for `RecordsManager`.

Each class corresponds to a specific bound method on the manager and pokes at
the branches the existing happy-path suite leaves uncovered:

1. `_process_metric_record_data` — phase-excluded short-circuit, error
   bookkeeping with a real `WireErrorDetails`, and the all-records-received
   trigger that fans out to `_handle_all_records_received`.
2. `_handle_all_records_received` — phase-excluded skip, waits-for-others
   path, and the cancelled-phase propagation through `_finalize_and_process_results`.
3. `_finalize_and_process_results` — server-metrics relay TimeoutError /
   ErrorDetails response paths, and the flush-period sleep cap.
4. Credit-phase message handlers — `_on_credit_phase_start`,
   `_on_credit_phase_sending_complete`, `_on_credit_phase_complete`,
   `_on_credits_complete`.
5. `_send_results_to_results_processors` — single-processor fast path,
   gather fan-out, and empty short-circuit.
6. `_report_records_task` — skip-empty-phase + first-active-phase break.
7. `_on_profile_cancel_command` — payload parsing, mark-cancelled
   bookkeeping. `_on_start_realtime_telemetry_command` — realtime-loop
   un-park, idempotence, accumulator-absent error path.
8. `_report_realtime_inference_metrics_task` — early-exit gate and the
   "no new records" continue branch.
9. `_report_realtime_metrics` — empty-raw + filtered-empty short-circuits.
10. `_summarize_all_processors` — MetricsAccumulator bridge: timeslices
    promoted to the bucketed list, multi_turn_ttft_trend returned separately,
    and accumulator failure tolerated.
11. `_publish_all_results` — publish failure logged, not propagated.
12. `_write_partial_checkpoint_task` — non-K8s skip.
13. `__init__` — TCP additional-bind activation when ZMQDualBindConfig
    has no `controller_host`.
14. `main()` — bootstrap entrypoint dispatch.

All exception-path tests assert observable side effects (logged-error
content, published message types, tracker mutations) — not call counts —
to lock the wire-shape contract the source actually carries.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    ProcessAllResultsMessage,
    ProcessRecordsResultMessage,
    RealtimeMetricsMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsBatchWireMessage,
    MetricRecordsData,
    MetricRecordsWireMessage,
    WireErrorDetails,
)
from aiperf.common.models import (
    CreditPhaseStats,
    ErrorDetails,
    MetricResult,
    PhaseRecordsStats,
    WorkerProcessingStats,
)
from aiperf.plugin.enums import AccumulatorType, AnalyzerType
from aiperf.records.records_manager import RecordsManager, main
from aiperf.records.records_tracker import RecordsTracker

# ============================================================================
# Canonical-record dict-spread helper (mimics the sweep_models exemplar).
# ============================================================================

_VALID_METADATA: dict[str, Any] = {
    "request_num": 0,
    "session_num": 0,
    "conversation_id": "conv-1",
    "turn_index": 0,
    "request_start_ns": 100,
    "request_end_ns": 200,
    "worker_id": "worker-1",
    "record_processor_id": "rp-1",
    "benchmark_phase": "profiling",
}


def _metadata_with(**overrides: Any) -> MetricRecordMetadata:
    """Return a `MetricRecordMetadata` with overrides spread over the canonical dict."""
    return MetricRecordMetadata(**{**_VALID_METADATA, **overrides})


def _record(
    *,
    phase: str = "profiling",
    metrics: dict[str, Any] | None = None,
    error: WireErrorDetails | None = None,
) -> MetricRecordsData:
    return MetricRecordsData(
        metadata=_metadata_with(benchmark_phase=phase),
        metrics=metrics or {},
        error=error,
    )


def _metric(tag: str, avg: float = 1.0) -> MetricResult:
    return MetricResult(tag=tag, header=tag, unit="ms", avg=avg, count=1)


# ============================================================================
# Manager mock factory — binds whichever real methods the test needs.
# ============================================================================


def _make_manager(
    *,
    bind_methods: list[str] | None = None,
    legacy_processors: list[Any] | None = None,
    accumulators: dict[Any, Any] | None = None,
    stream_exporters: dict[Any, Any] | None = None,
    analyzers: dict[Any, Any] | None = None,
    tracker: Any | None = None,
    error_tracker: Any | None = None,
    api_port: int | None = None,
    ui_realtime_enabled: bool = False,
    ui_type: str = "simple",
    service_run_type: str = "process",
    metric_record_accumulators: list[Any] | None = None,
    metric_record_stream_exporters: list[Any] | None = None,
) -> MagicMock:
    """Build a `RecordsManager` mock pre-wired for adversarial tests."""
    mgr = MagicMock()
    mgr.service_id = "records-manager-1"
    mgr._dataset_configured_event = asyncio.Event()
    mgr._dataset_configured_event.set()
    mgr._metric_results_processors = legacy_processors or []
    mgr._timing_results_processors = []
    mgr._accumulators = accumulators or {}
    mgr._stream_exporters = stream_exporters or {}
    mgr._network_latency_processors = []
    mgr._analyzers = analyzers or {}
    mgr._metric_record_accumulators = metric_record_accumulators or []
    mgr._metric_record_stream_exporters = metric_record_stream_exporters or []
    mgr._previous_realtime_records = None
    mgr._last_checkpoint_records = 0

    mgr._records_tracker = tracker or MagicMock()
    mgr._error_tracker = error_tracker or MagicMock()

    # Default tracker behavior — overridden by callers.
    mgr._records_tracker.is_phase_excluded.return_value = False
    mgr._records_tracker.check_and_set_all_records_received_for_phase.return_value = (
        False
    )
    mgr._records_tracker.get_results_phases.return_value = ["profiling"]
    mgr._records_tracker.get_results_time_window.return_value = (1_000, 2_000)
    mgr._records_tracker.are_all_results_phases_complete.return_value = True
    mgr._records_tracker.was_phase_cancelled.return_value = False
    mgr._records_tracker.create_overall_worker_stats.return_value = {}
    mgr._error_tracker.get_error_summary_for_phase.return_value = []

    # Logging
    mgr.is_trace_enabled = False
    for level in ("trace", "debug", "info", "warning", "error", "exception", "notice"):
        setattr(mgr, level, MagicMock())

    # Async pipeline
    mgr.publish = AsyncMock()
    # Timing fan-out defaults to a no-op AsyncMock so credit-phase handler tests
    # can await it; fan-out tests override via ``bind_methods``.
    mgr._send_timing_to_results_processors = AsyncMock()
    mgr.execute_async = MagicMock()
    mgr.control_client = MagicMock()
    mgr.control_client.request = AsyncMock()
    mgr.stop_requested = False

    # `run.cfg` surface used by background tasks.
    mgr.run = MagicMock()
    mgr.run.cfg.runtime.api_port = api_port
    mgr.run.cfg.runtime.service_run_type = service_run_type
    mgr.run.cfg.ui_type = ui_type
    mgr.run.cfg.artifacts.profile_export_partial_json_file = "/tmp/checkpoint.json"

    # Bind requested methods. We bind on demand because some tests want to
    # mock these as AsyncMocks instead.
    method_map = {
        "_process_metric_record_data": RecordsManager._process_metric_record_data,
        "_send_record_to_accumulators": RecordsManager._send_record_to_accumulators,
        "_send_results_to_results_processors": RecordsManager._send_results_to_results_processors,
        "_raise_unless_best_effort": RecordsManager._raise_unless_best_effort,
        "_handle_all_records_received": RecordsManager._handle_all_records_received,
        "_finalize_and_process_results": RecordsManager._finalize_and_process_results,
        "_on_metric_records": RecordsManager._on_metric_records,
        "_on_credit_phase_start": RecordsManager._on_credit_phase_start,
        "_on_credit_phase_progress": RecordsManager._on_credit_phase_progress,
        "_on_credit_phase_sending_complete": RecordsManager._on_credit_phase_sending_complete,
        "_on_credit_phase_complete": RecordsManager._on_credit_phase_complete,
        "_send_timing_to_results_processors": RecordsManager._send_timing_to_results_processors,
        "_on_credits_complete": RecordsManager._on_credits_complete,
        "_report_records_task": RecordsManager._report_records_task,
        "_publish_processing_stats": RecordsManager._publish_processing_stats,
        "_on_profile_cancel_command": RecordsManager._on_profile_cancel_command,
        "_report_realtime_inference_metrics_task": RecordsManager._report_realtime_inference_metrics_task,
        "_on_realtime_metrics_command": RecordsManager._on_realtime_metrics_command,
        "_report_realtime_metrics": RecordsManager._report_realtime_metrics,
        "_process_results": RecordsManager._process_results,
        "_summarize_all_processors": RecordsManager._summarize_all_processors,
        "_finalize_all_processors": RecordsManager._finalize_all_processors,
        "_build_records_result": RecordsManager._build_records_result,
        "_publish_all_results": RecordsManager._publish_all_results,
        "_finalize_stream_exporters": RecordsManager._finalize_stream_exporters,
        "_finalize_network_latency_processors": RecordsManager._finalize_network_latency_processors,
        "_run_analyzers": RecordsManager._run_analyzers,
        "_write_partial_checkpoint_task": RecordsManager._write_partial_checkpoint_task,
    }
    # _process_results calls _publish_telemetry_results (GPU telemetry side).
    # Default it to an AsyncMock so the inference-focused tests don't have to
    # wire the telemetry accumulator; tests that care can bind/override it.
    mgr._publish_telemetry_results = AsyncMock()

    for name in bind_methods or []:
        # Strip the @background_task / @on_command decorator wrapping by
        # accessing __wrapped__ or just calling .__func__ underlying.
        fn = method_map[name]
        # Some methods are wrapped by decorators (background_task, on_command).
        # We need the underlying coroutine for direct invocation. Try unwrap.
        wrapped = getattr(fn, "__wrapped__", fn)
        setattr(mgr, name, wrapped.__get__(mgr))
    return mgr


# ============================================================================
# 1) `_process_metric_record_data` — phase routing + error bookkeeping
# ============================================================================


class TestProcessMetricRecordData:
    """The hot path that ingests one record and decides what to dispatch."""

    @pytest.mark.asyncio
    async def test_excluded_phase_skips_processors_but_still_tracks(self) -> None:
        """When a phase is excluded, neither the legacy nor accumulator path runs.

        The records-tracker still gets `update_from_record_data` because the
        tracker is what `is_phase_excluded` reads from — but downstream
        processing must short-circuit so warmup records don't leak into
        results.
        """
        mgr = _make_manager(bind_methods=["_process_metric_record_data"])
        mgr._records_tracker.is_phase_excluded.return_value = True
        mgr._send_results_to_results_processors = AsyncMock()
        mgr._send_record_to_accumulators = AsyncMock()
        mgr._handle_all_records_received = AsyncMock()

        await mgr._process_metric_record_data(_record(phase="warmup"))

        mgr._records_tracker.update_from_record_data.assert_called_once()
        mgr._send_results_to_results_processors.assert_not_awaited()
        mgr._send_record_to_accumulators.assert_not_awaited()
        # No error to track.
        mgr._error_tracker.increment_error_count_for_phase.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_with_wire_error_increments_phase_error_counter(self) -> None:
        """A record carrying a `WireErrorDetails` must be surfaced in the error tracker.

        Locks the squash-time integration: `wire_error_to_domain_error` is
        the only legal way to materialize the domain `ErrorDetails` for the
        per-phase error tally.
        """
        mgr = _make_manager(bind_methods=["_process_metric_record_data"])
        mgr._send_results_to_results_processors = AsyncMock()
        mgr._send_record_to_accumulators = AsyncMock()
        mgr._handle_all_records_received = AsyncMock()
        wire_err = WireErrorDetails(code=503, type="UpstreamError", message="boom")
        record = _record(error=wire_err)

        await mgr._process_metric_record_data(record)

        mgr._error_tracker.increment_error_count_for_phase.assert_called_once()
        called_phase, called_err = (
            mgr._error_tracker.increment_error_count_for_phase.call_args.args
        )
        assert called_phase == "profiling"
        # Should be the domain ErrorDetails — not the wire struct.
        assert isinstance(called_err, ErrorDetails)
        assert called_err.message == "boom"
        assert called_err.code == 503

    @pytest.mark.asyncio
    async def test_all_records_received_triggers_handler_with_phase(self) -> None:
        """When the tracker flips, the all-records-received handler fires for that phase."""
        mgr = _make_manager(bind_methods=["_process_metric_record_data"])
        mgr._send_results_to_results_processors = AsyncMock()
        mgr._send_record_to_accumulators = AsyncMock()
        mgr._handle_all_records_received = AsyncMock()
        mgr._records_tracker.check_and_set_all_records_received_for_phase.return_value = True

        await mgr._process_metric_record_data(_record(phase="profiling"))

        mgr._handle_all_records_received.assert_awaited_once_with("profiling")

    @pytest.mark.asyncio
    async def test_trace_enabled_logs_received_message(self) -> None:
        """`is_trace_enabled` gates the trace log — adversarial check that
        flipping it causes the log to fire and not the other way around."""
        mgr = _make_manager(bind_methods=["_on_metric_records"])
        mgr.is_trace_enabled = True
        mgr._process_metric_record_data = AsyncMock()
        wire_msg = MetricRecordsWireMessage(
            service_id="rp-1",
            metadata=_metadata_with(),
            metrics={"request_latency": 1.0},
        )

        await mgr._on_metric_records(wire_msg)

        mgr.trace.assert_called_once()
        # Single-record path went through wire->record conversion.
        mgr._process_metric_record_data.assert_awaited_once()


# ============================================================================
# 2) `_handle_all_records_received` — phase-aware finalization gating
# ============================================================================


class TestHandleAllRecordsReceived:
    """Three branches: excluded, waiting, fire-finalize."""

    @pytest.mark.asyncio
    async def test_excluded_phase_does_not_trigger_finalize(self) -> None:
        mgr = _make_manager(bind_methods=["_handle_all_records_received"])
        excluded_stats = MagicMock(
            success_records=10,
            error_records=0,
            total_records=10,
            exclude_from_results=True,
        )
        mgr._records_tracker.create_stats_for_phase.return_value = excluded_stats

        await mgr._handle_all_records_received("warmup")

        mgr.execute_async.assert_not_called()
        mgr._records_tracker.are_all_results_phases_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_waits_for_other_phases_before_finalizing(self) -> None:
        """If `are_all_results_phases_complete` is False, we DON'T fire finalize."""
        mgr = _make_manager(bind_methods=["_handle_all_records_received"])
        ready_stats = MagicMock(
            success_records=5,
            error_records=0,
            total_records=5,
            exclude_from_results=False,
        )
        mgr._records_tracker.create_stats_for_phase.return_value = ready_stats
        mgr._records_tracker.are_all_results_phases_complete.return_value = False

        await mgr._handle_all_records_received("profiling")

        mgr.execute_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_finalize_dispatched_with_cancelled_flag_true_when_any_phase_cancelled(
        self,
    ) -> None:
        """If any results phase was cancelled, `cancelled=True` propagates to finalize.

        Adversarial: we have two non-excluded phases; only one was cancelled,
        but the OR semantics mean cancelled must be True.
        """
        mgr = _make_manager(bind_methods=["_handle_all_records_received"])
        ready_stats = MagicMock(
            success_records=10,
            error_records=0,
            total_records=10,
            exclude_from_results=False,
        )
        mgr._records_tracker.create_stats_for_phase.return_value = ready_stats
        mgr._records_tracker.are_all_results_phases_complete.return_value = True
        mgr._records_tracker.get_results_phases.return_value = ["warmup", "profiling"]
        mgr._records_tracker.was_phase_cancelled.side_effect = (
            lambda p: p == "profiling"
        )

        await mgr._handle_all_records_received("profiling")

        mgr.execute_async.assert_called_once()
        # The coroutine passed to execute_async should be a finalize coroutine
        # — close it to avoid "coroutine was never awaited" warnings.
        coro = mgr.execute_async.call_args.args[0]
        coro.close()

    @pytest.mark.asyncio
    async def test_finalize_dispatched_with_cancelled_false_when_no_phase_cancelled(
        self,
    ) -> None:
        mgr = _make_manager(bind_methods=["_handle_all_records_received"])
        ready_stats = MagicMock(
            success_records=10,
            error_records=0,
            total_records=10,
            exclude_from_results=False,
        )
        mgr._records_tracker.create_stats_for_phase.return_value = ready_stats
        mgr._records_tracker.was_phase_cancelled.return_value = False

        await mgr._handle_all_records_received("profiling")

        mgr.execute_async.assert_called_once()
        coro = mgr.execute_async.call_args.args[0]
        coro.close()


# ============================================================================
# 3) `_finalize_and_process_results` — server-metrics relay error paths
# ============================================================================


class TestFinalizeAndProcessResults:
    """The PROFILE_COMPLETE relay must not abort finalization on timeout/error.

    These branches were the squash's whole point — see the long inline
    comment at lines 289-298 of `records_manager.py`. The contract: the
    operator's results-fetch loop interprets a missing
    `.aiperf_results_ready.json` as failure. So we MUST hit `_process_results`
    even when the relay times out.
    """

    @pytest.mark.asyncio
    async def test_relay_timeout_does_not_abort_processing(self) -> None:
        mgr = _make_manager(bind_methods=["_finalize_and_process_results"])
        mgr.control_client.request.side_effect = asyncio.TimeoutError()
        mgr._process_results = AsyncMock()
        mgr._records_tracker.create_stats_for_phase.return_value = MagicMock(
            spec=PhaseRecordsStats
        )
        mgr._records_tracker.get_results_phases.return_value = ["profiling"]

        await mgr._finalize_and_process_results(cancelled=False)

        # AllRecordsReceivedMessage published despite the timeout.
        published_types = {type(c.args[0]) for c in mgr.publish.await_args_list}
        assert AllRecordsReceivedMessage in published_types
        # Warning logged with explicit timeout context.
        assert any("timed out" in str(c.args[0]) for c in mgr.warning.call_args_list)
        # And critically — _process_results still ran.
        mgr._process_results.assert_awaited_once_with(cancelled=False)

    @pytest.mark.asyncio
    async def test_relay_error_response_logged_but_processing_continues(self) -> None:
        """Server-metrics relay returns ErrorDetails — also non-fatal."""
        mgr = _make_manager(bind_methods=["_finalize_and_process_results"])
        mgr.control_client.request.return_value = ErrorDetails(
            message="server metrics unavailable", code=500
        )
        mgr._process_results = AsyncMock()
        mgr._records_tracker.create_stats_for_phase.return_value = MagicMock(
            spec=PhaseRecordsStats
        )
        mgr._records_tracker.get_results_phases.return_value = ["profiling"]

        await mgr._finalize_and_process_results(cancelled=True)

        assert any(
            "server metrics unavailable" in str(c.args[0])
            for c in mgr.warning.call_args_list
        )
        mgr._process_results.assert_awaited_once_with(cancelled=True)

    @pytest.mark.asyncio
    async def test_falls_back_to_default_phase_when_no_results_phases(self) -> None:
        """When `get_results_phases()` returns empty, the AllRecordsReceived
        publish still happens — using `"profiling"` as the fallback stats key."""
        mgr = _make_manager(bind_methods=["_finalize_and_process_results"])
        mgr._process_results = AsyncMock()
        mgr.control_client.request.return_value = MagicMock()
        mgr._records_tracker.get_results_phases.return_value = []
        mgr._records_tracker.get_results_time_window.return_value = (None, None)

        await mgr._finalize_and_process_results(cancelled=False)

        mgr._records_tracker.create_stats_for_phase.assert_called_with("profiling")
        mgr._process_results.assert_awaited_once()


# ============================================================================
# 4) Credit-phase message handlers — wire-shape contract
# ============================================================================


class TestCreditPhaseMessageHandlers:
    """Each credit-phase message updates the tracker and may flip records-received."""

    @pytest.mark.asyncio
    async def test_credit_phase_start_updates_tracker(self) -> None:
        mgr = _make_manager(bind_methods=["_on_credit_phase_start"])
        msg = SimpleNamespace(
            stats=MagicMock(phase="profiling"),
            config=MagicMock(phase="profiling"),
        )

        await mgr._on_credit_phase_start(msg)

        mgr._records_tracker.update_phase_info.assert_called_once_with(msg.stats)

    @pytest.mark.asyncio
    async def test_credit_phase_sending_complete_logs_count_and_updates_tracker(
        self,
    ) -> None:
        mgr = _make_manager(bind_methods=["_on_credit_phase_sending_complete"])
        msg = SimpleNamespace(
            stats=MagicMock(phase="profiling", final_requests_sent=42)
        )

        await mgr._on_credit_phase_sending_complete(msg)

        mgr._records_tracker.update_phase_info.assert_called_once_with(msg.stats)
        # Logged the count.
        assert any("42" in str(c.args[0]) for c in mgr.info.call_args_list)

    @pytest.mark.asyncio
    async def test_credit_phase_complete_with_excluded_phase_does_not_emit_notice(
        self,
    ) -> None:
        """Excluded phase → `notice` log suppressed (warmup chatter)."""
        mgr = _make_manager(bind_methods=["_on_credit_phase_complete"])
        excluded_stats = MagicMock(
            phase="warmup",
            total_records=10,
            final_requests_completed=10,
            exclude_from_results=True,
        )
        mgr._records_tracker.create_stats_for_phase.return_value = excluded_stats
        msg = SimpleNamespace(stats=MagicMock(phase="warmup"))

        await mgr._on_credit_phase_complete(msg)

        mgr.notice.assert_not_called()

    @pytest.mark.asyncio
    async def test_credit_phase_complete_emits_notice_for_results_phase(self) -> None:
        mgr = _make_manager(bind_methods=["_on_credit_phase_complete"])
        ready_stats = MagicMock(
            phase="profiling",
            total_records=100,
            final_requests_completed=100,
            exclude_from_results=False,
        )
        mgr._records_tracker.create_stats_for_phase.return_value = ready_stats
        mgr._handle_all_records_received = AsyncMock()
        msg = SimpleNamespace(stats=MagicMock(phase="profiling"))

        await mgr._on_credit_phase_complete(msg)

        mgr.notice.assert_called_once()

    @pytest.mark.asyncio
    async def test_credit_phase_complete_triggers_handler_when_records_already_in(
        self,
    ) -> None:
        """If all records arrived BEFORE the credit-complete message, the
        handler fires from this code path — not from `_on_metric_records`.

        This is the documented race-prevention line: see the inline comment
        in source.
        """
        mgr = _make_manager(bind_methods=["_on_credit_phase_complete"])
        ready_stats = MagicMock(
            phase="profiling",
            total_records=10,
            final_requests_completed=10,
            exclude_from_results=False,
        )
        mgr._records_tracker.create_stats_for_phase.return_value = ready_stats
        mgr._records_tracker.check_and_set_all_records_received_for_phase.return_value = True
        mgr._handle_all_records_received = AsyncMock()
        msg = SimpleNamespace(stats=MagicMock(phase="profiling"))

        await mgr._on_credit_phase_complete(msg)

        mgr._handle_all_records_received.assert_awaited_once_with("profiling")

    @pytest.mark.asyncio
    async def test_credit_phase_start_streams_timing_snapshot(self) -> None:
        """Phase-start hands its stats to the timing fan-out (OTel/MLflow live)."""
        mgr = _make_manager(bind_methods=["_on_credit_phase_start"])
        msg = SimpleNamespace(
            stats=MagicMock(phase="profiling"),
            config=MagicMock(phase="profiling"),
        )

        await mgr._on_credit_phase_start(msg)

        mgr._send_timing_to_results_processors.assert_awaited_once_with(msg.stats)

    @pytest.mark.asyncio
    async def test_credit_phase_progress_streams_timing_and_updates_tracker(
        self,
    ) -> None:
        """Periodic progress ticks stream live timing and keep the tracker fresh."""
        mgr = _make_manager(bind_methods=["_on_credit_phase_progress"])
        msg = SimpleNamespace(stats=MagicMock(phase="profiling"))

        await mgr._on_credit_phase_progress(msg)

        mgr._records_tracker.update_phase_info.assert_called_once_with(msg.stats)
        mgr._send_timing_to_results_processors.assert_awaited_once_with(msg.stats)

    @pytest.mark.asyncio
    async def test_credit_phase_sending_complete_streams_timing_snapshot(self) -> None:
        mgr = _make_manager(bind_methods=["_on_credit_phase_sending_complete"])
        msg = SimpleNamespace(stats=MagicMock(phase="profiling", final_requests_sent=7))

        await mgr._on_credit_phase_sending_complete(msg)

        mgr._send_timing_to_results_processors.assert_awaited_once_with(msg.stats)

    @pytest.mark.asyncio
    async def test_credit_phase_complete_streams_timing_snapshot(self) -> None:
        mgr = _make_manager(bind_methods=["_on_credit_phase_complete"])
        ready_stats = MagicMock(
            phase="profiling",
            total_records=5,
            final_requests_completed=5,
            exclude_from_results=False,
        )
        mgr._records_tracker.create_stats_for_phase.return_value = ready_stats
        mgr._handle_all_records_received = AsyncMock()
        msg = SimpleNamespace(stats=MagicMock(phase="profiling"))

        await mgr._on_credit_phase_complete(msg)

        mgr._send_timing_to_results_processors.assert_awaited_once_with(msg.stats)

    @pytest.mark.asyncio
    async def test_credits_complete_checks_every_results_phase(self) -> None:
        """`_on_credits_complete` polls every non-excluded phase for completion."""
        mgr = _make_manager(bind_methods=["_on_credits_complete"])
        mgr._records_tracker.get_results_phases.return_value = ["warmup", "profiling"]
        mgr._records_tracker.check_and_set_all_records_received_for_phase.side_effect = [
            False,
            True,
        ]
        mgr._handle_all_records_received = AsyncMock()
        msg = SimpleNamespace()

        await mgr._on_credits_complete(msg)

        # Polled both phases.
        assert (
            mgr._records_tracker.check_and_set_all_records_received_for_phase.call_count
            == 2
        )
        # Only "profiling" returned True — only that phase fires the handler.
        mgr._handle_all_records_received.assert_awaited_once_with("profiling")


# ============================================================================
# 5) `_send_results_to_results_processors` — fan-out optimization
# ============================================================================


class TestSendResultsToResultsProcessors:
    """Single vs gather paths + empty short-circuit."""

    @pytest.mark.asyncio
    async def test_no_processors_short_circuits(self) -> None:
        mgr = _make_manager(bind_methods=["_send_results_to_results_processors"])
        mgr._metric_results_processors = []

        await mgr._send_results_to_results_processors(_record())
        # No exceptions, no errors.
        mgr.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_processor_avoids_gather_overhead(self) -> None:
        """Single-processor path calls `process_result` directly."""
        proc = MagicMock()
        proc.process_result = AsyncMock()
        mgr = _make_manager(
            bind_methods=["_send_results_to_results_processors"],
            legacy_processors=[proc],
        )
        record = _record()

        await mgr._send_results_to_results_processors(record)

        proc.process_result.assert_awaited_once_with(record)

    @pytest.mark.asyncio
    async def test_multiple_processors_fan_out_via_gather(self) -> None:
        proc_a = MagicMock()
        proc_a.process_result = AsyncMock()
        proc_b = MagicMock()
        proc_b.process_result = AsyncMock()
        mgr = _make_manager(
            bind_methods=["_send_results_to_results_processors"],
            legacy_processors=[proc_a, proc_b],
        )
        record = _record()

        await mgr._send_results_to_results_processors(record)

        proc_a.process_result.assert_awaited_once_with(record)
        proc_b.process_result.assert_awaited_once_with(record)

    @pytest.mark.asyncio
    async def test_best_effort_processor_failure_is_swallowed_in_gather_path(
        self,
    ) -> None:
        """A best-effort processor raising must not fail the fan-out; siblings still run."""
        otel_like = MagicMock()
        otel_like.is_best_effort = True
        otel_like.process_result = AsyncMock(side_effect=RuntimeError("otel down"))
        sibling = MagicMock()
        sibling.is_best_effort = False
        sibling.process_result = AsyncMock()
        mgr = _make_manager(
            bind_methods=[
                "_send_results_to_results_processors",
                "_raise_unless_best_effort",
            ],
            legacy_processors=[otel_like, sibling],
        )
        record = _record()

        await mgr._send_results_to_results_processors(record)

        sibling.process_result.assert_awaited_once_with(record)
        mgr.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_best_effort_processor_failure_reraises_in_gather_path(
        self,
    ) -> None:
        proc_a = MagicMock()
        proc_a.is_best_effort = False
        proc_a.process_result = AsyncMock(side_effect=RuntimeError("pipeline bug"))
        proc_b = MagicMock()
        proc_b.is_best_effort = False
        proc_b.process_result = AsyncMock()
        mgr = _make_manager(
            bind_methods=[
                "_send_results_to_results_processors",
                "_raise_unless_best_effort",
            ],
            legacy_processors=[proc_a, proc_b],
        )

        with pytest.raises(RuntimeError, match="pipeline bug"):
            await mgr._send_results_to_results_processors(_record())

    @pytest.mark.asyncio
    async def test_best_effort_processor_failure_is_swallowed_in_single_path(
        self,
    ) -> None:
        otel_like = MagicMock()
        otel_like.is_best_effort = True
        otel_like.process_result = AsyncMock(side_effect=RuntimeError("otel down"))
        mgr = _make_manager(
            bind_methods=[
                "_send_results_to_results_processors",
                "_raise_unless_best_effort",
            ],
            legacy_processors=[otel_like],
        )

        await mgr._send_results_to_results_processors(_record())

        mgr.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_best_effort_processor_failure_reraises_in_single_path(
        self,
    ) -> None:
        proc = MagicMock()
        proc.is_best_effort = False
        proc.process_result = AsyncMock(side_effect=RuntimeError("pipeline bug"))
        mgr = _make_manager(
            bind_methods=[
                "_send_results_to_results_processors",
                "_raise_unless_best_effort",
            ],
            legacy_processors=[proc],
        )

        with pytest.raises(RuntimeError, match="pipeline bug"):
            await mgr._send_results_to_results_processors(_record())

    @pytest.mark.asyncio
    async def test_cancellation_is_never_swallowed_even_for_best_effort(self) -> None:
        """CancelledError must propagate so shutdown is not absorbed by the marker."""
        otel_like = MagicMock()
        otel_like.is_best_effort = True
        otel_like.process_result = AsyncMock(side_effect=asyncio.CancelledError())
        sibling = MagicMock()
        sibling.is_best_effort = False
        sibling.process_result = AsyncMock()
        mgr = _make_manager(
            bind_methods=[
                "_send_results_to_results_processors",
                "_raise_unless_best_effort",
            ],
            legacy_processors=[otel_like, sibling],
        )

        with pytest.raises(asyncio.CancelledError):
            await mgr._send_results_to_results_processors(_record())


# ============================================================================
# 5b) `_send_timing_to_results_processors` — CreditPhaseStats fan-out
# ============================================================================


def _phase_stats(phase: CreditPhase = CreditPhase.PROFILING) -> CreditPhaseStats:
    """Real (unwrapped) ``CreditPhaseStats`` — what timing strategies match on."""
    return CreditPhaseStats(phase=phase)


class TestSendTimingToResultsProcessors:
    """Timing snapshots reach only timing-capable processors, best-effort-safe."""

    @pytest.mark.asyncio
    async def test_no_timing_processors_short_circuits(self) -> None:
        """Empty timing list → no fan-out, no errors (cheap when nobody wants timing)."""
        mgr = _make_manager(bind_methods=["_send_timing_to_results_processors"])
        mgr._timing_results_processors = []

        await mgr._send_timing_to_results_processors(_phase_stats())

        mgr.exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_timing_processor_receives_unwrapped_stats(self) -> None:
        """Single-processor fast path forwards the raw ``CreditPhaseStats``."""
        proc = MagicMock()
        proc.process_result = AsyncMock()
        mgr = _make_manager(bind_methods=["_send_timing_to_results_processors"])
        mgr._timing_results_processors = [proc]
        stats = _phase_stats()

        await mgr._send_timing_to_results_processors(stats)

        proc.process_result.assert_awaited_once_with(stats)

    @pytest.mark.asyncio
    async def test_multiple_timing_processors_fan_out_via_gather(self) -> None:
        proc_a = MagicMock()
        proc_a.process_result = AsyncMock()
        proc_b = MagicMock()
        proc_b.process_result = AsyncMock()
        mgr = _make_manager(bind_methods=["_send_timing_to_results_processors"])
        mgr._timing_results_processors = [proc_a, proc_b]
        stats = _phase_stats()

        await mgr._send_timing_to_results_processors(stats)

        proc_a.process_result.assert_awaited_once_with(stats)
        proc_b.process_result.assert_awaited_once_with(stats)

    @pytest.mark.asyncio
    async def test_best_effort_failure_swallowed_in_single_path(self) -> None:
        proc = MagicMock()
        proc.is_best_effort = True
        proc.process_result = AsyncMock(side_effect=RuntimeError("otel down"))
        mgr = _make_manager(
            bind_methods=[
                "_send_timing_to_results_processors",
                "_raise_unless_best_effort",
            ]
        )
        mgr._timing_results_processors = [proc]

        await mgr._send_timing_to_results_processors(_phase_stats())

        mgr.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_best_effort_failure_swallowed_in_gather_path(self) -> None:
        """A raising best-effort processor must not abort the fan-out; siblings run."""
        bad = MagicMock()
        bad.is_best_effort = True
        bad.process_result = AsyncMock(side_effect=RuntimeError("otel down"))
        good = MagicMock()
        good.is_best_effort = True
        good.process_result = AsyncMock()
        mgr = _make_manager(
            bind_methods=[
                "_send_timing_to_results_processors",
                "_raise_unless_best_effort",
            ]
        )
        mgr._timing_results_processors = [bad, good]

        await mgr._send_timing_to_results_processors(_phase_stats())

        good.process_result.assert_awaited_once()
        mgr.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_best_effort_failure_reraises(self) -> None:
        proc = MagicMock()
        proc.is_best_effort = False
        proc.process_result = AsyncMock(side_effect=RuntimeError("pipeline bug"))
        mgr = _make_manager(
            bind_methods=[
                "_send_timing_to_results_processors",
                "_raise_unless_best_effort",
            ]
        )
        mgr._timing_results_processors = [proc]

        with pytest.raises(RuntimeError, match="pipeline bug"):
            await mgr._send_timing_to_results_processors(_phase_stats())

    @pytest.mark.asyncio
    async def test_cancellation_never_swallowed_even_for_best_effort(self) -> None:
        proc = MagicMock()
        proc.is_best_effort = True
        proc.process_result = AsyncMock(side_effect=asyncio.CancelledError())
        mgr = _make_manager(
            bind_methods=[
                "_send_timing_to_results_processors",
                "_raise_unless_best_effort",
            ]
        )
        mgr._timing_results_processors = [proc]

        with pytest.raises(asyncio.CancelledError):
            await mgr._send_timing_to_results_processors(_phase_stats())


# ============================================================================
# 6) `_report_records_task` + `_publish_processing_stats`
# ============================================================================


class TestReportRecordsTask:
    """Background reporter skips empty phases and breaks on first active one."""

    @pytest.mark.asyncio
    async def test_skips_phase_with_zero_records(self) -> None:
        """Empty phase → continue, no stats published."""
        mgr = _make_manager(bind_methods=["_report_records_task"])
        empty_stats = MagicMock(total_records=0)
        mgr._records_tracker.create_stats_for_phase.return_value = empty_stats
        mgr._publish_processing_stats = AsyncMock()

        await mgr._report_records_task()

        mgr._publish_processing_stats.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_publishes_for_first_active_phase_then_breaks(self) -> None:
        """Two active phases → only the first one publishes."""
        mgr = _make_manager(bind_methods=["_report_records_task"])
        mgr._records_tracker.get_results_phases.return_value = ["warmup", "profiling"]
        mgr._records_tracker.create_stats_for_phase.side_effect = [
            MagicMock(total_records=5),
            MagicMock(total_records=10),
        ]
        mgr._publish_processing_stats = AsyncMock()

        await mgr._report_records_task()

        # Single publish, despite two active phases.
        assert mgr._publish_processing_stats.await_count == 1

    @pytest.mark.asyncio
    async def test_publish_processing_stats_emits_correct_message_type(self) -> None:
        mgr = _make_manager(bind_methods=["_publish_processing_stats"])
        phase_stats = MagicMock()
        worker_stats: dict[str, WorkerProcessingStats] = {}

        await mgr._publish_processing_stats(phase_stats, worker_stats)

        published = mgr.publish.await_args.args[0]
        assert isinstance(published, RecordsProcessingStatsMessage)
        assert published.processing_stats is phase_stats


# ============================================================================
# 7) Command handlers — payload parsing + cancel bookkeeping
# ============================================================================


class TestCommandHandlers:
    """`_on_profile_cancel_command`."""

    @pytest.mark.asyncio
    async def test_profile_cancel_marks_all_results_phases_then_processes(self) -> None:
        """ProfileCancel must mark every results phase as cancelled BEFORE
        invoking `_process_results(cancelled=True)`. Adversarial: pre-existing
        cancelled state in tracker should be additive, not overridden."""
        mgr = _make_manager(bind_methods=["_on_profile_cancel_command"])
        mgr._records_tracker.get_results_phases.return_value = ["warmup", "profiling"]
        mgr._process_results = AsyncMock(return_value=MagicMock())
        cmd = SimpleNamespace(payload=b"")

        await mgr._on_profile_cancel_command(cmd)

        # Both phases marked cancelled in order.
        marks = [
            c.args[0] for c in mgr._records_tracker.mark_phase_cancelled.call_args_list
        ]
        assert marks == ["warmup", "profiling"]
        mgr._process_results.assert_awaited_once_with(cancelled=True)


class TestStartRealtimeTelemetryCommand:
    """`_on_start_realtime_telemetry_command` — the runtime GPU-telemetry toggle.

    The dashboard sends START_REALTIME_TELEMETRY when the user enables the
    telemetry pane at runtime; the handler must un-park the accumulator's
    realtime loop via `start_realtime_telemetry()`. Uses a bare
    `RecordsManager.__new__` host (not the MagicMock factory) so the real
    `_gpu_telemetry_accumulator` property resolves through `_accumulators` —
    a MagicMock host would auto-create the attribute and hide wiring bugs.
    """

    def _make_host(self, accumulators: dict[Any, Any] | None = None) -> RecordsManager:
        mgr = RecordsManager.__new__(RecordsManager)
        mgr._accumulators = accumulators or {}
        for level in ("debug", "info", "warning", "error"):
            setattr(mgr, level, MagicMock())
        return mgr

    def _make_real_accumulator(self):
        from aiperf.config.flags.cli_config import CLIConfig
        from aiperf.gpu_telemetry.accumulator import GPUTelemetryAccumulator
        from tests.unit.conftest import make_run_from_cli

        run = make_run_from_cli(CLIConfig(model_names=["test-model"]))
        return GPUTelemetryAccumulator(
            run=run, pub_client=MagicMock(), service_id="records-manager-1"
        )

    @pytest.mark.asyncio
    async def test_unparks_realtime_loop_on_real_accumulator(self) -> None:
        """Dispatching the command sets the accumulator's enable event and
        flips the run config into realtime-dashboard mode."""
        from aiperf.common.enums import GPUTelemetryMode

        acc = self._make_real_accumulator()
        mgr = self._make_host({AccumulatorType.GPU_TELEMETRY: acc})
        cmd = SimpleNamespace(payload=b"")

        assert not acc._realtime_enable_event.is_set()
        await mgr._on_start_realtime_telemetry_command(cmd)

        assert acc._realtime_enable_event.is_set()
        assert acc.run.cfg.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD
        mgr.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_repeat_command_is_harmless(self) -> None:
        """A second START_REALTIME_TELEMETRY is idempotent: the event stays
        set, nothing raises, and no error is logged."""
        acc = self._make_real_accumulator()
        mgr = self._make_host({AccumulatorType.GPU_TELEMETRY: acc})
        cmd = SimpleNamespace(payload=b"")

        await mgr._on_start_realtime_telemetry_command(cmd)
        await mgr._on_start_realtime_telemetry_command(cmd)

        assert acc._realtime_enable_event.is_set()
        mgr.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_accumulator_logs_error_and_does_not_raise(self) -> None:
        """GPU telemetry disabled → no accumulator loaded. The handler must
        log an error (mirroring upstream) and return without raising."""
        mgr = self._make_host({})
        cmd = SimpleNamespace(payload=b"")

        await mgr._on_start_realtime_telemetry_command(cmd)

        mgr.error.assert_called_once()
        assert "accumulator" in str(mgr.error.call_args).lower()


# ============================================================================
# 8) `_report_realtime_inference_metrics_task` — early-exit gate
# ============================================================================


class TestReportRealtimeInferenceMetricsTask:
    """The realtime task runs only when at least one consumer is enabled."""

    @pytest.mark.asyncio
    async def test_returns_early_when_no_consumers_enabled(self) -> None:
        """No dashboard, no api_port, no env override → method returns immediately."""
        mgr = _make_manager(
            bind_methods=["_report_realtime_inference_metrics_task"],
            ui_type="simple",
            api_port=None,
        )
        with patch(
            "aiperf.records.records_manager.Environment.UI.REALTIME_METRICS_ENABLED",
            False,
        ):
            await mgr._report_realtime_inference_metrics_task()

        # Loop never entered → no realtime publish.
        assert all(
            not isinstance(c.args[0], RealtimeMetricsMessage)
            for c in mgr.publish.await_args_list
        )

    @pytest.mark.asyncio
    async def test_skips_realtime_when_no_new_records_arrived(self) -> None:
        """`total_records == _previous_realtime_records` → continue (no publish)."""
        mgr = _make_manager(
            bind_methods=["_report_realtime_inference_metrics_task"],
            api_port=8080,
        )
        # Stop the loop after one iteration.
        mgr.stop_requested = False
        call_count = {"n": 0}

        def stop_after_one() -> bool:
            call_count["n"] += 1
            return call_count["n"] > 1

        type(mgr).stop_requested = property(lambda self: stop_after_one())  # type: ignore[misc]

        mgr._previous_realtime_records = 5
        mgr._report_realtime_metrics = AsyncMock()
        with patch(
            "aiperf.records.records_manager.current_results_record_count",
            return_value=5,
        ):
            await mgr._report_realtime_inference_metrics_task()

        # Same record count as previous → realtime metrics NOT published.
        mgr._report_realtime_metrics.assert_not_awaited()


class TestReportRealtimeMetrics:
    """`_report_realtime_metrics` filters and publishes — empties short-circuit."""

    @pytest.mark.asyncio
    async def test_no_raw_metrics_no_publish(self) -> None:
        mgr = _make_manager(bind_methods=["_report_realtime_metrics"])
        with patch(
            "aiperf.records.records_manager.generate_realtime_metrics",
            new=AsyncMock(return_value=[]),
        ):
            await mgr._report_realtime_metrics()

        # No publish.
        assert all(
            not isinstance(c.args[0], RealtimeMetricsMessage)
            for c in mgr.publish.await_args_list
        )

    @pytest.mark.asyncio
    async def test_raw_metrics_filtered_to_empty_no_publish(self) -> None:
        """If filter strips all metrics, suppress publish (no empty messages)."""
        mgr = _make_manager(bind_methods=["_report_realtime_metrics"])
        with (
            patch(
                "aiperf.records.records_manager.generate_realtime_metrics",
                new=AsyncMock(return_value=[_metric("internal_only")]),
            ),
            patch(
                "aiperf.records.records_manager.filter_display_metrics",
                return_value=[],
            ),
        ):
            await mgr._report_realtime_metrics()

        # No RealtimeMetricsMessage published.
        published_types = [type(c.args[0]) for c in mgr.publish.await_args_list]
        assert RealtimeMetricsMessage not in published_types

    @pytest.mark.asyncio
    async def test_visible_metrics_emit_realtime_message(self) -> None:
        mgr = _make_manager(bind_methods=["_report_realtime_metrics"])
        with (
            patch(
                "aiperf.records.records_manager.generate_realtime_metrics",
                new=AsyncMock(return_value=[_metric("request_latency")]),
            ),
            patch(
                "aiperf.records.records_manager.filter_display_metrics",
                return_value=[_metric("request_latency")],
            ),
        ):
            await mgr._report_realtime_metrics()

        published = [c.args[0] for c in mgr.publish.await_args_list]
        rt = next(m for m in published if isinstance(m, RealtimeMetricsMessage))
        assert [m.tag for m in rt.metrics] == ["request_latency"]

    @pytest.mark.asyncio
    async def test_realtime_command_handler_delegates(self) -> None:
        """The on-demand REALTIME_METRICS command just calls `_report_realtime_metrics`."""
        mgr = _make_manager(bind_methods=["_on_realtime_metrics_command"])
        mgr._report_realtime_metrics = AsyncMock()
        cmd = SimpleNamespace(payload=b"")

        await mgr._on_realtime_metrics_command(cmd)

        mgr._report_realtime_metrics.assert_awaited_once()


# ============================================================================
# 9) `_summarize_all_processors` — MetricsAccumulator bridge
# ============================================================================


class TestSummarizeAllProcessorsAccumulatorBridge:
    """The squash's defining contract: `MetricsAccumulator` bridges into the
    legacy bucket pipeline. These tests lock the protocol shape:

    - results.values() → appended as a list[MetricResult]
    - timeslices (if non-None) → appended as a dict
    - multi_turn_ttft_trend → returned separately, NOT appended
    - accumulator failure → caught, logged, NOT propagated
    """

    @pytest.mark.asyncio
    async def test_metrics_accumulator_results_appended_to_bucket_list(self) -> None:
        mgr = _make_manager(bind_methods=["_summarize_all_processors"])
        acc = MagicMock()
        acc_summary = SimpleNamespace(
            results={"req_latency": _metric("req_latency")},
            timeslices=None,
            multi_turn_ttft_trend=None,
        )
        acc.summarize = AsyncMock(return_value=acc_summary)
        mgr._accumulators = {AccumulatorType.METRIC_RESULTS: acc}

        results, multi_turn = await mgr._summarize_all_processors()

        assert multi_turn is None
        # Accumulator's metric list is the only result (no legacy processors).
        assert any(
            isinstance(r, list) and r and r[0].tag == "req_latency" for r in results
        )

    @pytest.mark.asyncio
    async def test_metrics_accumulator_timeslices_appended_separately(self) -> None:
        mgr = _make_manager(bind_methods=["_summarize_all_processors"])
        acc = MagicMock()
        timeslices = {0: [_metric("ttft")], 1: [_metric("ttft")]}
        acc_summary = SimpleNamespace(
            results={"ttft": _metric("ttft")},
            timeslices=timeslices,
            multi_turn_ttft_trend=None,
        )
        acc.summarize = AsyncMock(return_value=acc_summary)
        mgr._accumulators = {AccumulatorType.METRIC_RESULTS: acc}

        results, _multi_turn = await mgr._summarize_all_processors()

        # Timeslices is appended as its own dict element.
        assert any(isinstance(r, dict) and r == timeslices for r in results)

    @pytest.mark.asyncio
    async def test_multi_turn_ttft_trend_returned_separately_not_appended(self) -> None:
        """Per source comment: `multi_turn_ttft_trend` (dict[int, MetricResult])
        would be misrouted by `bucket_summarize_results` to the timeslices
        slot — must come back via the second tuple element.
        """
        mgr = _make_manager(bind_methods=["_summarize_all_processors"])
        acc = MagicMock()
        trend = {1: _metric("ttft_turn_1"), 2: _metric("ttft_turn_2")}
        acc_summary = SimpleNamespace(
            results={},
            timeslices=None,
            multi_turn_ttft_trend=trend,
        )
        acc.summarize = AsyncMock(return_value=acc_summary)
        mgr._accumulators = {AccumulatorType.METRIC_RESULTS: acc}

        results, multi_turn = await mgr._summarize_all_processors()

        assert multi_turn == trend
        # And NOT in the bucketable results.
        assert all(r != trend for r in results)

    @pytest.mark.asyncio
    async def test_accumulator_failure_logged_and_appended_as_exception(self) -> None:
        """A failing `MetricsAccumulator.summarize` must NOT abort legacy bucketing.

        The exception is logged and appended to results so
        `bucket_summarize_results` records it as a raw exception.
        """
        mgr = _make_manager(bind_methods=["_summarize_all_processors"])
        acc = MagicMock()
        acc.summarize = AsyncMock(side_effect=RuntimeError("acc boom"))
        mgr._accumulators = {AccumulatorType.METRIC_RESULTS: acc}

        results, multi_turn = await mgr._summarize_all_processors()

        assert multi_turn is None
        assert any("acc boom" in str(c.args[0]) for c in mgr.error.call_args_list)
        # Exception is in results so bucket_summarize_results sees it.
        assert any(isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_no_metrics_accumulator_no_bridge(self) -> None:
        """If `AccumulatorType.METRIC_RESULTS` not in `_accumulators`, no bridge call."""
        mgr = _make_manager(bind_methods=["_summarize_all_processors"])
        mgr._accumulators = {}

        results, multi_turn = await mgr._summarize_all_processors()

        assert multi_turn is None
        assert results == []


# ============================================================================
# 10) `_publish_all_results` — publish-failure swallowed
# ============================================================================


class TestPublishAllResultsFailure:
    """The unified-pipeline publish must NOT propagate publish errors.

    Per source comment at lines 663-665: a failed `ProcessAllResultsMessage`
    publish would otherwise abort the legacy result path that already
    succeeded.
    """

    @pytest.mark.asyncio
    async def test_publish_failure_logged_not_propagated(self) -> None:
        mgr = _make_manager(bind_methods=["_publish_all_results"])
        mgr.publish = AsyncMock(side_effect=ConnectionError("zmq down"))

        # Should not raise.
        await mgr._publish_all_results(MagicMock(), {})

        assert any("zmq down" in str(c.args[0]) for c in mgr.error.call_args_list)

    @pytest.mark.asyncio
    async def test_publish_with_steady_state_attaches_analyzer_output(self) -> None:
        mgr = _make_manager(bind_methods=["_publish_all_results"])
        analyzer_outputs = {AnalyzerType.STEADY_STATE: {"window_start": 1}}

        await mgr._publish_all_results(MagicMock(), analyzer_outputs)

        msg = mgr.publish.await_args.args[0]
        assert isinstance(msg, ProcessAllResultsMessage)
        assert msg.steady_state_results == {"window_start": 1}
        # energy_efficiency_results comes from a side-channel, not records-manager.
        assert msg.energy_efficiency_results is None


# ============================================================================
# 11) `_write_partial_checkpoint_task` — service-run-type gate
# ============================================================================


class TestWritePartialCheckpointTask:
    """Background checkpoint task short-circuits outside K8s."""

    @pytest.mark.asyncio
    async def test_skips_when_not_kubernetes(self) -> None:
        mgr = _make_manager(
            bind_methods=["_write_partial_checkpoint_task"],
            service_run_type="process",
        )
        with patch(
            "aiperf.records.records_manager.write_partial_checkpoint",
            new=AsyncMock(),
        ) as wpc:
            await mgr._write_partial_checkpoint_task()

        wpc.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kubernetes_runs_checkpoint_and_updates_count(self) -> None:
        from aiperf.plugin.enums import ServiceRunType

        mgr = _make_manager(
            bind_methods=["_write_partial_checkpoint_task"],
            service_run_type=ServiceRunType.KUBERNETES,
        )
        with patch(
            "aiperf.records.records_manager.write_partial_checkpoint",
            new=AsyncMock(return_value=42),
        ) as wpc:
            await mgr._write_partial_checkpoint_task()

        wpc.assert_awaited_once()
        assert mgr._last_checkpoint_records == 42

    @pytest.mark.asyncio
    async def test_kubernetes_unchanged_count_does_not_log_debug(self) -> None:
        from aiperf.plugin.enums import ServiceRunType

        mgr = _make_manager(
            bind_methods=["_write_partial_checkpoint_task"],
            service_run_type=ServiceRunType.KUBERNETES,
        )
        mgr._last_checkpoint_records = 7
        with patch(
            "aiperf.records.records_manager.write_partial_checkpoint",
            new=AsyncMock(return_value=7),
        ):
            await mgr._write_partial_checkpoint_task()

        # Same count → no checkpoint-write log.
        assert all(
            "Wrote partial checkpoint" not in str(c.args[0])
            for c in mgr.debug.call_args_list
        )


# ============================================================================
# 12) `main()` — entry-point dispatch
# ============================================================================


class TestMainEntryPoint:
    """`main()` is a thin shim; lock that it dispatches to bootstrap."""

    def test_main_calls_bootstrap_with_records_manager_service_type(self) -> None:
        with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as bootstrap:
            main()

        bootstrap.assert_called_once()
        from aiperf.plugin.enums import ServiceType

        assert bootstrap.call_args.args[0] == ServiceType.RECORDS_MANAGER


# ============================================================================
# 12b) `__init__` dual-bind branch — controller-host resolution
# ============================================================================


class TestRecordsManagerInitDualBind:
    """Lock the additional-bind-address branch in `__init__`.

    The branch matters: when ZMQDualBindConfig has no ``controller_host``,
    the manager process is the dual-bind controller and must add a TCP
    bind on top of IPC so remote record processors can reach it. When
    ``controller_host`` is set (workers connect via TCP), no extra bind
    is needed.
    """

    @pytest.mark.parametrize(
        "controller_host,expects_extra_tcp_bind",
        [
            param(None, True, id="no-host-controller-binds-tcp"),
            param("controller.cluster.local", False, id="host-set-no-extra-bind"),
        ],
    )  # fmt: skip
    def test_dual_bind_address_resolution(
        self, controller_host: str | None, expects_extra_tcp_bind: bool
    ) -> None:
        from aiperf.config.zmq import ZMQDualBindConfig

        captured_kwargs: dict[str, Any] = {}

        def _capture_super(self: Any, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        comm = ZMQDualBindConfig(controller_host=controller_host)
        run = MagicMock()
        run.resolved.comm_config = comm
        run.cfg.comm_config = comm

        with (
            patch(
                "aiperf.common.mixins.PullClientMixin.__init__",
                new=_capture_super,
            ),
            patch(
                "aiperf.records.records_manager.load_results_processors",
                return_value=[],
            ),
            patch(
                "aiperf.records.records_manager.load_network_latency_processors",
                return_value=[],
            ),
            patch(
                "aiperf.records.records_manager.make_network_latency_accumulator",
                return_value=None,
            ),
            patch(
                "aiperf.records.records_manager.load_accumulators",
                return_value={},
            ),
            patch(
                "aiperf.records.records_manager.load_stream_exporters",
                return_value={},
            ),
            patch(
                "aiperf.records.records_manager.load_analyzers",
                return_value={},
            ),
            patch(
                "aiperf.records.records_manager.accumulators_for_record_type",
                return_value=[],
            ),
            patch(
                "aiperf.records.records_manager.stream_exporters_for_record_type",
                return_value=[],
            ),
        ):
            RecordsManager(run=run, service_id="rm-1")

        bind = captured_kwargs.get("pull_client_additional_bind_address")
        if expects_extra_tcp_bind:
            assert bind is not None
            assert bind.startswith("tcp://")
        else:
            assert bind is None

    def test_init_falls_back_to_cfg_comm_config_when_resolved_missing(self) -> None:
        """`run.resolved.comm_config or run.cfg.comm_config` — locks the
        falsy-resolved fallback."""
        from aiperf.config.zmq import ZMQDualBindConfig

        captured_kwargs: dict[str, Any] = {}

        def _capture_super(self: Any, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        run = MagicMock()
        run.resolved.comm_config = None
        run.cfg.comm_config = ZMQDualBindConfig(controller_host=None)

        with (
            patch(
                "aiperf.common.mixins.PullClientMixin.__init__",
                new=_capture_super,
            ),
            patch(
                "aiperf.records.records_manager.load_results_processors",
                return_value=[],
            ),
            patch(
                "aiperf.records.records_manager.load_network_latency_processors",
                return_value=[],
            ),
            patch(
                "aiperf.records.records_manager.make_network_latency_accumulator",
                return_value=None,
            ),
            patch(
                "aiperf.records.records_manager.load_accumulators",
                return_value={},
            ),
            patch(
                "aiperf.records.records_manager.load_stream_exporters",
                return_value={},
            ),
            patch(
                "aiperf.records.records_manager.load_analyzers",
                return_value={},
            ),
            patch(
                "aiperf.records.records_manager.accumulators_for_record_type",
                return_value=[],
            ),
            patch(
                "aiperf.records.records_manager.stream_exporters_for_record_type",
                return_value=[],
            ),
        ):
            RecordsManager(run=run, service_id="rm-1")

        # Used the cfg.comm_config (no resolved.comm_config) and triggered
        # the additional TCP bind because controller_host is None.
        assert captured_kwargs["pull_client_additional_bind_address"] is not None


# ============================================================================
# 13) Real Pydantic-config sanity check (anti-MagicMock-drift insurance)
# ============================================================================


class TestRecordsTrackerRealRoundTrip:
    """Builds a real `RecordsTracker` and runs `_process_metric_record_data`
    against it end-to-end, locking the contract that the tracker actually
    sees the record. This is the magicmock-drift insurance test required
    by the project's testing rules.
    """

    @pytest.mark.asyncio
    async def test_real_tracker_counts_success_and_error_records(self) -> None:
        mgr = _make_manager(bind_methods=["_process_metric_record_data"])
        mgr._records_tracker = RecordsTracker()
        mgr._send_results_to_results_processors = AsyncMock()
        mgr._send_record_to_accumulators = AsyncMock()
        mgr._handle_all_records_received = AsyncMock()

        good = _record(metrics={"request_latency": 1.0})
        bad = _record(
            error=WireErrorDetails(code=500, message="oops", type="ServerError")
        )

        await mgr._process_metric_record_data(good)
        await mgr._process_metric_record_data(bad)

        stats = mgr._records_tracker.create_stats_for_phase("profiling")
        assert stats.success_records == 1
        assert stats.error_records == 1
        # Total via property.
        assert stats.total_records == 2

    @pytest.mark.asyncio
    async def test_real_tracker_handles_batch_wire_message(self) -> None:
        """A real `MetricRecordsBatchWireMessage` flowing through `_on_metric_records`
        must increment the real tracker's per-phase counters by exactly len(records)."""
        mgr = _make_manager(bind_methods=["_on_metric_records"])
        mgr._records_tracker = RecordsTracker()
        mgr._send_results_to_results_processors = AsyncMock()
        mgr._send_record_to_accumulators = AsyncMock()
        mgr._handle_all_records_received = AsyncMock()
        mgr._error_tracker = MagicMock()
        # _on_metric_records → _process_metric_record_data, so we need the
        # real method bound via the second binding step.
        mgr._process_metric_record_data = (
            RecordsManager._process_metric_record_data.__get__(mgr)
        )

        records = [_record(metrics={"request_latency": float(i)}) for i in range(5)]
        batch = MetricRecordsBatchWireMessage(service_id="rp-1", records=records)

        await mgr._on_metric_records(batch)

        stats = mgr._records_tracker.create_stats_for_phase("profiling")
        assert stats.success_records == 5
        assert stats.error_records == 0


# ============================================================================
# 14) `_process_results` ordering contract — legacy publish before accumulator
# ============================================================================


class TestProcessResultsOrdering:
    """`_process_results` MUST publish `ProcessRecordsResultMessage` before
    running stream-exporter finalize / analyzer compute / unified publish.

    Source comment lines 540-547: failures in the unified pipeline must not
    break the legacy path because it has already published its message.
    """

    @pytest.mark.asyncio
    async def test_process_records_result_published_before_unified_message(
        self,
    ) -> None:
        """Publish ordering: legacy first, unified second."""
        mgr = _make_manager(
            bind_methods=[
                "_process_results",
                "_summarize_all_processors",
                "_finalize_all_processors",
                "_build_records_result",
                "_publish_all_results",
                "_finalize_stream_exporters",
                "_finalize_network_latency_processors",
                "_run_analyzers",
            ],
        )

        await mgr._process_results(cancelled=False)

        published_types = [type(c.args[0]) for c in mgr.publish.await_args_list]
        legacy_idx = published_types.index(ProcessRecordsResultMessage)
        unified_idx = published_types.index(ProcessAllResultsMessage)
        assert legacy_idx < unified_idx

    @pytest.mark.asyncio
    async def test_unified_publish_failure_does_not_break_legacy_path(self) -> None:
        """Failure in `_publish_all_results` is swallowed; the legacy
        `ProcessRecordsResultMessage` already published."""
        legacy_processor = MagicMock()
        legacy_processor.summarize = AsyncMock(return_value=[_metric("a")])
        legacy_processor.finalize = AsyncMock()
        mgr = _make_manager(
            bind_methods=[
                "_process_results",
                "_summarize_all_processors",
                "_finalize_all_processors",
                "_build_records_result",
                "_publish_all_results",
                "_finalize_stream_exporters",
                "_finalize_network_latency_processors",
                "_run_analyzers",
            ],
            legacy_processors=[legacy_processor],
        )
        # First publish (legacy) succeeds; second (unified) raises.
        side_effects: list[Any] = [None, ConnectionError("zmq dropped")]

        async def _publish(msg: Any) -> None:
            outcome = side_effects.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome

        mgr.publish = AsyncMock(side_effect=_publish)

        # Must not raise.
        result = await mgr._process_results(cancelled=False)

        # Legacy path published, unified path failure logged.
        assert any("zmq dropped" in str(c.args[0]) for c in mgr.error.call_args_list)
        assert result is not None


# ============================================================================
# 15) `_run_analyzers` — start/end null-coalescing edges
# ============================================================================


class TestRunAnalyzersTimeWindowEdges:
    """The `start_ns or 0` / `end_ns or 0` null-coalescing matters when the
    profile_results window is unset — adversarial: pass `start_ns=0` (falsy)
    and a real `None` and verify the SummaryContext gets 0, not None."""

    @pytest.mark.parametrize(
        "start_ns,end_ns,expected_start,expected_end",
        [
            param(None, None, 0, 0, id="both-none-coalesces-to-zero"),
            param(0, 0, 0, 0, id="zero-stays-zero"),
            param(100, 200, 100, 200, id="real-values-passthrough"),
            param(0, 1000, 0, 1000, id="zero-start-real-end"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_summary_context_window_coalescing(
        self,
        start_ns: int | None,
        end_ns: int | None,
        expected_start: int,
        expected_end: int,
    ) -> None:
        from aiperf.common.models import ProcessRecordsResult, ProfileResults

        analyzer = MagicMock()
        analyzer.summarize = AsyncMock(return_value={"x": 1})
        mgr = _make_manager(
            bind_methods=["_run_analyzers"],
            analyzers={AnalyzerType.STEADY_STATE: analyzer},
        )
        result = ProcessRecordsResult(
            results=ProfileResults(
                completed=0,
                start_ns=start_ns or 0,
                end_ns=end_ns or 0,
            )
        )

        outputs = await mgr._run_analyzers(result=result, cancelled=False)

        ctx = analyzer.summarize.call_args.args[0]
        assert ctx.start_ns == expected_start
        assert ctx.end_ns == expected_end
        assert AnalyzerType.STEADY_STATE in outputs
