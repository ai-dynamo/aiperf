# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``RecordsManager._process_results()``.

The ``accumulator`` plugin category drives metric summarization:
``MetricsAccumulator`` returns :class:`AccumulatorMetricsSummary`
(``results: dict[tag, MetricResult]``, ``timeslices``); GPU telemetry /
server metrics accumulators return list-shaped results.

The pipeline:

1. ``_await_telemetry_ingest_complete`` drains the GPU telemetry producer.
2. ``_deliver_network_rtt_to_accumulators`` wires optional RTT calibration.
3. ``_summarize_metric_record_accumulators`` exports metric-record accumulators.
4. ``_finalize_stream_exporters`` flushes JSONL writers concurrently.
5. ``ProcessRecordsResultMessage`` is published.
6. ``ProcessAllResultsMessage`` is published for the SystemController fan-in.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
from pytest import param

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    ProcessAllResultsMessage,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
)
from aiperf.common.models import (
    ErrorDetails,
    MetricResult,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProfileMetricDurationCoverage,
    TimesliceResult,
)
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn, StreamingContent, WorkerToRouterMessage
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.plugin.enums import AccumulatorType, StreamExporterType, TimingMode
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_tracker import RecordsTracker
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from tests.unit.conftest import make_benchmark_run
from tests.unit.post_processors.conftest import create_metric_records_data

# ---------------------------------------------------------------------------
# Stub fixtures
# ---------------------------------------------------------------------------


_STUB_METRIC_RESULT = MetricResult(
    tag="request_latency",
    header="Request Latency",
    unit="ms",
    avg=100.0,
    count=10,
)


def _make_summary_accumulator(
    results: list[MetricResult] | None = None,
    *,
    timeslices: list[TimesliceResult] | None = None,
    summarize_exc: BaseException | None = None,
) -> MagicMock:
    """Stub for an ``AccumulatorProtocol`` returning :class:`AccumulatorMetricsSummary`."""
    acc = MagicMock()
    acc.__class__.__name__ = "StubMetricsAccumulator"
    if summarize_exc is not None:
        acc.summarize = AsyncMock(side_effect=summarize_exc)
    else:
        results_dict = {
            r.tag: r
            for r in (results if results is not None else [_STUB_METRIC_RESULT])
        }
        acc.summarize = AsyncMock(
            return_value=AccumulatorMetricsSummary(
                results=results_dict,
                timeslices=timeslices,
            )
        )
    return acc


def _make_list_accumulator(
    results: list[MetricResult] | None = None,
    summarize_exc: BaseException | None = None,
) -> MagicMock:
    """Stub for an accumulator returning ``list[MetricResult]``."""
    acc = MagicMock()
    acc.__class__.__name__ = "StubListAccumulator"
    if summarize_exc is not None:
        acc.summarize = AsyncMock(side_effect=summarize_exc)
    else:
        acc.summarize = AsyncMock(
            return_value=results if results is not None else [_STUB_METRIC_RESULT]
        )
    return acc


def _make_stub_stream_exporter() -> MagicMock:
    exp = MagicMock()
    exp.finalize = AsyncMock()
    return exp


def _make_manager_mock(
    *,
    accumulators: dict[AccumulatorType, MagicMock] | None = None,
    stream_exporters: dict[StreamExporterType, MagicMock] | None = None,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
    user_config_telemetry_disabled: bool = True,
    user_config_server_metrics_disabled: bool = True,
) -> MagicMock:
    """Build a mock ``RecordsManager`` with the unified pipeline methods bound.

    GPU telemetry / server metrics accumulators are absent by default and
    the user_config flags disable both side-channel publishes — those
    paths are exercised by separate target-side tests, not here.
    """
    mgr = MagicMock()
    mgr._accumulators = accumulators or {}
    # The byte-exact summary engine only summarizes accumulators that are
    # registered as metric_record-typed; mirror the production gate by treating
    # every supplied accumulator as a metric_record accumulator here.
    mgr._metric_record_accumulators = list((accumulators or {}).values())
    mgr._stream_exporters = stream_exporters or {}
    mgr._gpu_telemetry_accumulator = None
    mgr._server_metrics_accumulator = None
    # Branch-stats snapshot (read by _process_results).
    mgr._latest_branch_stats = None
    # No stall-watchdog degradation by default.
    mgr._incomplete_reason = None

    # Records tracker — drives the time window via PROFILING phase stats.
    phase_stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        start_ns=start_ns,
        requests_end_ns=end_ns,
    )
    mgr._records_tracker.create_stats_for_phase.return_value = phase_stats

    # Error tracker — empty errors keep the success path.
    mgr._error_tracker.get_error_summary_for_phase.return_value = []

    # v2 config — disable telemetry / server-metrics side channels via run.cfg.
    mgr.run = MagicMock()
    mgr.run.cfg.gpu_telemetry_disabled = user_config_telemetry_disabled
    mgr.run.cfg.server_metrics_disabled = user_config_server_metrics_disabled
    mgr.run.cfg.scenario = None
    mgr.run.cfg.get_profiling_phases.return_value = []

    # Logging
    mgr.debug = MagicMock()
    mgr.info = MagicMock()
    mgr.error = MagicMock()
    mgr.warning = MagicMock()
    mgr.exception = MagicMock()

    # Service identity + publish
    mgr.service_id = "test_records_manager"
    mgr.publish = AsyncMock()

    # Single-flight guard state read by the real _process_results wrapper.
    mgr._process_results_lock = asyncio.Lock()
    mgr._processed_results = {}

    # Bind real methods
    mgr._process_results = RecordsManager._process_results.__get__(mgr)
    mgr._process_results_impl = RecordsManager._process_results_impl.__get__(mgr)
    mgr._finalize_record_processor_artifacts = AsyncMock()
    mgr._await_telemetry_ingest_complete = AsyncMock(return_value=[])
    mgr._summarize_metric_record_accumulators = (
        RecordsManager._summarize_metric_record_accumulators.__get__(mgr)
    )
    mgr._has_records_for_phase = RecordsManager._has_records_for_phase.__get__(mgr)
    mgr._summarize_warmup_metric_records = (
        RecordsManager._summarize_warmup_metric_records.__get__(mgr)
    )
    mgr._validate_profile_metric_duration_coverage = (
        RecordsManager._validate_profile_metric_duration_coverage.__get__(mgr)
    )
    mgr._iter_concrete_phase_stats = RecordsManager._iter_concrete_phase_stats.__get__(
        mgr
    )
    mgr._summarize_one_accumulator = RecordsManager._summarize_one_accumulator.__get__(
        mgr
    )
    mgr._bucket_accumulator_summary = (
        RecordsManager._bucket_accumulator_summary.__get__(mgr)
    )
    mgr._analyzers = []
    mgr._run_analyzers = RecordsManager._run_analyzers.__get__(mgr)
    mgr._finalize_stream_exporters = RecordsManager._finalize_stream_exporters.__get__(
        mgr
    )
    mgr._publish_all_results = RecordsManager._publish_all_results.__get__(mgr)
    mgr._publish_telemetry_results = RecordsManager._publish_telemetry_results.__get__(
        mgr
    )
    return mgr


# ---------------------------------------------------------------------------
# Tests: accumulator summarize fan-out
# ---------------------------------------------------------------------------


class TestProcessResultsAccumulatorPath:
    """``_process_results`` runs ``summarize`` on every accumulator and bridges
    both the typed :class:`AccumulatorMetricsSummary` shape and the legacy
    ``list[MetricResult]`` shape into the published
    :class:`ProcessRecordsResultMessage`."""

    @pytest.mark.asyncio
    async def test_calls_summarize_on_all_accumulators(self) -> None:
        acc1 = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc2 = _make_list_accumulator([])

        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: acc1,
                AccumulatorType.GPU_TELEMETRY: acc2,
            }
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        acc1.summarize.assert_awaited_once()
        acc2.summarize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publishes_process_records_result_message(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessRecordsResultMessage) for m in published)

    @pytest.mark.asyncio
    async def test_returns_process_records_result(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert isinstance(result, ProcessRecordsResult)
        assert result.results.records is not None
        assert _STUB_METRIC_RESULT in result.results.records

    @pytest.mark.asyncio
    async def test_local_artifact_barrier_failure_preserves_summary(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._is_kubernetes_run = MagicMock(return_value=False)
        mgr._finalize_record_processor_artifacts = AsyncMock(
            side_effect=RuntimeError("artifact barrier failed")
        )

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.results.records == [_STUB_METRIC_RESULT]
        assert result.results.completed == 1
        assert any(
            error.message == "RuntimeError('artifact barrier failed')"
            and error.details
            == {
                "stage": "record_processor_artifact_finalize",
                "fatal": False,
            }
            for error in result.errors
        )
        mgr.error.assert_called_once_with(
            "Non-fatal record-processor artifact finalization failure: "
            "RuntimeError('artifact barrier failed')"
        )

    @pytest.mark.asyncio
    async def test_kubernetes_artifact_barrier_failure_remains_fatal(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._is_kubernetes_run = MagicMock(return_value=True)
        mgr._finalize_record_processor_artifacts = AsyncMock(
            side_effect=RuntimeError("artifact barrier failed")
        )

        with pytest.raises(RuntimeError, match="artifact barrier failed"):
            await mgr._process_results_impl(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        acc.summarize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_legacy_list_shape_accumulator_results_extended(self) -> None:
        """``list[MetricResult]`` accumulator output is appended to records."""
        acc_list = _make_list_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.GPU_TELEMETRY: acc_list})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert _STUB_METRIC_RESULT in (result.results.records or [])

    @pytest.mark.asyncio
    async def test_accumulator_summarize_failure_does_not_abort(self) -> None:
        """A failing summarize is wrapped into ``result.errors`` but the
        unified pipeline still runs."""
        failing = _make_summary_accumulator(
            summarize_exc=RuntimeError("summarize boom")
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: failing})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        # Errors logged + included in result.errors
        mgr.error.assert_called()
        assert any("summarize boom" in str(err.message or err) for err in result.errors)

    @pytest.mark.asyncio
    async def test_agentx_metric_coverage_failure_is_fatal(self) -> None:
        """A duration-based AgentX phase below 95% remains exportable but fatal."""
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc.profile_metric_duration_coverage.return_value = (
            ProfileMetricDurationCoverage(
                phase_name="profiling",
                expected_duration_seconds=3600.0,
                required_ratio=0.98,
                ttft_ratio=0.861194,
                inter_token_latency_ratio=0.861569,
            )
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._has_multiple_phase_instances.return_value = False
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 3600.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert len(result.fatal_errors) == 1
        assert result.fatal_errors[0].type == "ProfileMetricCoverageError"
        assert "required 95.0%" in result.fatal_errors[0].message
        assert "check inference server logs" in result.fatal_errors[0].message
        assert result.results.runtime_submission_invalid_reasons == [
            "insufficient_profile_metric_coverage"
        ]
        assert result.results.metric_duration_coverage[0].ttft_ratio == pytest.approx(
            0.861194
        )

    @pytest.mark.asyncio
    async def test_agentx_metric_coverage_validation_failure_has_distinct_reason(
        self,
    ) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc.profile_metric_duration_coverage = None
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._has_multiple_phase_instances.return_value = False
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 3600.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.fatal_errors[0].type == "ProfileMetricCoverageValidationError"
        assert result.results.runtime_submission_invalid_reasons == [
            "profile_metric_coverage_validation_failed"
        ]

    @pytest.mark.asyncio
    async def test_agentx_metric_coverage_uses_real_accumulator_ratio(self) -> None:
        phase_start_ns = 1_000 * NANOS_PER_SECOND
        accumulator = MetricsAccumulator(make_benchmark_run(streaming=True))
        await accumulator.process_record(
            create_metric_records_data(
                session_num=0,
                request_start_ns=1_900 * NANOS_PER_SECOND,
                request_end_ns=1_902 * NANOS_PER_SECOND,
                results=[
                    {"time_to_first_token": NANOS_PER_SECOND},
                    {"inter_token_latency": 100_000_000},
                ],
            )
        )
        accumulator.summarize = AsyncMock(
            return_value=AccumulatorMetricsSummary(
                results={_STUB_METRIC_RESULT.tag: _STUB_METRIC_RESULT},
                timeslices=None,
            )
        )
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: accumulator},
            start_ns=phase_start_ns,
        )
        mgr._has_multiple_phase_instances.return_value = False
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 1_000.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        coverage = result.results.metric_duration_coverage[0]
        assert coverage.ttft_ratio == pytest.approx(0.901)
        assert coverage.inter_token_latency_ratio == pytest.approx(0.902)
        assert coverage.passed is False
        assert result.results.runtime_submission_invalid_reasons == [
            "insufficient_profile_metric_coverage"
        ]

    @pytest.mark.asyncio
    async def test_agentx_metric_coverage_passes_at_threshold(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc.profile_metric_duration_coverage.return_value = (
            ProfileMetricDurationCoverage(
                phase_name="profiling",
                expected_duration_seconds=3600.0,
                required_ratio=0.98,
                ttft_ratio=0.98,
                inter_token_latency_ratio=0.999,
            )
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._has_multiple_phase_instances.return_value = False
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 3600.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.fatal_errors == []
        assert result.results.runtime_submission_invalid_reasons == []

    @pytest.mark.asyncio
    async def test_agentx_metric_coverage_accepts_late_streaming_activity(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc.profile_metric_duration_coverage.return_value = (
            ProfileMetricDurationCoverage(
                phase_name="profiling",
                expected_duration_seconds=3600.0,
                required_ratio=0.98,
                ttft_ratio=0.979504,
                inter_token_latency_ratio=1.0,
            )
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._has_multiple_phase_instances.return_value = False
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 3600.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.fatal_errors == []
        assert result.results.runtime_submission_invalid_reasons == []

    @pytest.mark.asyncio
    async def test_agentx_short_unsafe_smoke_skips_metric_coverage(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._has_multiple_phase_instances.return_value = False
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 30.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        acc.profile_metric_duration_coverage.assert_not_called()
        assert result.fatal_errors == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "last_content_seconds,expected_ratio,passes",
        [
            param(1790, 1790 / 1800, True, id="cancelled-stream-near-end"),
            param(1710, 0.95, True, id="exact-threshold"),
            param(1709, 1709 / 1800, False, id="stalled-before-threshold"),
            param(1830, 1.0, True, id="grace-period-clamped"),
            param(-10, 0.0, False, id="before-phase-clamped"),
            param(None, 0.0, False, id="cancelled-without-content"),
        ],
    )  # fmt: skip
    async def test_process_results_cancelled_stream_coverage(
        self, last_content_seconds: int | None, expected_ratio: float, passes: bool
    ) -> None:
        """Cancellation alone cannot rescue a tail with no recent content."""
        acc = _make_summary_accumulator()
        acc.profile_metric_duration_coverage.return_value = (
            ProfileMetricDurationCoverage(
                phase_name="profiling",
                expected_duration_seconds=1800,
                required_ratio=0.95,
                ttft_ratio=0.8,
                inter_token_latency_ratio=0.9,
            )
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._has_multiple_phase_instances.return_value = False
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 1800
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]
        mgr._records_tracker.create_stats_for_phase.return_value = PhaseRecordsStats(
            phase=CreditPhase.PROFILING,
            start_ns=100_000_000_000,
            requests_end_ns=1940_000_000_000,
            final_requests_completed=10,
            final_requests_cancelled=1,
            grace_period_timeout_triggered=True,
            last_streaming_content_ns=(
                (100 + last_content_seconds) * 1_000_000_000
                if last_content_seconds is not None
                else None
            ),
        )

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        coverage = result.results.metric_duration_coverage[0]
        assert coverage.streaming_content_ratio == pytest.approx(expected_ratio)
        assert coverage.ttft_ratio == 0.8
        assert coverage.inter_token_latency_ratio == 0.9
        assert coverage.passed is passes
        assert bool(result.fatal_errors) is not passes
        assert result.results.runtime_submission_invalid_reasons == (
            [] if passes else ["insufficient_profile_metric_coverage"]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("credit_returns", [True, False])
    async def test_cancelled_credit_content_survives_wire_and_phase_stats(
        self,
        benchmark_run: BenchmarkRun,
        credit_returns: bool,
    ) -> None:
        """Credit cancellation preserves liveness without creating a metric record."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            profiling_index=0,
            phase_name="first",
            timing_mode=TimingMode.REQUEST_RATE,
            expected_duration_sec=1800,
        )
        progress = PhaseProgressTracker(config)
        lifecycle = PhaseLifecycle(config)
        lifecycle.start()
        phase_start_ns = lifecycle.started_at_ns
        assert phase_start_ns is not None
        progress.increment_sent(
            TurnToSend(
                conversation_id="long",
                x_correlation_id="long",
                turn_index=0,
                num_turns=1,
            )
        )
        lifecycle.mark_sending_complete(timeout_triggered=True)
        progress.freeze_sent_counts()
        handler = CreditCallbackHandler(MagicMock())
        handler.register_phase(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            progress=progress,
            lifecycle=lifecycle,
            stop_checker=MagicMock(),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )
        message = CreditReturn(
            credit=Credit(
                id=0,
                phase=CreditPhase.PROFILING,
                phase_index=1,
                conversation_id="long",
                x_correlation_id="long",
                turn_index=0,
                num_turns=1,
                issued_at_ns=phase_start_ns,
            ),
            cancelled=True,
            last_streaming_content_ns=phase_start_ns + 1790_000_000_000,
        )
        router = StickyCreditRouter(run=benchmark_run, service_id="coverage-router")
        router.set_streaming_content_callback(handler.on_streaming_content)
        content = StreamingContent(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            timestamp_ns=phase_start_ns
            + (1709 if credit_returns else 1790) * 1_000_000_000,
        )
        decoded_content = msgspec.msgpack.decode(
            msgspec.msgpack.encode(content), type=WorkerToRouterMessage
        )
        await router._handle_router_message("worker", decoded_content)
        assert (
            progress.create_stats(lifecycle).last_streaming_content_ns
            == content.timestamp_ns
        )
        assert progress.in_flight == 1
        assert not progress.all_credits_returned_event.is_set()
        if credit_returns:
            decoded = msgspec.msgpack.decode(
                msgspec.msgpack.encode(message), type=WorkerToRouterMessage
            )
            await handler.on_credit_return("worker", decoded)
            assert progress.all_credits_returned_event.is_set()
        progress.observe_streaming_content(phase_start_ns + 1000_000_000_000)
        progress.observe_streaming_content(None)
        lifecycle.mark_complete(grace_period_triggered=True)
        progress.freeze_completed_counts()
        stats = progress.create_stats(lifecycle)
        tracker = RecordsTracker()
        tracker.update_phase_info(
            type(stats).model_validate_json(stats.model_dump_json())
        )
        second_config = config.model_copy(
            update={"phase_index": 2, "profiling_index": 1, "phase_name": "second"}
        )
        second_lifecycle = PhaseLifecycle(second_config)
        second_lifecycle.start()
        tracker.update_phase_info(
            PhaseProgressTracker(second_config).create_stats(second_lifecycle)
        )

        accumulator = MetricsAccumulator(run=benchmark_run)
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: accumulator}
        )
        mgr._records_tracker = tracker
        mgr._has_multiple_phase_instances.return_value = True
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phases = []
        for name in ("first", "second"):
            phase = MagicMock()
            phase.name = name
            phase.duration = 1800
            phases.append(phase)
        mgr.run.cfg.get_profiling_phases.return_value = phases

        coverage, errors = mgr._validate_profile_metric_duration_coverage(
            CreditPhase.PROFILING, False
        )

        assert coverage[0].passed is True
        assert coverage[0].streaming_content_ratio == pytest.approx(1790 / 1800)
        assert coverage[0].ttft_ratio == coverage[0].inter_token_latency_ratio == 0.0
        assert coverage[1].passed is False
        assert len(errors) == 1
        assert errors[0].details["phase_name"] == "second"
        assert accumulator.record_count == 0
        assert stats.final_requests_completed == 0
        assert stats.final_requests_cancelled == int(credit_returns)
        assert (
            tracker.create_stats_for_phase(CreditPhase.PROFILING, 1).total_records == 0
        )

    @pytest.mark.asyncio
    async def test_empty_accumulators_produces_empty_records(self) -> None:
        mgr = _make_manager_mock(accumulators={})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert isinstance(result, ProcessRecordsResult)
        assert result.results.records == []

    @pytest.mark.asyncio
    async def test_timeslices_propagated_to_profile_results(self) -> None:
        """``timeslices`` from AccumulatorMetricsSummary populates
        ``ProfileResults.timeslices``."""
        slice_metrics = {
            "request_latency": MetricResult(
                tag="request_latency",
                header="Latency",
                unit="ms",
                avg=100.0,
                count=5,
            )
        }
        timeslices = [
            TimesliceResult(
                start_ns=1_000_000_000,
                end_ns=2_000_000_000,
                metric_results=slice_metrics,
            )
        ]
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT], timeslices=timeslices)
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.results.timeslices is not None
        assert len(result.results.timeslices) == 1
        assert result.results.timeslices[0].start_ns == 1_000_000_000
        assert result.results.timeslices[0].metric_results == slice_metrics


# ---------------------------------------------------------------------------
# Tests: cancelled flag propagation
# ---------------------------------------------------------------------------


class TestProcessResultsCancelled:
    @pytest.mark.asyncio
    async def test_cancelled_true_propagated_to_profile_results(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=True)

        assert result.results.was_cancelled is True
        assert result.results.is_complete is False
        assert (
            result.results.incomplete_reason
            == "Run cancelled before all records were processed."
        )

    @pytest.mark.asyncio
    async def test_cancelled_false_propagated_to_profile_results(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.results.was_cancelled is False


# ---------------------------------------------------------------------------
# Tests: _finalize_stream_exporters integration
# ---------------------------------------------------------------------------


class TestProcessResultsStreamExporters:
    @pytest.mark.asyncio
    async def test_telemetry_drain_precedes_summary_and_finalize(self) -> None:
        order: list[str] = []
        drain_error = ErrorDetails(
            message="telemetry drain incomplete",
            details={"stage": "gpu_telemetry_drain"},
        )
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        exp = _make_stub_stream_exporter()

        async def summarize(*_args, **_kwargs) -> AccumulatorMetricsSummary:
            order.append("summarize")
            return AccumulatorMetricsSummary(
                results={_STUB_METRIC_RESULT.tag: _STUB_METRIC_RESULT}
            )

        async def finalize() -> None:
            order.append("finalize")

        async def await_drain() -> list[ErrorDetails]:
            order.append("drain")
            return [drain_error]

        acc.summarize.side_effect = summarize
        exp.finalize.side_effect = finalize
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )
        mgr._await_telemetry_ingest_complete = AsyncMock(side_effect=await_drain)

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert order == ["drain", "summarize", "finalize"]
        assert drain_error in result.errors

    @pytest.mark.asyncio
    async def test_stream_exporters_finalized(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        exp = _make_stub_stream_exporter()
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        exp.finalize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_stream_exporters_is_noop(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={},
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessAllResultsMessage) for m in published)

    @pytest.mark.asyncio
    async def test_finalize_failure_is_published_in_process_result(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        exp = _make_stub_stream_exporter()
        exp.finalize.side_effect = OSError("stream flush disk full")
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert len(result.errors) == 1
        assert result.errors[0].type == "OSError"
        messages = [
            call.args[0]
            for call in mgr.publish.await_args_list
            if isinstance(call.args[0], ProcessRecordsResultMessage)
        ]
        assert len(messages) == 1
        assert messages[0].results.errors == result.errors


# ---------------------------------------------------------------------------
# Tests: analyzer execution + ProcessAllResultsMessage publish
# ---------------------------------------------------------------------------


def _get_published_all_results(mgr: MagicMock) -> ProcessAllResultsMessage | None:
    """Return the published ``ProcessAllResultsMessage`` if any."""
    for call in mgr.publish.await_args_list:
        msg = call.args[0]
        if isinstance(msg, ProcessAllResultsMessage):
            return msg
    return None


class TestProcessResultsAllResultsPublish:
    """``_process_results`` publishes :class:`ProcessAllResultsMessage` for the
    SystemController fan-in."""

    @pytest.mark.asyncio
    async def test_publishes_process_all_results_message(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        msg = _get_published_all_results(mgr)
        assert msg is not None


class TestProcessResultsSingleFlight:
    """``_process_results`` is single-flight per phase: the natural finalize task
    and the PROCESS_RECORDS / PROFILE_CANCEL commands can all reach it, but only
    the first does the work; later calls return the cached result without
    re-publishing or re-finalizing the stream exporters."""

    @pytest.mark.asyncio
    async def test_second_call_returns_cached_and_does_not_republish(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        exp = _make_stub_stream_exporter()
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )

        first = await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)
        publish_count = mgr.publish.await_count

        second = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert second is first
        # No additional publishes and the exporter is finalized exactly once.
        assert mgr.publish.await_count == publish_count
        exp.finalize.assert_awaited_once()


class TestProcessResultsServerMetricsOwnership:
    """RecordsManager never publishes manager-owned server metrics."""

    @pytest.mark.asyncio
    async def test_server_metrics_enabled_still_publishes_only_records_fanin(
        self,
    ) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            user_config_server_metrics_disabled=False,
        )
        await mgr._process_results(phase=CreditPhase.WARMUP, cancelled=True)

        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessRecordsResultMessage) for m in published)
        server_metrics_messages = [
            m for m in published if isinstance(m, ProcessServerMetricsResultMessage)
        ]
        assert server_metrics_messages == []
