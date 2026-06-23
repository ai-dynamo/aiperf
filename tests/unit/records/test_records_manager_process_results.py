# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``RecordsManager._process_results()``.

The ``accumulator`` plugin category drives metric summarization:
``MetricsAccumulator`` returns :class:`AccumulatorMetricsSummary`
(``results: dict[tag, MetricResult]``, ``timeslices``); GPU telemetry /
server metrics accumulators return list-shaped results.

The pipeline:

1. ``_summarize_all_accumulators`` runs ``summarize()`` on every loaded
   accumulator, buckets the output by shape, and accumulates errors.
2. ``_finalize_stream_exporters`` flushes JSONL writers concurrently.
3. ``build_process_records_result`` assembles a :class:`ProcessRecordsResult`.
4. ``ProcessRecordsResultMessage`` is published.
5. ``_run_analyzers`` runs every loaded :class:`AnalyzerProtocol` over a
   single :class:`SummaryContext`; output is published on
   :class:`ProcessAllResultsMessage`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.enums import CreditPhase, ProfileCancelReason
from aiperf.common.messages import (
    ProcessAllResultsMessage,
    ProcessRecordsResultMessage,
    ProfileCancelCommand,
)
from aiperf.common.models import (
    ErrorDetailsCount,
    MetricResult,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProfileResults,
    TimesliceResult,
)
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.plugin.enums import AccumulatorType, AnalyzerType, StreamExporterType
from aiperf.records.records_manager import RecordsManager

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
    """Stub for a legacy-shaped accumulator returning ``list[MetricResult]``."""
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


def _make_stub_analyzer(
    name: str,
    summarize_result: Any | None = None,
    summarize_exc: BaseException | None = None,
) -> MagicMock:
    a = MagicMock()
    a.__class__.__name__ = name
    if summarize_exc is not None:
        a.summarize = AsyncMock(side_effect=summarize_exc)
    else:
        a.summarize = AsyncMock(return_value=summarize_result or {"name": name})
    return a


def _make_manager_mock(
    *,
    accumulators: dict[AccumulatorType, MagicMock] | None = None,
    stream_exporters: dict[StreamExporterType, MagicMock] | None = None,
    analyzers: dict[AnalyzerType, MagicMock] | None = None,
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
    mgr._stream_exporters = stream_exporters or {}
    mgr._analyzers = analyzers or {}
    mgr._gpu_telemetry_accumulator = None
    mgr._server_metrics_accumulator = None

    # Records tracker — drives the time window via PROFILING phase stats.
    phase_stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        start_ns=start_ns,
        requests_end_ns=end_ns,
    )
    mgr._records_tracker.create_stats_for_phase.return_value = phase_stats

    # Error tracker — empty errors keep the success path.
    mgr._error_tracker.get_error_summary_for_phase.return_value = []

    # User config — disable telemetry / server-metrics side channels.
    mgr.user_config = MagicMock()
    mgr.user_config.gpu_telemetry_disabled = user_config_telemetry_disabled
    mgr.user_config.server_metrics_disabled = user_config_server_metrics_disabled

    # Logging
    mgr.debug = MagicMock()
    mgr.info = MagicMock()
    mgr.error = MagicMock()
    mgr.warning = MagicMock()
    mgr.exception = MagicMock()

    # Service identity + publish
    mgr.service_id = "test_records_manager"
    mgr.publish = AsyncMock()

    # Orchestrator branch_stats snapshot — agentx-side hook; default no DAG.
    mgr._snapshot_branch_stats = MagicMock(return_value=None)

    # Bind real methods
    mgr._process_results = RecordsManager._process_results.__get__(mgr)
    mgr._summarize_all_accumulators = (
        RecordsManager._summarize_all_accumulators.__get__(mgr)
    )
    mgr._summarize_one_accumulator = RecordsManager._summarize_one_accumulator.__get__(
        mgr
    )
    mgr._bucket_accumulator_summary = (
        RecordsManager._bucket_accumulator_summary.__get__(mgr)
    )
    mgr._finalize_stream_exporters = RecordsManager._finalize_stream_exporters.__get__(
        mgr
    )
    mgr._run_analyzers = RecordsManager._run_analyzers.__get__(mgr)
    mgr._publish_all_results = RecordsManager._publish_all_results.__get__(mgr)
    mgr._publish_telemetry_results = RecordsManager._publish_telemetry_results.__get__(
        mgr
    )
    mgr._publish_server_metrics_results = (
        RecordsManager._publish_server_metrics_results.__get__(mgr)
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

    @pytest.mark.asyncio
    async def test_cancelled_false_propagated_to_profile_results(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.results.was_cancelled is False

    @pytest.mark.asyncio
    async def test_cancelled_propagated_to_summary_context(self) -> None:
        """Analyzers see ``ctx.cancelled`` matching the call's cancelled flag."""
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        analyzer = _make_stub_analyzer("Analyzer1")
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            analyzers={AnalyzerType.ACCURACY_RESULTS: analyzer},
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=True)

        ctx: SummaryContext = analyzer.summarize.call_args[0][0]
        assert ctx.cancelled is True


# ---------------------------------------------------------------------------
# Tests: _finalize_stream_exporters integration
# ---------------------------------------------------------------------------


class TestProcessResultsStreamExporters:
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


class TestProcessResultsAnalyzers:
    """Analyzers run via ``_run_analyzers`` and have their outputs surfaced
    by the records-manager pipeline."""

    @pytest.mark.asyncio
    async def test_publishes_process_all_results_message(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        msg = _get_published_all_results(mgr)
        assert msg is not None

    @pytest.mark.asyncio
    async def test_no_analyzers_publishes_message(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            analyzers={},
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        msg = _get_published_all_results(mgr)
        assert msg is not None

    @pytest.mark.asyncio
    async def test_analyzer_failure_logged_and_skipped(self) -> None:
        """A failing analyzer logs but does not abort the message publish."""
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        failing = _make_stub_analyzer(
            "BrokenAnalyzer", summarize_exc=RuntimeError("analyze boom")
        )
        del failing.required_accumulators
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            analyzers={AnalyzerType.ACCURACY_RESULTS: failing},
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        # Error logged via mgr.error (compute_analyzer_outputs's policy)
        assert any("analyze boom" in str(c.args[0]) for c in mgr.error.call_args_list)
        msg = _get_published_all_results(mgr)
        assert msg is not None

    @pytest.mark.asyncio
    async def test_analyzer_receives_summary_context_with_accumulators(self) -> None:
        """Analyzers get a ``SummaryContext`` carrying the loaded accumulators."""
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        analyzer = _make_stub_analyzer("Analyzer")
        del analyzer.required_accumulators
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            analyzers={AnalyzerType.ACCURACY_RESULTS: analyzer},
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        ctx: SummaryContext = analyzer.summarize.call_args[0][0]
        assert isinstance(ctx, SummaryContext)
        assert ctx.accumulators[AccumulatorType.METRIC_RESULTS] is acc

    @pytest.mark.asyncio
    async def test_analyzer_summary_context_has_time_window(self) -> None:
        """``SummaryContext.start_ns`` / ``end_ns`` come from the records-tracker
        time window, mirrored on ``ProfileResults``."""
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        analyzer = _make_stub_analyzer("Analyzer")
        del analyzer.required_accumulators
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            analyzers={AnalyzerType.ACCURACY_RESULTS: analyzer},
            start_ns=42_000,
            end_ns=99_000,
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        ctx: SummaryContext = analyzer.summarize.call_args[0][0]
        assert ctx.start_ns == 42_000
        assert ctx.end_ns == 99_000


# ---------------------------------------------------------------------------
# Tests: _run_analyzers standalone semantics
# ---------------------------------------------------------------------------


class TestRunAnalyzers:
    """Direct tests on ``RecordsManager._run_analyzers``."""

    @pytest.mark.asyncio
    async def test_run_analyzers_with_no_analyzers_returns_empty(self) -> None:
        mgr = _make_manager_mock(analyzers={})
        result = ProcessRecordsResult(
            results=ProfileResults(records=None, completed=0, start_ns=0, end_ns=0)
        )

        outputs = await mgr._run_analyzers(result=result, cancelled=False)

        assert outputs == {}

    @pytest.mark.asyncio
    async def test_run_analyzers_returns_outputs_keyed_by_analyzer_type(self) -> None:
        analyzer = _make_stub_analyzer("Analyzer", summarize_result={"key": "value"})
        del analyzer.required_accumulators
        mgr = _make_manager_mock(analyzers={AnalyzerType.ACCURACY_RESULTS: analyzer})
        result = ProcessRecordsResult(
            results=ProfileResults(records=None, completed=0, start_ns=100, end_ns=200)
        )

        outputs = await mgr._run_analyzers(result=result, cancelled=False)

        assert outputs == {AnalyzerType.ACCURACY_RESULTS: {"key": "value"}}


class TestProfileCancelFinalizes:
    """A ProfileCancelCommand must ALWAYS finalize (produce a
    ProcessRecordsResultMessage), even when only WARMUP ran and PROFILING never
    started. The absence of this finalization is exactly what hung the run on a
    warmup-failure abort, so pin it directly.
    """

    @pytest.mark.asyncio
    async def test_profile_cancel_publishes_result_and_marks_cancelled(self) -> None:
        mgr = _make_manager_mock()
        mgr._on_profile_cancel_command = (
            RecordsManager._on_profile_cancel_command.__get__(mgr)
        )

        result = await mgr._on_profile_cancel_command(
            ProfileCancelCommand(
                service_id="timing", reason=ProfileCancelReason.WARMUP_FAILURE
            )
        )

        # Finalization happened: a result is returned AND a result message is
        # published (the signal the system controller waits on to shut down).
        assert isinstance(result, ProcessRecordsResult)
        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessRecordsResultMessage) for m in published)
        mgr._records_tracker.mark_phase_cancelled.assert_called_once_with(
            CreditPhase.PROFILING
        )


# Reference imports kept so static-analysis sees the protocol surface used
# by the SummaryContext assertions above.
_ = ErrorDetailsCount
