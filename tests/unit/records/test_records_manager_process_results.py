# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8s ``RecordsManager._process_results()``.

Adapted from the metrics-accumulator branch's tests for the source-side
unified accumulator export pipeline. The K8s shape is different: the
legacy ``_metric_results_processors`` summarize/finalize fan-out runs as
the primary path (publishing ``ProcessRecordsResultMessage``), and the
new accumulator/analyzer pipeline runs *after* via
:meth:`RecordsManager._finalize_stream_exporters` and
:meth:`RecordsManager._run_analyzers` and publishes the unified
``ProcessAllResultsMessage`` with the analyzer outputs.

Key K8s shape differences from the source branch (preserved with skips):

* ``_process_results(cancelled: bool)`` — no ``phase=`` kwarg; the
  results-tracker phase window is built internally from
  ``get_results_phases()`` / ``get_results_time_window()``.
* No ``ExporterManager`` integration in ``_process_results`` — that lives
  on the SystemController side. ``ProcessAllResultsMessage`` does not
  carry ``exported_artifacts`` populated from this path.
* No per-accumulator ``ExportContext`` construction —
  ``compute_analyzer_outputs`` builds a single ``SummaryContext`` for the
  whole analyzer run.
* ``ProcessAllResultsMessage`` carries ``steady_state_results`` and
  ``energy_efficiency_results`` (analyzer outputs), not
  per-accumulator ``telemetry_results``/``server_metrics_results``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.messages import (
    ProcessAllResultsMessage,
    ProcessRecordsResultMessage,
)
from aiperf.common.models import (
    ErrorDetailsCount,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)
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


def _make_legacy_processor(
    summarize_result: list[MetricResult] | None = None,
    summarize_exc: BaseException | None = None,
) -> MagicMock:
    """Stub for the legacy ``ResultsProcessorProtocol`` — owned by
    ``self._metric_results_processors`` in ``_process_results``."""
    proc = MagicMock()
    proc.__class__.__name__ = "StubLegacyProcessor"
    if summarize_exc is not None:
        proc.summarize = AsyncMock(side_effect=summarize_exc)
    else:
        proc.summarize = AsyncMock(
            return_value=summarize_result
            if summarize_result is not None
            else [_STUB_METRIC_RESULT]
        )
    proc.finalize = AsyncMock()
    return proc


def _make_stub_accumulator(summarize_result: Any | None = None) -> MagicMock:
    """Stub accumulator — only needs to be present in ``self._accumulators``;
    its surface is exercised through ``SummaryContext.accumulators`` by
    analyzers."""
    acc = MagicMock()
    acc.__class__.__name__ = "StubAccumulator"
    acc.summarize = AsyncMock(return_value=summarize_result)
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
    legacy_processors: list[MagicMock] | None = None,
    accumulators: dict[AccumulatorType, MagicMock] | None = None,
    stream_exporters: dict[StreamExporterType, MagicMock] | None = None,
    analyzers: dict[AnalyzerType, MagicMock] | None = None,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
    results_phases: list[str] | None = None,
) -> MagicMock:
    """Build a mock ``RecordsManager`` with ``_process_results`` /
    ``_finalize_stream_exporters`` / ``_run_analyzers`` bound.

    Mirrors the source-branch helper but adapts to K8s's shape: there's
    no ``phase=`` kwarg, and we wire the records-tracker / error-tracker
    minimally so ``build_process_records_result`` succeeds.
    """
    mgr = MagicMock()
    mgr._metric_results_processors = legacy_processors or []
    mgr._accumulators = accumulators or {}
    mgr._stream_exporters = stream_exporters or {}
    mgr._analyzers = analyzers or {}

    # Records tracker — drives the time window in build_process_records_result.
    mgr._records_tracker.get_results_time_window.return_value = (start_ns, end_ns)
    mgr._records_tracker.get_results_phases.return_value = results_phases or [
        "profiling"
    ]

    # Error tracker — empty errors keep the success path.
    mgr._error_tracker.get_error_summary_for_phase.return_value = []

    # Logging
    mgr.debug = MagicMock()
    mgr.info = MagicMock()
    mgr.error = MagicMock()
    mgr.warning = MagicMock()
    mgr.exception = MagicMock()

    # Service identity + publish
    mgr.service_id = "test_records_manager"
    mgr.publish = AsyncMock()

    # Bind real methods
    mgr._process_results = RecordsManager._process_results.__get__(mgr)
    mgr._summarize_all_processors = RecordsManager._summarize_all_processors.__get__(
        mgr
    )
    mgr._finalize_all_processors = RecordsManager._finalize_all_processors.__get__(mgr)
    mgr._build_records_result = RecordsManager._build_records_result.__get__(mgr)
    mgr._publish_all_results = RecordsManager._publish_all_results.__get__(mgr)
    mgr._finalize_stream_exporters = RecordsManager._finalize_stream_exporters.__get__(
        mgr
    )
    mgr._run_analyzers = RecordsManager._run_analyzers.__get__(mgr)

    # Telemetry publish runs inside _process_results. These tests exercise the
    # inference/analyzer path, so stub the GPU-telemetry side: no accumulator
    # means the real publisher emits a results=None message harmlessly, but the
    # tests don't assert on it, so an AsyncMock keeps them focused.
    mgr._publish_telemetry_results = AsyncMock()

    return mgr


# ---------------------------------------------------------------------------
# Tests: legacy ``_metric_results_processors`` path
# ---------------------------------------------------------------------------


class TestProcessResultsLegacyPath:
    """The K8s ``_process_results`` first runs the legacy results-processor
    fan-out (``summarize`` + ``finalize``) and publishes
    ``ProcessRecordsResultMessage`` from those bucketed results."""

    @pytest.mark.asyncio
    async def test_calls_summarize_and_finalize_on_all_processors(self) -> None:
        proc1 = _make_legacy_processor([_STUB_METRIC_RESULT])
        proc2 = _make_legacy_processor([])

        mgr = _make_manager_mock(legacy_processors=[proc1, proc2])

        await mgr._process_results(cancelled=False)

        proc1.summarize.assert_awaited_once()
        proc2.summarize.assert_awaited_once()
        proc1.finalize.assert_awaited_once()
        proc2.finalize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publishes_process_records_result_message(self) -> None:
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(legacy_processors=[proc])

        await mgr._process_results(cancelled=False)

        # Both legacy and unified messages get published.
        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessRecordsResultMessage) for m in published)

    @pytest.mark.asyncio
    async def test_returns_process_records_result(self) -> None:
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(legacy_processors=[proc])

        result = await mgr._process_results(cancelled=False)

        assert isinstance(result, ProcessRecordsResult)
        assert result.results.records is not None
        assert _STUB_METRIC_RESULT in result.results.records

    @pytest.mark.asyncio
    async def test_processor_summarize_failure_does_not_abort(self) -> None:
        """A failing summarize is wrapped into ``result.errors`` but the
        unified pipeline still runs."""
        failing = _make_legacy_processor(summarize_exc=RuntimeError("summarize boom"))
        mgr = _make_manager_mock(legacy_processors=[failing])

        result = await mgr._process_results(cancelled=False)

        # Errors logged + included in result.errors
        mgr.error.assert_called()
        assert any("summarize boom" in str(err.message or err) for err in result.errors)

    @pytest.mark.asyncio
    async def test_processor_finalize_failure_logged(self) -> None:
        failing = _make_legacy_processor([_STUB_METRIC_RESULT])
        failing.finalize.side_effect = RuntimeError("flush failed")
        mgr = _make_manager_mock(legacy_processors=[failing])

        await mgr._process_results(cancelled=False)

        # Logged via mgr.error (per K8s _process_results contract); the
        # legacy path swallows finalize failures so the unified path runs.
        assert any("flush failed" in str(c.args[0]) for c in mgr.error.call_args_list)

    @pytest.mark.asyncio
    async def test_empty_legacy_processors_produces_empty_records(self) -> None:
        mgr = _make_manager_mock(legacy_processors=[])

        result = await mgr._process_results(cancelled=False)

        assert isinstance(result, ProcessRecordsResult)
        assert result.results.records == []


# ---------------------------------------------------------------------------
# Tests: cancelled flag propagation
# ---------------------------------------------------------------------------


class TestProcessResultsCancelled:
    @pytest.mark.asyncio
    async def test_cancelled_true_propagated_to_profile_results(self) -> None:
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(legacy_processors=[proc])

        result = await mgr._process_results(cancelled=True)

        assert result.results.was_cancelled is True

    @pytest.mark.asyncio
    async def test_cancelled_false_propagated_to_profile_results(self) -> None:
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(legacy_processors=[proc])

        result = await mgr._process_results(cancelled=False)

        assert result.results.was_cancelled is False

    @pytest.mark.asyncio
    async def test_cancelled_propagated_to_summary_context(self) -> None:
        """Analyzers see ``ctx.cancelled`` matching the call's cancelled flag."""
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        analyzer = _make_stub_analyzer("Analyzer1")
        mgr = _make_manager_mock(
            legacy_processors=[proc],
            analyzers={AnalyzerType.STEADY_STATE: analyzer},
        )

        await mgr._process_results(cancelled=True)

        ctx: SummaryContext = analyzer.summarize.call_args[0][0]
        assert ctx.cancelled is True


# ---------------------------------------------------------------------------
# Tests: ``_finalize_stream_exporters`` integration
# ---------------------------------------------------------------------------


class TestProcessResultsStreamExporters:
    @pytest.mark.asyncio
    async def test_stream_exporters_finalized(self) -> None:
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        exp = _make_stub_stream_exporter()
        mgr = _make_manager_mock(
            legacy_processors=[proc],
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )

        await mgr._process_results(cancelled=False)

        exp.finalize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_stream_exporters_is_noop(self) -> None:
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(legacy_processors=[proc], stream_exporters={})

        await mgr._process_results(cancelled=False)

        # Process completes successfully and publishes both messages.
        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessAllResultsMessage) for m in published)


# ---------------------------------------------------------------------------
# Tests: analyzer execution and ``ProcessAllResultsMessage`` publish
# ---------------------------------------------------------------------------


def _get_published_all_results(mgr: MagicMock) -> ProcessAllResultsMessage | None:
    """Return the published ``ProcessAllResultsMessage`` if any."""
    for call in mgr.publish.await_args_list:
        msg = call.args[0]
        if isinstance(msg, ProcessAllResultsMessage):
            return msg
    return None


class TestProcessResultsAnalyzers:
    """Analyzers run via ``_run_analyzers`` and surface in
    ``ProcessAllResultsMessage.steady_state_results`` /
    ``energy_efficiency_results``."""

    @pytest.mark.asyncio
    async def test_publishes_process_all_results_message(self) -> None:
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(legacy_processors=[proc])

        await mgr._process_results(cancelled=False)

        msg = _get_published_all_results(mgr)
        assert msg is not None

    @pytest.mark.asyncio
    async def test_steady_state_analyzer_output_attached(self) -> None:
        """Output of the STEADY_STATE analyzer surfaces on the unified message."""
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        ss_output = {"window_start": 100, "window_end": 200}
        analyzer = _make_stub_analyzer(
            "SteadyStateAnalyzer", summarize_result=ss_output
        )
        mgr = _make_manager_mock(
            legacy_processors=[proc],
            analyzers={AnalyzerType.STEADY_STATE: analyzer},
        )

        await mgr._process_results(cancelled=False)

        msg = _get_published_all_results(mgr)
        assert msg is not None
        assert msg.steady_state_results == ss_output

    @pytest.mark.asyncio
    async def test_no_analyzers_publishes_message_with_none_outputs(self) -> None:
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(legacy_processors=[proc], analyzers={})

        await mgr._process_results(cancelled=False)

        msg = _get_published_all_results(mgr)
        assert msg is not None
        assert msg.steady_state_results is None
        assert msg.energy_efficiency_results is None

    @pytest.mark.asyncio
    async def test_analyzer_failure_logged_and_skipped(self) -> None:
        """A failing analyzer logs but does not abort the message publish."""
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        failing = _make_stub_analyzer(
            "BrokenAnalyzer", summarize_exc=RuntimeError("analyze boom")
        )
        mgr = _make_manager_mock(
            legacy_processors=[proc],
            analyzers={AnalyzerType.STEADY_STATE: failing},
        )

        await mgr._process_results(cancelled=False)

        # Error logged via mgr.error (compute_analyzer_outputs's policy)
        assert any("analyze boom" in str(c.args[0]) for c in mgr.error.call_args_list)
        msg = _get_published_all_results(mgr)
        assert msg is not None
        # Failed analyzer not present in outputs
        assert msg.steady_state_results is None

    @pytest.mark.asyncio
    async def test_analyzer_receives_summary_context_with_accumulators(self) -> None:
        """Analyzers get a ``SummaryContext`` carrying the loaded accumulators."""
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        acc = _make_stub_accumulator()
        analyzer = _make_stub_analyzer("Analyzer")
        mgr = _make_manager_mock(
            legacy_processors=[proc],
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            analyzers={AnalyzerType.STEADY_STATE: analyzer},
        )

        await mgr._process_results(cancelled=False)

        ctx: SummaryContext = analyzer.summarize.call_args[0][0]
        assert isinstance(ctx, SummaryContext)
        assert ctx.accumulators[AccumulatorType.METRIC_RESULTS] is acc

    @pytest.mark.asyncio
    async def test_analyzer_summary_context_has_time_window(self) -> None:
        """``SummaryContext.start_ns`` / ``end_ns`` come from the records-tracker
        time window, mirrored on ``ProfileResults``."""
        proc = _make_legacy_processor([_STUB_METRIC_RESULT])
        analyzer = _make_stub_analyzer("Analyzer")
        mgr = _make_manager_mock(
            legacy_processors=[proc],
            analyzers={AnalyzerType.STEADY_STATE: analyzer},
            start_ns=42_000,
            end_ns=99_000,
        )

        await mgr._process_results(cancelled=False)

        ctx: SummaryContext = analyzer.summarize.call_args[0][0]
        assert ctx.start_ns == 42_000
        assert ctx.end_ns == 99_000


# ---------------------------------------------------------------------------
# Tests: ``_run_analyzers`` standalone semantics
# ---------------------------------------------------------------------------


class TestRunAnalyzers:
    """Direct tests on ``RecordsManager._run_analyzers``."""

    @pytest.mark.asyncio
    async def test_run_analyzers_with_no_analyzers_returns_empty(self) -> None:
        mgr = _make_manager_mock(analyzers={})
        result = ProcessRecordsResult(
            results=ProfileResults(completed=0, start_ns=0, end_ns=0)
        )

        outputs = await mgr._run_analyzers(result=result, cancelled=False)

        assert outputs == {}

    @pytest.mark.asyncio
    async def test_run_analyzers_returns_outputs_keyed_by_analyzer_type(self) -> None:
        analyzer = _make_stub_analyzer("Analyzer", summarize_result={"key": "value"})
        mgr = _make_manager_mock(analyzers={AnalyzerType.STEADY_STATE: analyzer})
        result = ProcessRecordsResult(
            results=ProfileResults(completed=0, start_ns=100, end_ns=200)
        )

        outputs = await mgr._run_analyzers(result=result, cancelled=False)

        assert outputs == {AnalyzerType.STEADY_STATE: {"key": "value"}}


# ---------------------------------------------------------------------------
# Tests: source-branch behavior we deliberately don't test (skips preserve intent)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="k8s _process_results does not call accumulator.export_results. "
    "The accumulator path runs per-record via _send_record_to_accumulators; "
    "summarization happens inside analyzers via SummaryContext.accumulators."
)
def test_calls_export_results_on_all_accumulators_source_only() -> None:
    """Source branch built per-accumulator ``ExportContext`` and called
    ``acc.export_results(ctx)`` in ``_process_results``. K8s does not — see
    ``TestProcessResultsAnalyzers`` for the SummaryContext-based equivalent."""


@pytest.mark.skip(
    reason="k8s does not construct ExporterManager inside _process_results. "
    "ProcessAllResultsMessage carries no exported_artifacts populated from "
    "this path; that's owned by the SystemController / ExporterManager flow."
)
def test_message_contains_exported_artifacts_source_only() -> None:
    """Source branch's ``ExporterManager.exported_file_infos`` was attached to
    ``ProcessAllResultsMessage.exported_artifacts``. K8s leaves that field at
    its default empty dict — exporter integration lives elsewhere."""


@pytest.mark.skip(
    reason="k8s ProcessAllResultsMessage carries telemetry_results / "
    "server_metrics_results as default None; they are populated by other "
    "side-channel pipelines (gpu_telemetry_processor / "
    "server_metrics_processor categories), not by RecordsManager._process_results."
)
def test_message_contains_typed_results_source_only() -> None:
    """Source branch extracted ``TelemetryExportData`` / ``ServerMetricsResults``
    from accumulator outputs and attached them to ``ProcessAllResultsMessage``.
    K8s routes telemetry/server-metrics records via separate top-level
    ``gpu_telemetry_processor`` / ``server_metrics_processor`` plugin
    categories that bypass ``RecordsManager`` entirely."""


@pytest.mark.skip(
    reason="k8s _process_results has no phase=... kwarg; the phase window "
    "is built internally from records_tracker.get_results_phases() / "
    "get_results_time_window(). See test_analyzer_summary_context_has_time_window."
)
def test_per_accumulator_export_context_source_only() -> None:
    """Source branch built per-accumulator ``ExportContext`` with different
    time windows and error sources (GPU_TELEMETRY: no end_ns; SERVER_METRICS:
    fallback timestamps; others: phase window). K8s does not — analyzer
    contexts share one ``SummaryContext`` whose time window comes from the
    records-tracker."""


# Reference imports kept so static-analysis sees the protocol surface used
# by the SummaryContext assertions above.
_ = ErrorDetailsCount
