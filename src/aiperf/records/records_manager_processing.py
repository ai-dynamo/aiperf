# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for RecordsManager: realtime metrics filtering and results bucketing."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Protocol

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import PluginDisabled, PostProcessorDisabled
from aiperf.common.models import (
    ErrorDetails,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    AccumulatorType,
    AnalyzerType,
    PluginType,
    StreamExporterType,
)

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import (
        AccumulatorProtocol,
        AnalyzerProtocol,
        StreamExporterProtocol,
        SummaryContext,
    )
    from aiperf.common.models import BranchStats
    from aiperf.config import BenchmarkRun
    from aiperf.network_latency.accumulator import NetworkLatencyAccumulator
    from aiperf.network_latency.protocols import NetworkLatencyProcessorProtocol
    from aiperf.post_processors.protocols import ResultsProcessorProtocol
    from aiperf.records.error_tracker import ErrorTracker
    from aiperf.records.records_tracker import RecordsTracker


class _LoaderHost(Protocol):
    """Minimal surface the processor loader uses on the owning service."""

    service_id: str
    run: BenchmarkRun
    pub_client: Any

    def attach_child_lifecycle(self, child: Any) -> None: ...
    def debug(self, msg: Any) -> None: ...
    def error(self, msg: Any) -> None: ...


def _has_results_processor_methods(processor: Any) -> bool:
    return all(
        callable(getattr(processor, method, None))
        for method in ("process_result", "summarize", "finalize")
    )


def load_results_processors(host: _LoaderHost) -> list[ResultsProcessorProtocol]:
    """Instantiate all enabled ``RESULTS_PROCESSOR`` plugins for ``host``.

    One bad processor must not abort the whole records manager, so individual
    construction failures are logged and skipped.
    """
    processors: list[ResultsProcessorProtocol] = []
    for entry in plugins.iter_entries(PluginType.RESULTS_PROCESSOR):
        try:
            ProcessorClass = plugins.get_class(PluginType.RESULTS_PROCESSOR, entry.name)
            results_processor = ProcessorClass(
                service_id=host.service_id,
                run=host.run,
                pub_client=host.pub_client,
            )
            if not _has_results_processor_methods(results_processor):
                host.debug(
                    f"Results processor {entry.name} does not implement the results protocol; skipping"
                )
                continue
            host.attach_child_lifecycle(results_processor)
            processors.append(results_processor)
            host.debug(
                f"Created results processor: {entry.name}: {results_processor.__class__.__name__}"
            )
        except PostProcessorDisabled:
            host.debug(
                f"Results processor {entry.name} is disabled and will not be used"
            )
        except Exception as e:  # noqa: BLE001 - one bad results processor must not abort the whole records manager; error is surfaced via host.error
            host.error(f"Failed to create results processor {entry.name}: {e}")
    return processors


def make_network_latency_accumulator(
    host: _LoaderHost,
) -> NetworkLatencyAccumulator | None:
    """Build the in-process RTT-sample accumulator, or None when probing is off.

    Returns None unless ``network_latency.should_probe`` is set, mirroring the
    other run-dependent loaders so RecordsManager.__init__ never touches
    ``host.run`` directly (keeps __init__ patchable in unit tests).
    """
    from aiperf.network_latency.accumulator import NetworkLatencyAccumulator

    if not host.run.cfg.network_latency.should_probe:
        return None
    return NetworkLatencyAccumulator(benchmark_id=host.run.benchmark_id)


def load_network_latency_processors(
    host: _LoaderHost,
) -> list[NetworkLatencyProcessorProtocol]:
    """Instantiate enabled ``RESULTS_PROCESSOR`` plugins that consume RTT samples.

    Network latency processors (e.g. ``NetworkLatencyJSONLWriter``) implement
    ``process_network_latency_sample`` rather than the metric-record
    ``process_result`` contract, so ``load_results_processors`` skips them. They
    self-disable via ``PostProcessorDisabled`` unless network latency probing is
    active, so on a normal run this returns an empty list. One bad processor must
    not abort the whole records manager.
    """
    from aiperf.network_latency.protocols import NetworkLatencyProcessorProtocol

    processors: list[NetworkLatencyProcessorProtocol] = []
    for entry in plugins.iter_entries(PluginType.RESULTS_PROCESSOR):
        try:
            ProcessorClass = plugins.get_class(PluginType.RESULTS_PROCESSOR, entry.name)
            results_processor = ProcessorClass(
                service_id=host.service_id,
                run=host.run,
                pub_client=host.pub_client,
            )
            if not isinstance(results_processor, NetworkLatencyProcessorProtocol):
                continue
            host.attach_child_lifecycle(results_processor)
            processors.append(results_processor)
            host.debug(
                f"Created network latency processor: {entry.name}: {results_processor.__class__.__name__}"
            )
        except PostProcessorDisabled:
            host.debug(
                f"Network latency processor {entry.name} is disabled and will not be used"
            )
        except Exception as e:  # noqa: BLE001 - one bad results processor must not abort the whole records manager; error is surfaced via host.error
            host.error(f"Failed to create network latency processor {entry.name}: {e}")
    return processors


async def generate_realtime_metrics(
    processors: list[ResultsProcessorProtocol],
    timeout: float = 30.0,
) -> list[MetricResult]:
    """Generate the real-time metrics for the profile run.

    Runs every processor's ``summarize`` in parallel with a short timeout and
    flattens the list-of-lists into a single list of ``MetricResult``.
    """
    results = await asyncio.gather(
        *[
            asyncio.wait_for(processor.summarize(), timeout=timeout)
            for processor in processors
        ],
        return_exceptions=True,
    )
    return [
        res
        for result in results
        if isinstance(result, list)
        for res in result
        if isinstance(res, MetricResult)
    ]


def filter_display_metrics(raw_metrics: list[MetricResult]) -> list[MetricResult]:
    """Filter out hidden metrics (INTERNAL/EXPERIMENTAL) for realtime display.

    Unregistered tags (plugin/external metrics) pass through unchanged.
    """
    from aiperf.metrics.metric_registry import MetricRegistry, MetricTypeError

    hidden_flags = MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL
    display_metrics: list[MetricResult] = []
    for m in raw_metrics:
        try:
            metric_cls = MetricRegistry.get_class(m.tag)
            if metric_cls.flags.has_any_flags(hidden_flags):
                continue
        except MetricTypeError:
            # Unregistered tag (plugin/external metric): include it in output as-is
            pass
        display_metrics.append(m)
    return display_metrics


def bucket_summarize_results(
    results: list[object],
) -> tuple[list[MetricResult], list, list[ErrorDetails], list[BaseException]]:
    """Sort ``asyncio.gather(return_exceptions=True)`` output by kind.

    Returns (records_results, timeslice_metric_results, error_results,
    raw_exceptions). Raw exceptions are returned separately so the caller can
    log them before wrapping into ``ErrorDetails``. ``timeslice_metric_results``
    is a list of per-slice metric lists in chronological order; an empty list
    if no timeslice processor produced results.
    """
    records_results: list[MetricResult] = []
    timeslice_metric_results: list = []
    error_results: list[ErrorDetails] = []
    raw_exceptions: list[BaseException] = []
    for result in results:
        if isinstance(result, list):
            # Timeslice payload is list of per-slice dicts; flat list of
            # MetricResult is the records-results bucket.
            if result and isinstance(result[0], dict):
                timeslice_metric_results = result
            else:
                records_results.extend(result)
        elif isinstance(result, ErrorDetails):
            error_results.append(result)
        elif isinstance(result, BaseException):
            raw_exceptions.append(result)
    return records_results, timeslice_metric_results, error_results, raw_exceptions


def build_process_records_result(
    *,
    records_results: list[MetricResult],
    timeslice_metric_results: list,
    error_results: list[ErrorDetails],
    tracker: RecordsTracker,
    error_tracker: ErrorTracker,
    cancelled: bool,
    multi_turn_ttft_trend: dict[int, MetricResult] | None = None,
    branch_stats: BranchStats | None = None,
) -> ProcessRecordsResult:
    """Assemble the final ``ProcessRecordsResult`` from bucketed summarize output."""
    start_ns, end_ns = tracker.get_results_time_window()
    error_summary: list = []
    for phase in tracker.get_results_phases():
        error_summary.extend(error_tracker.get_error_summary_for_phase(phase))
    return ProcessRecordsResult(
        results=ProfileResults(
            records=records_results,
            timeslice_metric_results=timeslice_metric_results or None,
            multi_turn_ttft_trend=multi_turn_ttft_trend,
            completed=len(records_results),
            start_ns=start_ns or time.time_ns(),
            end_ns=end_ns or time.time_ns(),
            error_summary=error_summary,
            was_cancelled=cancelled,
            branch_stats=branch_stats,
        ),
        errors=error_results,
    )


def load_accumulators(
    host: _LoaderHost,
) -> dict[AccumulatorType, AccumulatorProtocol]:
    """Instantiate all enabled ``ACCUMULATOR`` plugins for ``host``.

    Mirrors :func:`load_results_processors` but for the new accumulator plugin
    category introduced by the metrics-accumulator branch. K8s keeps the legacy
    ``RESULTS_PROCESSOR`` pipeline running in parallel; both populate from the
    same record stream so analyzers (steady-state, energy efficiency) have a
    columnar source while the existing exporters keep working unchanged.

    One disabled / failed accumulator must not abort the records manager;
    ``PluginDisabled`` is the explicit opt-out path, anything else is logged
    via ``host.error`` and skipped.
    """
    accumulators: dict[AccumulatorType, AccumulatorProtocol] = {}
    for entry in plugins.iter_entries(PluginType.ACCUMULATOR):
        try:
            AccumulatorClass = plugins.get_class(PluginType.ACCUMULATOR, entry.name)
            accumulator = AccumulatorClass(
                service_id=host.service_id,
                run=host.run,
                pub_client=host.pub_client,
            )
            host.attach_child_lifecycle(accumulator)
            accumulators[AccumulatorType(entry.name)] = accumulator
            host.debug(
                f"Created accumulator: {entry.name}: {accumulator.__class__.__name__}"
            )
        except PluginDisabled:
            host.debug(f"Accumulator {entry.name} is disabled and will not be used")
        except Exception as e:  # noqa: BLE001 - one bad accumulator must not abort the records manager
            host.error(f"Failed to create accumulator {entry.name}: {e}")
    return accumulators


def load_stream_exporters(
    host: _LoaderHost,
) -> dict[StreamExporterType, StreamExporterProtocol]:
    """Instantiate all enabled ``STREAM_EXPORTER`` plugins for ``host``.

    Stream exporters write each record to an external sink (JSONL, etc.) as
    it arrives; they are flushed via :meth:`StreamExporterProtocol.finalize`
    after all records are processed. Same disable/error policy as
    :func:`load_accumulators`.
    """
    exporters: dict[StreamExporterType, StreamExporterProtocol] = {}
    for entry in plugins.iter_entries(PluginType.STREAM_EXPORTER):
        try:
            ExporterClass = plugins.get_class(PluginType.STREAM_EXPORTER, entry.name)
            exporter = ExporterClass(
                service_id=host.service_id,
                run=host.run,
                pub_client=host.pub_client,
            )
            host.attach_child_lifecycle(exporter)
            exporters[StreamExporterType(entry.name)] = exporter
            host.debug(
                f"Created stream exporter: {entry.name}: {exporter.__class__.__name__}"
            )
        except PluginDisabled:
            host.debug(f"Stream exporter {entry.name} is disabled and will not be used")
        except Exception as e:  # noqa: BLE001 - one bad exporter must not abort the records manager
            host.error(f"Failed to create stream exporter {entry.name}: {e}")
    return exporters


def load_analyzers(
    host: _LoaderHost,
) -> dict[AnalyzerType, AnalyzerProtocol]:
    """Instantiate all enabled ``ANALYZER`` plugins for ``host``.

    Analyzers do not ingest records — they read from already-populated
    accumulators in :class:`SummaryContext` at summarize time. Disabled
    analyzers (e.g. ``SteadyStateAnalyzer`` when ``--steady-state`` is off)
    raise ``PluginDisabled`` from their constructor and are silently skipped.

    Cross-input analyzers (e.g. energy efficiency, which needs both GPU
    telemetry and inference records) are NOT loaded here — they run
    controller-side as plain functions because their accumulator dependencies
    live in separate processes. See
    ``docs/superpowers/specs/2026-05-02-cross-input-analyzers-design.md``.
    """
    analyzers: dict[AnalyzerType, AnalyzerProtocol] = {}
    for entry in plugins.iter_entries(PluginType.ANALYZER):
        try:
            AnalyzerClass = plugins.get_class(PluginType.ANALYZER, entry.name)
            # Analyzers in the source branch take ``user_config: UserConfig``;
            # accumulators take ``run: BenchmarkRun``. Pass both as kwargs so
            # whichever signature the analyzer ports to (sibling work) keeps
            # working — ``**kwargs: Any`` swallows the unused argument.
            analyzer = AnalyzerClass(
                run=host.run,
                user_config=getattr(host, "user_config", None),
            )
            analyzers[AnalyzerType(entry.name)] = analyzer
            host.debug(f"Created analyzer: {entry.name}: {analyzer.__class__.__name__}")
        except PluginDisabled:
            host.debug(f"Analyzer {entry.name} is disabled and will not be used")
        except Exception as e:  # noqa: BLE001 - one bad analyzer must not abort the records manager
            host.error(f"Failed to create analyzer {entry.name}: {e}")
    return analyzers


def accumulators_for_record_type(
    accumulators: dict[AccumulatorType, AccumulatorProtocol],
    record_type: str,
) -> list[AccumulatorProtocol]:
    """Return accumulators whose plugin metadata declares ``record_type``."""
    matched: list[AccumulatorProtocol] = []
    for entry in plugins.iter_entries(PluginType.ACCUMULATOR):
        record_types = entry.metadata.get("record_types", []) if entry.metadata else []
        if record_type not in record_types:
            continue
        acc_type = AccumulatorType(entry.name)
        if acc_type in accumulators:
            matched.append(accumulators[acc_type])
    return matched


def stream_exporters_for_record_type(
    exporters: dict[StreamExporterType, StreamExporterProtocol],
    record_type: str,
) -> list[StreamExporterProtocol]:
    """Return stream exporters whose plugin metadata declares ``record_type``."""
    matched: list[StreamExporterProtocol] = []
    for entry in plugins.iter_entries(PluginType.STREAM_EXPORTER):
        record_types = entry.metadata.get("record_types", []) if entry.metadata else []
        if record_type not in record_types:
            continue
        exp_type = StreamExporterType(entry.name)
        if exp_type in exporters:
            matched.append(exporters[exp_type])
    return matched


async def compute_analyzer_outputs(
    analyzers: dict[AnalyzerType, AnalyzerProtocol],
    summary_ctx: SummaryContext,
    *,
    log_error: Any | None = None,
    log_debug: Any | None = None,
) -> dict[AnalyzerType, Any]:
    """Run analyzers in dependency order, threading outputs through ``summary_ctx``.

    Each analyzer's result is recorded under ``summary_ctx.accumulator_outputs``
    keyed by ``str(analyzer_name)`` so downstream analyzers (e.g. energy
    efficiency depending on metric_results) can read it via
    :meth:`SummaryContext.get_output`.

    Disabled analyzers (``PluginDisabled``) are silently skipped; any other
    exception is logged via ``log_error`` (if provided) and the analyzer is
    omitted from the returned dict. A bad analyzer never aborts the rest.
    """
    outputs: dict[AnalyzerType, Any] = {}
    for analyzer_name, analyzer in analyzers.items():
        try:
            result = await analyzer.summarize(summary_ctx)
            outputs[analyzer_name] = result
            summary_ctx.accumulator_outputs[str(analyzer_name)] = result
        except PluginDisabled as e:
            if log_debug is not None:
                log_debug(f"Analyzer {analyzer_name} disabled: {e}")
        except Exception as e:  # noqa: BLE001 - one bad analyzer must not abort the rest
            if log_error is not None:
                log_error(f"Analyzer {analyzer_name} failed: {e!r}")
    return outputs
