# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for ``RecordsManager``: plugin loaders, realtime metrics filtering,
and summarize-output bucketing.

Splits the records-manager plumbing into testable pure functions so the
service body stays focused on lifecycle / message dispatch. Loaders here
honour the ``accumulator`` / ``stream_exporter`` / ``analyzer`` plugin
categories.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Protocol

from aiperf.common.enums import CreditPhase, MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import PluginDisabled, PostProcessorDisabled
from aiperf.common.models import (
    ErrorDetails,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
    TimesliceResult,
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
    from aiperf.common.config import ServiceConfig, UserConfig
    from aiperf.common.models.branch_stats import BranchStats
    from aiperf.records.error_tracker import ErrorTracker
    from aiperf.records.records_tracker import RecordsTracker


class _LoaderHost(Protocol):
    """Minimal surface the plugin loaders use on the owning service."""

    service_id: str
    user_config: UserConfig
    service_config: ServiceConfig
    pub_client: Any

    def attach_child_lifecycle(self, child: Any) -> None: ...
    def debug(self, msg: Any) -> None: ...
    def error(self, msg: Any) -> None: ...


def load_accumulators(
    host: _LoaderHost,
) -> dict[AccumulatorType, AccumulatorProtocol]:
    """Instantiate all enabled ``ACCUMULATOR`` plugins for ``host``.

    ``MetricsAccumulator`` (registered as ``accumulator:metric_results``)
    owns the columnar inference-record store; GPU telemetry and server
    metrics get their own accumulators routed by plugin metadata
    ``record_types``.

    Disabled accumulators (``PluginDisabled`` / ``PostProcessorDisabled``)
    are silently skipped — that's the explicit opt-out path. Anything else
    is logged via ``host.error`` and skipped so one bad accumulator never
    aborts the whole records manager.
    """
    accumulators: dict[AccumulatorType, AccumulatorProtocol] = {}
    for entry in plugins.iter_entries(PluginType.ACCUMULATOR):
        try:
            AccumulatorClass = plugins.get_class(PluginType.ACCUMULATOR, entry.name)
            accumulator = AccumulatorClass(
                service_id=host.service_id,
                service_config=host.service_config,
                user_config=host.user_config,
                pub_client=host.pub_client,
            )
            host.attach_child_lifecycle(accumulator)
            accumulators[AccumulatorType(entry.name)] = accumulator
            host.debug(
                f"Created accumulator: {entry.name}: {accumulator.__class__.__name__}"
            )
        except (PluginDisabled, PostProcessorDisabled):
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
                service_config=host.service_config,
                user_config=host.user_config,
                pub_client=host.pub_client,
            )
            host.attach_child_lifecycle(exporter)
            exporters[StreamExporterType(entry.name)] = exporter
            host.debug(
                f"Created stream exporter: {entry.name}: {exporter.__class__.__name__}"
            )
        except (PluginDisabled, PostProcessorDisabled):
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
    analyzers raise ``PluginDisabled`` from their constructor and are
    silently skipped.
    """
    analyzers: dict[AnalyzerType, AnalyzerProtocol] = {}
    for entry in plugins.iter_entries(PluginType.ANALYZER):
        try:
            AnalyzerClass = plugins.get_class(PluginType.ANALYZER, entry.name)
            analyzer = AnalyzerClass(
                service_id=host.service_id,
                service_config=host.service_config,
                user_config=host.user_config,
                pub_client=host.pub_client,
            )
            analyzers[AnalyzerType(entry.name)] = analyzer
            host.debug(f"Created analyzer: {entry.name}: {analyzer.__class__.__name__}")
        except (PluginDisabled, PostProcessorDisabled):
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


async def generate_realtime_metrics(
    accumulators: list[AccumulatorProtocol],
    timeout: float = 30.0,
) -> list[MetricResult]:
    """Generate the real-time metrics for the profile run.

    Runs every accumulator's ``summarize`` in parallel with a short timeout
    and flattens the results to a single list of ``MetricResult``. Tolerates
    accumulators that return either ``AccumulatorMetricsSummary`` (with a
    ``.results`` dict-of-MetricResult) or a plain ``list[MetricResult]`` —
    GPU telemetry / server metrics accumulators return list shape.
    """
    results = await asyncio.gather(
        *[asyncio.wait_for(acc.summarize(), timeout=timeout) for acc in accumulators],
        return_exceptions=True,
    )
    flat: list[MetricResult] = []
    for result in results:
        if isinstance(result, BaseException):
            continue
        # AccumulatorMetricsSummary.results is dict[tag, MetricResult]
        results_attr = getattr(result, "results", None)
        if isinstance(results_attr, dict):
            flat.extend(v for v in results_attr.values() if isinstance(v, MetricResult))
        elif isinstance(result, list):
            flat.extend(r for r in result if isinstance(r, MetricResult))
    return flat


def filter_display_metrics(raw_metrics: list[MetricResult]) -> list[MetricResult]:
    """Filter out hidden metrics for realtime display.

    Drops anything flagged ``INTERNAL``, ``EXPERIMENTAL``, or ``ERROR_ONLY``,
    plus anything with ``console_group=NONE`` — matches the contract used by
    the dashboard's realtime view (``RealtimeMetricsDashboard.on_realtime_metrics``).

    Unregistered tags (plugin/external metrics without a ``MetricRegistry``
    entry) pass through unchanged so a third-party metric is still surfaced.
    """
    from aiperf.metrics.metric_registry import MetricRegistry, MetricTypeError

    hidden_flags = (
        MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL | MetricFlags.ERROR_ONLY
    )
    display_metrics: list[MetricResult] = []
    for m in raw_metrics:
        try:
            metric_cls = MetricRegistry.get_class(m.tag)
            if metric_cls.flags.has_any_flags(hidden_flags):
                continue
            if metric_cls.console_group == MetricConsoleGroup.NONE:
                continue
        except MetricTypeError:
            # Unregistered tag (plugin/external metric): include as-is
            pass
        display_metrics.append(m)
    return display_metrics


def build_process_records_result(
    *,
    records_results: list[MetricResult],
    warmup_records_results: list[MetricResult] | None = None,
    timeslices: list[TimesliceResult],
    error_results: list[ErrorDetails],
    tracker: RecordsTracker,
    error_tracker: ErrorTracker,
    cancelled: bool,
    branch_stats: BranchStats | None = None,
    context_overflow_count: int = 0,
) -> ProcessRecordsResult:
    """Assemble the final ``ProcessRecordsResult`` from accumulator output.

    Single-phase ``CreditPhase.PROFILING`` model — ``RecordsTracker`` does
    not expose a multi-phase ``get_results_phases`` /
    ``get_results_time_window`` API.
    """
    phase_stats = tracker.create_stats_for_phase(CreditPhase.PROFILING)
    return ProcessRecordsResult(
        results=ProfileResults(
            records=records_results,
            warmup_records=warmup_records_results or None,
            timeslices=timeslices or None,
            completed=len(records_results),
            start_ns=phase_stats.start_ns or time.time_ns(),
            end_ns=phase_stats.requests_end_ns or time.time_ns(),
            error_summary=error_tracker.get_error_summary_for_phase(
                CreditPhase.PROFILING
            ),
            was_cancelled=cancelled,
            branch_stats=branch_stats,
            context_overflow_count=context_overflow_count,
        ),
        errors=error_results,
    )


async def compute_analyzer_outputs(
    analyzers: dict[AnalyzerType, AnalyzerProtocol],
    summary_ctx: SummaryContext,
    *,
    log_error: Any | None = None,
    log_debug: Any | None = None,
) -> dict[AnalyzerType, Any]:
    """Run analyzers in dependency order, threading outputs through ``summary_ctx``.

    Each analyzer's result is recorded under ``summary_ctx.accumulator_outputs``
    keyed by ``str(analyzer_name)`` so downstream analyzers can read it via
    :meth:`SummaryContext.get_output`.

    An analyzer is skipped if its declared ``required_accumulators`` are not
    all present in ``summary_ctx.accumulators``. Disabled analyzers
    (``PluginDisabled``) are silently skipped; any other exception is logged
    via ``log_error`` (if provided) and the analyzer is omitted from the
    returned dict — a bad analyzer never aborts the rest.
    """
    outputs: dict[AnalyzerType, Any] = {}
    for analyzer_name, analyzer in analyzers.items():
        required: set[Any] | None = getattr(analyzer, "required_accumulators", None)
        if required is not None:
            available = set(summary_ctx.accumulators.keys()) | {
                str(k) for k in summary_ctx.accumulators
            }
            missing = {r for r in required if r not in available}
            if missing:
                if log_debug is not None:
                    log_debug(
                        f"Analyzer {analyzer_name} skipped: missing accumulators {missing}"
                    )
                continue
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
