# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for ``RecordsManager``: plugin loaders, realtime metrics filtering,
and summarize-output bucketing.

Splits the records-manager plumbing into testable pure functions so the
service body stays focused on lifecycle / message dispatch. Loaders here
honour the ``accumulator`` / ``stream_exporter`` plugin
categories.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.enums import CreditPhase, MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import PluginDisabled, PostProcessorDisabled
from aiperf.common.logging import AIPerfLogger
from aiperf.common.models import MetricResult
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    AccumulatorType,
    PluginType,
    StreamExporterType,
)

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import (
        AccumulatorProtocol,
        AnalyzerProtocol,
        StreamExporterProtocol,
    )
    from aiperf.config.resolution.plan import BenchmarkRun


_logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class LoadedAnalyzer:
    """An analyzer plugin instance plus its declared dependencies by kind.

    ``required_accumulators`` names accumulators whose live instance the analyzer
    queries (``SummaryContext.get_accumulator``); ``required_summaries`` names
    accumulators whose summary output it reads (``SummaryContext.get_output``).
    RecordsManager runs the analyzer only when both are satisfied.
    """

    analyzer: AnalyzerProtocol
    """The instantiated analyzer plugin."""

    required_accumulators: list[str] = field(default_factory=list)
    """AccumulatorType names whose LIVE instance the analyzer queries via
    ``SummaryContext.get_accumulator``."""

    required_summaries: list[str] = field(default_factory=list)
    """AccumulatorType names whose SUMMARY output the analyzer reads via
    ``SummaryContext.get_output``."""


class _LoaderHost(Protocol):
    """Minimal surface the plugin loaders use on the owning service."""

    service_id: str
    run: BenchmarkRun
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
    are silently skipped — that's the explicit opt-out path. A construction
    failure of the load-bearing ``metric_results`` accumulator re-raises so
    ``RecordsManager.__init__`` fails loudly rather than silently producing
    empty results; every other accumulator (GPU telemetry / server metrics)
    is optional, so its failure is logged via ``host.error`` and skipped.
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
        except (PluginDisabled, PostProcessorDisabled):
            host.debug(f"Accumulator {entry.name} is disabled and will not be used")
        except Exception as e:  # noqa: BLE001 - optional accumulators must not abort the records manager
            host.error(f"Failed to create accumulator {entry.name}: {e}")
            # The metric_results accumulator is the sole summary producer; if it
            # cannot be built there is no fallback, so fail fast instead of
            # silently yielding empty results with exit 0.
            if AccumulatorType(entry.name) == AccumulatorType.METRIC_RESULTS:
                raise
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
        except (PluginDisabled, PostProcessorDisabled):
            host.debug(f"Stream exporter {entry.name} is disabled and will not be used")
        except Exception as e:  # noqa: BLE001 - one bad exporter must not abort the records manager
            host.error(f"Failed to create stream exporter {entry.name}: {e}")
    return exporters


def load_analyzers(host: _LoaderHost) -> list[LoadedAnalyzer]:
    """Instantiate all enabled ``ANALYZER`` plugins for ``host``.

    Analyzers are stateless summarize-time components (no lifecycle, no record
    ingestion) that read peer accumulators via the SummaryContext. Each entry is
    returned as a :class:`LoadedAnalyzer` carrying its declared dependencies by
    kind — ``required_accumulators`` (live instance) and ``required_summaries``
    (summary output) — so the caller can skip an analyzer whose dependencies are
    unavailable. Same disable/error policy as :func:`load_accumulators`.
    """
    analyzers: list[LoadedAnalyzer] = []
    for entry in plugins.iter_entries(PluginType.ANALYZER):
        try:
            AnalyzerClass = plugins.get_class(PluginType.ANALYZER, entry.name)
            analyzer = AnalyzerClass(
                service_id=host.service_id,
                run=host.run,
                pub_client=host.pub_client,
            )
            loaded = LoadedAnalyzer(
                analyzer=analyzer,
                required_accumulators=list(
                    entry.metadata.get("required_accumulators", [])
                ),
                required_summaries=list(entry.metadata.get("required_summaries", [])),
            )
            # Catch metadata typos loudly: a required name that is not a known
            # AccumulatorType would otherwise silently disable the analyzer at
            # every run (its dependency never "resolves").
            known = {str(t) for t in AccumulatorType}
            unknown = [
                r
                for r in (*loaded.required_accumulators, *loaded.required_summaries)
                if r not in known
            ]
            if unknown:
                host.error(
                    f"Analyzer {entry.name} declares unknown accumulator dependencies "
                    f"{unknown} (valid: {sorted(known)}); it will never run. Fix the "
                    "required_accumulators/required_summaries in plugins.yaml."
                )
            analyzers.append(loaded)
            host.debug(
                f"Created analyzer: {entry.name}: {analyzer.__class__.__name__} "
                f"(accumulators={loaded.required_accumulators}, "
                f"summaries={loaded.required_summaries})"
            )
        except (PluginDisabled, PostProcessorDisabled):
            host.debug(f"Analyzer {entry.name} is disabled and will not be used")
        except Exception as e:  # noqa: BLE001 - one bad analyzer must not abort the records manager
            host.error(f"Failed to create analyzer {entry.name}: {e}")
    return analyzers


async def generate_realtime_metrics(
    accumulators: list[AccumulatorProtocol],
    timeout: float = 30.0,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> list[MetricResult]:
    """Generate the real-time metrics for the profile run.

    Runs every accumulator's ``summarize`` in parallel with a short timeout
    and flattens the results to a single list of ``MetricResult``. Tolerates
    accumulators that return either ``AccumulatorMetricsSummary`` (with a
    ``.results`` dict-of-MetricResult) or a plain ``list[MetricResult]`` —
    GPU telemetry / server metrics accumulators return list shape.

    The realtime view is scoped to ``phase`` (PROFILING by default) so warmup
    records never dilute the live counts/throughput; the final export path
    applies the same phase mask.
    """
    ctx = SummaryContext(phase=phase)
    results = await asyncio.gather(
        *[
            asyncio.wait_for(acc.summarize(ctx), timeout=timeout)
            for acc in accumulators
        ],
        return_exceptions=True,
    )
    flat: list[MetricResult] = []
    for acc, result in zip(accumulators, results, strict=True):
        if isinstance(result, BaseException):
            # A persistently failing accumulator would otherwise leave the
            # realtime dashboard/log block silently stale with no trail.
            _logger.warning(
                f"Realtime summarize failed for {acc.__class__.__name__}: {result!r}"
            )
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
