# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for RecordsManager: realtime metrics filtering and results bucketing."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Protocol

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.models import (
    ErrorDetails,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
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
) -> tuple[list[MetricResult], dict, list[ErrorDetails], list[BaseException]]:
    """Sort ``asyncio.gather(return_exceptions=True)`` output by kind.

    Returns (records_results, timeslice_metric_results, error_results,
    raw_exceptions). Raw exceptions are returned separately so the caller can
    log them before wrapping into ``ErrorDetails``.
    """
    records_results: list[MetricResult] = []
    timeslice_metric_results: dict = {}
    error_results: list[ErrorDetails] = []
    raw_exceptions: list[BaseException] = []
    for result in results:
        if isinstance(result, list):
            records_results.extend(result)
        elif isinstance(result, dict):
            timeslice_metric_results = result
        elif isinstance(result, ErrorDetails):
            error_results.append(result)
        elif isinstance(result, BaseException):
            raw_exceptions.append(result)
    return records_results, timeslice_metric_results, error_results, raw_exceptions


def build_process_records_result(
    *,
    records_results: list[MetricResult],
    timeslice_metric_results: dict,
    error_results: list[ErrorDetails],
    tracker: RecordsTracker,
    error_tracker: ErrorTracker,
    cancelled: bool,
) -> ProcessRecordsResult:
    """Assemble the final ``ProcessRecordsResult`` from bucketed summarize output."""
    start_ns, end_ns = tracker.get_results_time_window()
    error_summary: list = []
    for phase in tracker.get_results_phases():
        error_summary.extend(error_tracker.get_error_summary_for_phase(phase))
    return ProcessRecordsResult(
        results=ProfileResults(
            records=records_results,
            timeslice_metric_results=timeslice_metric_results,
            completed=len(records_results),
            start_ns=start_ns or time.time_ns(),
            end_ns=end_ns or time.time_ns(),
            error_summary=error_summary,
            was_cancelled=cancelled,
        ),
        errors=error_results,
    )
