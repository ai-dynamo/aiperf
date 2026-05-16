# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for RecordsManager partial-checkpoint export and atomic file writes."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.common.models.export_models import JsonExportData
from aiperf.records.records_manager_processing import generate_realtime_metrics

if TYPE_CHECKING:
    from aiperf.config.benchmark import BenchmarkConfig
    from aiperf.post_processors.protocols import ResultsProcessorProtocol
    from aiperf.records.error_tracker import ErrorTracker
    from aiperf.records.records_tracker import RecordsTracker


def write_json_file_atomic(path: Path, content: bytes) -> None:
    """Write a JSON file atomically so readers never observe a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def current_results_record_count(tracker: RecordsTracker) -> int:
    """Return the total processed record count across all result phases."""
    return sum(
        tracker.create_stats_for_phase(phase).total_records
        for phase in tracker.get_results_phases()
    )


def build_partial_profile_results(
    records: list[MetricResult],
    tracker: RecordsTracker,
    error_tracker: ErrorTracker,
) -> ProfileResults:
    """Build a partial profile result snapshot from current in-memory state."""
    start_ns, end_ns = tracker.get_results_time_window()
    error_summary = []
    for phase in tracker.get_results_phases():
        error_summary.extend(error_tracker.get_error_summary_for_phase(phase))

    return ProfileResults(
        records=records,
        completed=current_results_record_count(tracker),
        start_ns=start_ns or time.time_ns(),
        end_ns=end_ns or time.time_ns(),
        error_summary=error_summary,
        was_cancelled=any(
            tracker.was_phase_cancelled(phase) for phase in tracker.get_results_phases()
        ),
    )


def generate_json_export_data(
    records: list[MetricResult],
    profile_results: ProfileResults,
    benchmark_config: BenchmarkConfig,
) -> JsonExportData:
    """Generate JsonExportData for ConfigMap publishing.

    Args:
        records: List of metric results from processing
        profile_results: The profile results containing timing and error info
        benchmark_config: The benchmark config used to run the benchmark

    Returns:
        JsonExportData ready for serialization to ConfigMap
    """
    try:
        aiperf_version = get_version("aiperf")
    except PackageNotFoundError:
        aiperf_version = "unknown"

    start_time = (
        datetime.fromtimestamp(profile_results.start_ns / NANOS_PER_SECOND)
        if profile_results.start_ns
        else None
    )
    end_time = (
        datetime.fromtimestamp(profile_results.end_ns / NANOS_PER_SECOND)
        if profile_results.end_ns
        else None
    )

    export_data = JsonExportData(
        schema_version=JsonExportData.SCHEMA_VERSION,
        aiperf_version=aiperf_version,
        benchmark_id=benchmark_config.benchmark_id,
        input_config=benchmark_config,
        was_cancelled=profile_results.was_cancelled,
        error_summary=profile_results.error_summary,
        start_time=start_time,
        end_time=end_time,
        telemetry_data=None,
    )

    for metric in records:
        if metric.tag:
            setattr(export_data, str(metric.tag), metric.to_json_result())

    return export_data


async def write_partial_checkpoint(
    *,
    tracker: RecordsTracker,
    error_tracker: ErrorTracker,
    processors: list[ResultsProcessorProtocol],
    benchmark_config: BenchmarkConfig,
    checkpoint_path: Path,
    last_checkpoint_records: int,
) -> int:
    """Persist a partial aggregate snapshot for recovery; returns new checkpoint count.

    Returns ``last_checkpoint_records`` unchanged when there is nothing new to
    write, otherwise returns the new total record count that was persisted.
    """
    total_records = current_results_record_count(tracker)
    if total_records == 0 or total_records == last_checkpoint_records:
        return last_checkpoint_records

    records = await generate_realtime_metrics(processors)
    if not records:
        return last_checkpoint_records

    profile_results = build_partial_profile_results(records, tracker, error_tracker)
    export_data = generate_json_export_data(records, profile_results, benchmark_config)
    export_data.checkpoint = True
    export_data.records_completed = total_records
    export_data.generated_at_ns = time.time_ns()

    payload = export_data.model_dump_json(
        indent=2, exclude_unset=True, exclude_none=True
    ).encode("utf-8")
    await asyncio.to_thread(write_json_file_atomic, checkpoint_path, payload)
    return total_records
