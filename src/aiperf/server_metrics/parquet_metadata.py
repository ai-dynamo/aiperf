# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet file metadata assembly for server metrics exports.

Split from `parquet_exporter.py` to keep that module under the
`tools/check_ergonomics.py` file/function size limits. The public surface
is the single helper ``build_parquet_metadata``; the rest of the functions
here are private per-section builders.
"""

from __future__ import annotations

import socket
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import orjson
import pyarrow as pa

from aiperf import __version__ as aiperf_version
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import TimeRangeFilter

if TYPE_CHECKING:
    from aiperf.server_metrics.accumulator import ServerMetricsAccumulator


def build_parquet_metadata(
    *,
    accumulator: ServerMetricsAccumulator,
    time_filter: TimeRangeFilter,
    label_keys: set[str],
) -> dict[bytes, bytes]:
    """Build Parquet file metadata for provenance tracking.

    Args:
        accumulator: Server metrics accumulator (provides run cfg + hierarchy).
        time_filter: Time range filter for the profiling window.
        label_keys: Set of label columns in this file's schema.

    Returns:
        Dictionary of metadata key-value pairs (both as bytes).
    """
    run = accumulator.run
    metadata: dict[bytes, bytes] = {
        b"aiperf.schema_version": b"1.0",
        b"aiperf.version": aiperf_version.encode("utf-8"),
        b"aiperf.benchmark_id": run.cfg.benchmark_id.encode("utf-8"),
        b"aiperf.export_timestamp_utc": datetime.now(timezone.utc)
        .isoformat()
        .encode("utf-8"),
        b"aiperf.exporter": b"ServerMetricsParquetExporter",
    }
    _add_system_info(metadata)
    _add_time_filter_info(metadata, time_filter)
    _add_config_info(metadata, run)
    _add_endpoint_info(metadata, accumulator)
    _add_schema_info(metadata, label_keys)
    _add_metric_type_counts(metadata, accumulator)
    metadata[b"aiperf.schema_note"] = (
        b"Label columns vary by endpoint/model. Use union_by_name=true for cross-file queries."
    )
    return metadata


def _add_system_info(metadata: dict[bytes, bytes]) -> None:
    metadata[b"aiperf.hostname"] = socket.gethostname().encode("utf-8")
    metadata[b"aiperf.python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ).encode()
    try:
        metadata[b"aiperf.pyarrow_version"] = pa.__version__.encode("utf-8")
    except AttributeError:
        metadata[b"aiperf.pyarrow_version"] = b"unknown"


def _add_time_filter_info(
    metadata: dict[bytes, bytes], time_filter: TimeRangeFilter
) -> None:
    if time_filter.start_ns is not None:
        metadata[b"aiperf.time_filter_start_ns"] = str(time_filter.start_ns).encode(
            "utf-8"
        )
    if time_filter.end_ns is None:
        return
    metadata[b"aiperf.time_filter_end_ns"] = str(time_filter.end_ns).encode("utf-8")
    if time_filter.start_ns is None:
        return
    duration_ns = time_filter.end_ns - time_filter.start_ns
    metadata[b"aiperf.profiling_duration_ns"] = str(duration_ns).encode("utf-8")
    metadata[b"aiperf.profiling_duration_seconds"] = str(
        duration_ns / 1_000_000_000
    ).encode("utf-8")


def _add_config_info(metadata: dict[bytes, bytes], run) -> None:
    # Dump entire user config with exclude_unset to capture actual benchmark settings
    config_dict = run.cfg.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    metadata[b"aiperf.input_config"] = orjson.dumps(config_dict)
    metadata[b"aiperf.model_names"] = orjson.dumps(run.cfg.get_model_names())

    profiling_phases = run.cfg.get_profiling_phases()
    if not profiling_phases:
        return
    first_phase = next(iter(profiling_phases.values()))
    if first_phase.concurrency is not None:
        metadata[b"aiperf.concurrency"] = str(first_phase.concurrency).encode("utf-8")
    rate = getattr(first_phase, "rate", None)
    if rate is not None:
        metadata[b"aiperf.request_rate"] = str(rate).encode("utf-8")


def _add_endpoint_info(
    metadata: dict[bytes, bytes], accumulator: ServerMetricsAccumulator
) -> None:
    hierarchy = accumulator.get_hierarchy_for_export()
    endpoint_urls = sorted(hierarchy.endpoints.keys())
    metadata[b"aiperf.endpoint_urls"] = orjson.dumps(endpoint_urls)
    metadata[b"aiperf.endpoint_count"] = str(len(endpoint_urls)).encode("utf-8")


def _add_schema_info(metadata: dict[bytes, bytes], label_keys: set[str]) -> None:
    metadata[b"aiperf.label_columns"] = orjson.dumps(sorted(label_keys))
    metadata[b"aiperf.label_count"] = str(len(label_keys)).encode("utf-8")


def _add_metric_type_counts(
    metadata: dict[bytes, bytes], accumulator: ServerMetricsAccumulator
) -> None:
    hierarchy = accumulator.get_hierarchy_for_export()
    type_counts = {"gauge": 0, "counter": 0, "histogram": 0}
    total_metrics = 0
    for time_series_collection in hierarchy.endpoints.values():
        for metric_entry in time_series_collection.metrics.values():
            total_metrics += 1
            if metric_entry.metric_type == PrometheusMetricType.GAUGE:
                type_counts["gauge"] += 1
            elif metric_entry.metric_type == PrometheusMetricType.COUNTER:
                type_counts["counter"] += 1
            elif metric_entry.metric_type == PrometheusMetricType.HISTOGRAM:
                type_counts["histogram"] += 1
    metadata[b"aiperf.metric_count"] = str(total_metrics).encode("utf-8")
    metadata[b"aiperf.metric_type_counts"] = orjson.dumps(type_counts)
