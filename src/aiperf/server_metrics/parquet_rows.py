# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row collection helpers for the Parquet server-metrics exporter.

Split from `parquet_exporter.py` to keep both files under the
`tools/check_ergonomics.py` file/function size limits. Each collector
yields rows in the normalized schema consumed by the exporter.
"""

from __future__ import annotations

import numpy as np

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricEntry,
)
from aiperf.server_metrics.units import infer_unit


def collect_scalar_rows(
    *,
    endpoint: str,
    metric_name: str,
    metric_entry: ServerMetricEntry,
    labels_dict: dict[str, str] | None,
    label_keys: set[str],
    time_filter: TimeRangeFilter | None,
) -> list[dict]:
    """Collect rows for gauge or counter metrics with delta calculations.

    For gauges: exports raw values at each timestamp.
    For counters: exports cumulative deltas from reference point at each timestamp.
    """
    time_series = metric_entry.data
    if not isinstance(time_series, ScalarTimeSeries):
        return []

    metric_type = metric_entry.metric_type
    is_gauge = metric_type == PrometheusMetricType.GAUGE
    unit = infer_unit(metric_name, metric_entry.description)
    unit_display = unit.display_name() if unit else None

    filtered_timestamps, filtered_values, reference_idx = _slice_scalar_series(
        time_series, time_filter
    )
    if len(filtered_timestamps) == 0:
        return []

    values_to_export = _scalar_values_to_export(
        time_series=time_series,
        filtered_values=filtered_values,
        reference_idx=reference_idx,
        is_gauge=is_gauge,
    )

    assert len(filtered_timestamps) == len(values_to_export), (
        f"Array length mismatch: {len(filtered_timestamps)} timestamps "
        f"!= {len(values_to_export)} values"
    )

    return [
        _build_scalar_row(
            endpoint=endpoint,
            metric_name=metric_name,
            metric_type=metric_type,
            unit_display=unit_display,
            description=metric_entry.description,
            timestamp=timestamp,
            value=value,
            labels_dict=labels_dict,
            label_keys=label_keys,
        )
        for timestamp, value in zip(filtered_timestamps, values_to_export, strict=False)
    ]


def collect_histogram_rows(
    *,
    endpoint: str,
    metric_name: str,
    metric_entry: ServerMetricEntry,
    labels_dict: dict[str, str] | None,
    label_keys: set[str],
    time_filter: TimeRangeFilter | None,
) -> list[dict]:
    """Collect rows for histogram metrics with delta calculations.

    Creates one row per bucket per timestamp (normalized schema).
    Each row includes cumulative deltas for sum/count, plus individual bucket delta.
    """
    time_series = metric_entry.data
    if not isinstance(time_series, HistogramTimeSeries) or len(time_series) == 0:
        return []

    unit = infer_unit(metric_name, metric_entry.description)
    unit_display = unit.display_name() if unit else None

    slice_data = _slice_histogram_series(time_series, time_filter)
    if slice_data is None:
        return []
    (
        filtered_timestamps,
        filtered_sums,
        filtered_counts,
        filtered_bucket_counts,
        reference_idx,
    ) = slice_data

    sum_deltas, count_deltas, bucket_deltas = _histogram_deltas(
        time_series=time_series,
        filtered_sums=filtered_sums,
        filtered_counts=filtered_counts,
        filtered_bucket_counts=filtered_bucket_counts,
        reference_idx=reference_idx,
    )

    rows: list[dict] = []
    bucket_les = time_series.bucket_les
    for i, timestamp in enumerate(filtered_timestamps):
        sum_delta = float(sum_deltas[i])
        count_delta = float(count_deltas[i])
        for j, bucket_le in enumerate(bucket_les):
            rows.append(
                _build_histogram_row(
                    endpoint=endpoint,
                    metric_name=metric_name,
                    unit_display=unit_display,
                    description=metric_entry.description,
                    timestamp=timestamp,
                    labels_dict=labels_dict,
                    label_keys=label_keys,
                    sum_delta=sum_delta,
                    count_delta=count_delta,
                    bucket_le=bucket_le,
                    bucket_delta=float(bucket_deltas[i, j]),
                )
            )
    return rows


def _slice_scalar_series(
    time_series: ScalarTimeSeries,
    time_filter: TimeRangeFilter | None,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    time_mask = (
        time_series.get_time_mask(time_filter)
        if time_filter
        else np.ones(len(time_series), dtype=bool)
    )
    filtered_timestamps = time_series.timestamps[time_mask]
    filtered_values = time_series.values[time_mask]
    reference_idx = time_series.get_reference_idx(time_filter) if time_filter else None
    return filtered_timestamps, filtered_values, reference_idx


def _scalar_values_to_export(
    *,
    time_series: ScalarTimeSeries,
    filtered_values: np.ndarray,
    reference_idx: int | None,
    is_gauge: bool,
) -> np.ndarray:
    if is_gauge:
        return filtered_values
    reference_value = (
        time_series.values[reference_idx]
        if reference_idx is not None
        else filtered_values[0]
    )
    # Counters: cumulative deltas from reference; counter resets clamp to 0.
    deltas = filtered_values - reference_value
    return np.maximum(deltas, 0.0)


def _build_scalar_row(
    *,
    endpoint: str,
    metric_name: str,
    metric_type: PrometheusMetricType,
    unit_display: str | None,
    description: str | None,
    timestamp,
    value,
    labels_dict: dict[str, str] | None,
    label_keys: set[str],
) -> dict:
    return {
        "endpoint_url": endpoint,
        "metric_name": metric_name,
        "metric_type": metric_type,
        "unit": unit_display,
        "description": description,
        "timestamp_ns": int(timestamp),
        **{
            label_key: labels_dict.get(label_key) if labels_dict else None
            for label_key in label_keys
        },
        "value": float(value),
        "sum": None,
        "count": None,
        "bucket_le": None,
        "bucket_count": None,
    }


def _slice_histogram_series(
    time_series: HistogramTimeSeries,
    time_filter: TimeRangeFilter | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None] | None:
    if time_filter:
        reference_idx, final_idx = time_series.get_indices_for_filter(time_filter)
        first_idx = np.searchsorted(
            time_series.timestamps, time_filter.start_ns, side="left"
        )
    else:
        reference_idx = None
        first_idx = 0
        final_idx = len(time_series) - 1

    filtered_timestamps = time_series.timestamps[first_idx : final_idx + 1]
    if len(filtered_timestamps) == 0:
        return None
    filtered_sums = time_series.sums[first_idx : final_idx + 1]
    filtered_counts = time_series.counts[first_idx : final_idx + 1]
    filtered_bucket_counts = time_series.bucket_counts[first_idx : final_idx + 1]
    return (
        filtered_timestamps,
        filtered_sums,
        filtered_counts,
        filtered_bucket_counts,
        reference_idx,
    )


def _histogram_deltas(
    *,
    time_series: HistogramTimeSeries,
    filtered_sums: np.ndarray,
    filtered_counts: np.ndarray,
    filtered_bucket_counts: np.ndarray,
    reference_idx: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference_idx is not None:
        reference_sum = time_series.sums[reference_idx]
        reference_count = time_series.counts[reference_idx]
        reference_buckets = time_series.bucket_counts[reference_idx]
    else:
        reference_sum = filtered_sums[0]
        reference_count = filtered_counts[0]
        reference_buckets = filtered_bucket_counts[0]

    sum_deltas = np.maximum(filtered_sums - reference_sum, 0.0)
    count_deltas = np.maximum(filtered_counts - reference_count, 0.0)
    bucket_deltas = np.maximum(filtered_bucket_counts - reference_buckets, 0.0)
    return sum_deltas, count_deltas, bucket_deltas


def _build_histogram_row(
    *,
    endpoint: str,
    metric_name: str,
    unit_display: str | None,
    description: str | None,
    timestamp,
    labels_dict: dict[str, str] | None,
    label_keys: set[str],
    sum_delta: float,
    count_delta: float,
    bucket_le: str,
    bucket_delta: float,
) -> dict:
    return {
        "endpoint_url": endpoint,
        "metric_name": metric_name,
        "metric_type": PrometheusMetricType.HISTOGRAM,
        "unit": unit_display,
        "description": description,
        "timestamp_ns": int(timestamp),
        **{
            label_key: labels_dict.get(label_key) if labels_dict else None
            for label_key in label_keys
        },
        "value": None,
        "sum": sum_delta,
        "count": count_delta,
        "bucket_le": bucket_le,
        "bucket_count": bucket_delta,
    }
