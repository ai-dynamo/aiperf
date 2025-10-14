# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for validating aggregate metrics against raw JSONL data.

This module provides utilities to:
1. Extract metric values from raw JSONL records
2. Compute statistical aggregates (avg, min, max, std, percentiles)
3. Validate computed statistics match the JSON export data

Example usage:
    # Validate all metrics at once
    computed_metrics = validate_all_metrics(result.jsonl, result.json)

    # Or validate specific metrics
    latency_values = extract_metric_values(result.jsonl, "request_latency")
    computed = compute_stats(latency_values)
    validate_metric_stats(computed, result.json.request_latency, "request_latency")
"""

from typing import Any

import numpy as np
from pydantic import BaseModel

from aiperf.common.constants import STAT_KEYS
from aiperf.common.models import JsonExportData, MetricRecordInfo


class ComputedStats(BaseModel):
    """Computed statistics from raw metric values."""

    avg: float
    min: float
    max: float
    std: float
    p1: float
    p5: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    count: int


def compute_stats(values: list[float]) -> ComputedStats:
    """Compute statistical aggregates from a list of values.

    Args:
        values: List of numeric values to compute statistics for

    Returns:
        ComputedStats object with all statistical measures

    Raises:
        ValueError: If values list is empty
    """
    if not values:
        msg = "Cannot compute statistics on empty list"
        raise ValueError(msg)

    arr = np.array(values)

    return ComputedStats(
        avg=float(np.mean(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        p1=float(np.percentile(arr, 1)),
        p5=float(np.percentile(arr, 5)),
        p10=float(np.percentile(arr, 10)),
        p25=float(np.percentile(arr, 25)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        count=len(values),
    )


def extract_metric_values(
    records: list[MetricRecordInfo], metric_name: str
) -> list[float]:
    """Extract all values for a specific metric from JSONL records.

    Args:
        records: List of MetricRecordInfo objects from JSONL file
        metric_name: Name of the metric to extract (e.g., 'request_latency')

    Returns:
        List of metric values as floats
    """
    values = []
    for record in records:
        if metric_name in record.metrics:
            values.append(record.metrics[metric_name].value)
    return values


def validate_metric_stats(
    computed: ComputedStats,
    json_metric: Any,
    metric_name: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Validate that computed statistics match the JSON export data.

    Args:
        computed: Computed statistics from raw JSONL data
        json_metric: JsonMetricResult from the JSON export
        metric_name: Name of the metric being validated (for error messages)
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Raises:
        AssertionError: If any statistic doesn't match within tolerance
    """
    if json_metric is None:
        msg = f"Metric '{metric_name}' not found in JSON export"
        raise AssertionError(msg)

    for stat in STAT_KEYS:
        computed_val = getattr(computed, stat)
        json_val = getattr(json_metric, stat, None)

        if json_val is None:
            continue

        if not np.isclose(computed_val, json_val, rtol=rtol, atol=atol):
            msg = (
                f"Mismatch for {metric_name}.{stat}: "
                f"computed={computed_val}, json={json_val}, "
                f"diff={abs(computed_val - json_val)}"
            )
            raise AssertionError(msg)


def validate_all_metrics(
    jsonl_records: list[MetricRecordInfo],
    json_export: JsonExportData,
    rtol: float = 1,
    atol: float = 1,
) -> dict[str, ComputedStats]:
    """Validate all metrics in JSON export against computed values from JSONL.

    Args:
        jsonl_records: List of MetricRecordInfo from JSONL file
        json_export: JsonExportData from JSON export file
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Returns:
        Dictionary mapping metric names to their computed statistics

    Raises:
        ValueError: If jsonl_records is empty
        AssertionError: If any metric validation fails
    """
    # Get all metric names from the first record to know what to validate
    if not jsonl_records:
        msg = "No JSONL records to validate"
        raise ValueError(msg)

    metric_names = list(jsonl_records[0].metrics.keys())
    computed_metrics = {}

    for metric_name in metric_names:
        # Extract values from JSONL
        values = extract_metric_values(jsonl_records, metric_name)

        if not values:
            continue

        # Compute statistics
        computed = compute_stats(values)
        computed_metrics[metric_name] = computed

        # Get corresponding JSON metric
        json_metric = getattr(json_export, metric_name, None)

        # Validate if it exists in JSON export
        if json_metric is not None:
            validate_metric_stats(computed, json_metric, metric_name, rtol, atol)

    return computed_metrics
