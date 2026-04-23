# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-metric helpers for the dashboard: titles, stat discovery, column resolution."""

from aiperf.plot.constants import ALL_STAT_KEYS
from aiperf.plot.metric_names import get_metric_display_name

# Stats that can be suffixed to metrics (e.g., inter_chunk_latency_avg)
SINGLE_RUN_STAT_SUFFIXES = ["avg", "p50", "p95", "std", "min", "max", "range"]


def get_plot_title(plot_id: str, plot_configs: dict | None = None) -> str:
    """
    Get display title for a plot ID.

    Args:
        plot_id: Plot ID (e.g., 'pareto', 'custom-ttft-vs-latency')
        plot_configs: Dict of all plot configs (default and custom)

    Returns:
        Human-readable plot title
    """
    # Check plot_configs FIRST (works for both default and custom)
    if plot_configs and plot_id in plot_configs:
        config = plot_configs[plot_id]
        return config.get("title", plot_id.replace("-", " ").title())

    # Fallback: format the plot_id
    return plot_id.replace("-", " ").title()


def get_available_stats_for_metric(runs: list, metric_name: str) -> list[str]:
    """
    Get list of available stats for a given metric across all runs.

    Args:
        runs: List of RunData objects
        metric_name: Name of the metric to check

    Returns:
        List of available stat keys (e.g., ["avg", "p50", "p90"])
    """
    if not runs:
        return ALL_STAT_KEYS

    # Special case: concurrency is metadata, not a metric
    if metric_name == "concurrency":
        return ["value"]

    # Sample first run to get available stats
    first_run = runs[0]

    # Handle derived metrics
    if metric_name == "output_token_throughput_per_user":
        # Based on output_token_throughput
        metric = first_run.get_metric("output_token_throughput")
    elif metric_name == "output_token_throughput_per_gpu":
        # Try direct metric first, fallback to base throughput
        metric = first_run.get_metric("output_token_throughput_per_gpu")
        if metric is None:
            metric = first_run.get_metric("output_token_throughput")
    else:
        # Standard metric
        metric = first_run.get_metric(metric_name)

    if metric is None:
        return ALL_STAT_KEYS

    # Extract available stats
    if isinstance(metric, dict):
        return [k for k in metric if k != "unit"]
    else:
        # MetricResult object - check which stat attributes exist and are not None
        return [
            stat for stat in ALL_STAT_KEYS if getattr(metric, stat, None) is not None
        ]


def get_single_run_metrics_with_stats(
    columns: list[str], excluded_columns: list[str]
) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Process DataFrame columns to extract base metrics and their available stats.

    Groups compound metrics (e.g., inter_chunk_latency_avg, inter_chunk_latency_p50)
    into a single base metric with available stat options.

    Args:
        columns: List of column names from DataFrame
        excluded_columns: List of column names to exclude

    Returns:
        Tuple of:
        - List of metric options for dropdown (base metrics only)
        - Dict mapping base metric name to list of available stats
    """
    metric_stats: dict[str, list[str]] = {}
    simple_metrics: list[str] = []

    for col in columns:
        if col in excluded_columns:
            continue

        # Check if this column has a stat suffix
        found_stat = None
        base_metric = None
        for stat in SINGLE_RUN_STAT_SUFFIXES:
            suffix = f"_{stat}"
            if col.endswith(suffix):
                base_metric = col[: -len(suffix)]
                found_stat = stat
                break

        if base_metric and found_stat:
            # This is a compound metric with stat suffix
            if base_metric not in metric_stats:
                metric_stats[base_metric] = []
            metric_stats[base_metric].append(found_stat)
        else:
            # Simple metric without stat suffix
            simple_metrics.append(col)

    # Build dropdown options
    options: list[dict] = []

    # Add simple metrics first
    for metric in simple_metrics:
        options.append({"label": get_metric_display_name(metric), "value": metric})

    # Add compound metrics (base names only)
    for base_metric in sorted(metric_stats.keys()):
        options.append(
            {"label": get_metric_display_name(base_metric), "value": base_metric}
        )

    # For simple metrics, set their stats to just "avg"
    all_metric_stats = {metric: ["avg"] for metric in simple_metrics}
    all_metric_stats.update(metric_stats)

    return options, all_metric_stats


def get_stat_options_for_single_run_metric(
    metric_name: str, metric_stats: dict[str, list[str]]
) -> list[dict]:
    """
    Get stat dropdown options for a specific metric.

    Args:
        metric_name: Base metric name
        metric_stats: Dict mapping metric names to available stats

    Returns:
        List of dropdown options for stats
    """
    stats = metric_stats.get(metric_name, ["avg"])

    # Define label mapping
    stat_labels = {
        "avg": "Average",
        "p50": "p50 (Median)",
        "p95": "p95",
        "std": "Std Dev",
        "min": "Minimum",
        "max": "Maximum",
        "range": "Range",
    }

    # Order stats consistently
    ordered_stats = ["avg", "p50", "p95", "std", "min", "max", "range"]
    options = []
    for stat in ordered_stats:
        if stat in stats:
            options.append({"label": stat_labels.get(stat, stat), "value": stat})

    return options if options else [{"label": "Average", "value": "avg"}]


def resolve_single_run_column_name(
    metric_name: str, stat: str | None, metric_stats: dict[str, list[str]]
) -> str:
    """
    Resolve the actual DataFrame column name from metric + stat.

    For compound metrics like inter_chunk_latency, combines metric_stat.
    For simple metrics, returns the metric name directly.

    Args:
        metric_name: Base metric name
        stat: Selected stat (may be None or "avg" for simple metrics)
        metric_stats: Dict mapping metric names to available stats

    Returns:
        Actual column name to use for DataFrame access
    """
    stats = metric_stats.get(metric_name, ["avg"])

    # If metric has only "avg" and it's a simple metric, return metric_name
    if stats == ["avg"]:
        return metric_name

    # For compound metrics, combine metric + stat
    if stat and stat in stats:
        return f"{metric_name}_{stat}"

    # Fallback: return first available stat
    return f"{metric_name}_{stats[0]}" if stats else metric_name
