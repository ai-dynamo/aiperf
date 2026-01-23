# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the visualization package.

This module defines file patterns, default paths, plot settings, and other
configuration constants used throughout the visualization functionality.
"""

from pathlib import Path

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum

# File patterns for AIPerf profiling output files. These reference the canonical definitions from OutputDefaults
PROFILE_EXPORT_JSONL = OutputDefaults.PROFILE_EXPORT_JSONL_FILE.name
PROFILE_EXPORT_AIPERF_JSON = OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE.name
PROFILE_EXPORT_TIMESLICES_CSV = (
    OutputDefaults.PROFILE_EXPORT_AIPERF_TIMESLICES_CSV_FILE.name
)
PROFILE_EXPORT_GPU_TELEMETRY_JSONL = (
    OutputDefaults.PROFILE_EXPORT_GPU_TELEMETRY_JSONL_FILE.name
)
SERVER_METRICS_EXPORT_JSON = OutputDefaults.SERVER_METRICS_EXPORT_JSON_FILE.name
SERVER_METRICS_EXPORT_CSV = OutputDefaults.SERVER_METRICS_EXPORT_CSV_FILE.name
SERVER_METRICS_EXPORT_JSONL = OutputDefaults.SERVER_METRICS_EXPORT_JSONL_FILE.name
SERVER_METRICS_EXPORT_PARQUET = "server_metrics_export.parquet"

# Default output directory and filenames
DEFAULT_OUTPUT_DIR = Path("plots")
DEFAULT_PNG_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "png"
PLOT_LOG_FILE = "aiperf_plot.log"

# Dashboard defaults
DEFAULT_DASHBOARD_PORT = 8050


class PlotMode(CaseInsensitiveStrEnum):
    """Available output modes for plot generation."""

    PNG = "png"
    DASHBOARD = "dashboard"


class PlotTheme(CaseInsensitiveStrEnum):
    """Available themes for plot styling."""

    LIGHT = "light"
    DARK = "dark"


DEFAULT_PLOT_WIDTH = 1600
DEFAULT_PLOT_HEIGHT = 800
DEFAULT_PLOT_DPI = 150

PLOT_FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif"

NVIDIA_GREEN = "#76B900"
NVIDIA_DARK = "#0a0a0a"
NVIDIA_GOLD = "#F4E5C3"
NVIDIA_WHITE = "#FFFFFF"
NVIDIA_DARK_BG = "#1a1a1a"
NVIDIA_GRAY = "#999999"
NVIDIA_BORDER_DARK = "#333333"
NVIDIA_BORDER_LIGHT = "#CCCCCC"
NVIDIA_TEXT_LIGHT = "#E0E0E0"
NVIDIA_CARD_BG = "#252525"
OUTLIER_RED = "#E74C3C"

# Direction indicators for derived metrics (not in MetricRegistry)
# Maps metric name to direction: True = ↑ (higher is better), False = ↓ (lower is better)
DERIVED_METRIC_DIRECTIONS = {
    "output_token_throughput_per_gpu": True,  # nosec
    "output_token_throughput_per_user": True,  # nosec
}


DARK_THEME_COLORS = {
    "primary": NVIDIA_GREEN,
    "secondary": NVIDIA_GOLD,
    "background": NVIDIA_DARK_BG,
    "paper": NVIDIA_CARD_BG,
    "text": NVIDIA_TEXT_LIGHT,
    "grid": NVIDIA_BORDER_DARK,
    "border": NVIDIA_BORDER_DARK,
}

LIGHT_THEME_COLORS = {
    "primary": NVIDIA_GREEN,
    "secondary": NVIDIA_GRAY,
    "background": NVIDIA_WHITE,
    "paper": NVIDIA_WHITE,
    "text": NVIDIA_DARK,
    "grid": NVIDIA_BORDER_LIGHT,
    "border": NVIDIA_BORDER_LIGHT,
}


DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
DEFAULT_PERCENTILE = "p50"
AVAILABLE_STATS = ["avg", "min", "max", "std"]

# All available statistic keys - ordered for UI display (most common first)
ALL_STAT_KEYS = [
    "p50",
    "avg",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
    "std",
    "p1",
    "p5",
    "p10",
    "p25",
    "p75",
]

# Human-readable labels for statistics (used in dropdowns)
STAT_LABELS = {
    "avg": "Average",
    "min": "Minimum",
    "max": "Maximum",
    "std": "Std Dev",
    "p1": "p1",
    "p5": "p5",
    "p10": "p10",
    "p25": "p25",
    "p50": "p50 (Median)",
    "p75": "p75",
    "p90": "p90",
    "p95": "p95",
    "p99": "p99",
}

# Patterns indicating cumulative metrics where run-level reference lines don't make sense
# (the aggregated value is a sum/total, not comparable to per-timeslice values)
CUMULATIVE_METRIC_PATTERNS = ["total"]

# Non-metric keys in the aggregated JSON (used for filtering)
NON_METRIC_KEYS = {
    "schema_version",
    "aiperf_version",
    "benchmark_id",
    "input_config",
    "telemetry_data",
    "start_time",
    "end_time",
    "was_cancelled",
    "error_summary",
}

# Metric category rules for grouping in UI dropdowns.
# Dict order defines display order. Keywords are matched against lowercase metric names.
METRIC_CATEGORY_RULES: dict[str, list[str]] = {
    "Latency Metrics": ["latency", "ttft", "itl", "ttst", "ttfo"],
    "Throughput Metrics": ["throughput", "goodput"],
    "Counts & Lengths": ["count", "sequence_length", "isl", "osl"],
    "Configuration": ["concurrency", "duration"],
    "GPU Telemetry": ["gpu", "memory", "power", "temperature"],
    "Server Metrics": ["vllm:", "triton:", "http_", "dynamo_", "nvidia_"],
    "Other Metrics": [],  # Fallback category (no keywords)
}
