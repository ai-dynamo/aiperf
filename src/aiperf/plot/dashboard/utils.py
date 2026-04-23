# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions shared between dashboard builder and callbacks.

This module re-exports helpers from internal submodules so existing callers
(`from aiperf.plot.dashboard.utils import ...`) keep working after the split
into `_container`, `_dataframe`, and `_metrics`.
"""

from aiperf.plot.dashboard._container import create_plot_container_component
from aiperf.plot.dashboard._dataframe import (
    _convert_to_numeric,
    _match_trace_point_to_run_idx,
    add_run_idx_to_figure,
    extract_metric_value,
    prepare_timeseries_dataframe,
    runs_to_dataframe,
)
from aiperf.plot.dashboard._metrics import (
    SINGLE_RUN_STAT_SUFFIXES,
    get_available_stats_for_metric,
    get_plot_title,
    get_single_run_metrics_with_stats,
    get_stat_options_for_single_run_metric,
    resolve_single_run_column_name,
)

__all__ = [
    "SINGLE_RUN_STAT_SUFFIXES",
    "_convert_to_numeric",
    "_match_trace_point_to_run_idx",
    "add_run_idx_to_figure",
    "create_plot_container_component",
    "extract_metric_value",
    "get_available_stats_for_metric",
    "get_plot_title",
    "get_single_run_metrics_with_stats",
    "get_stat_options_for_single_run_metric",
    "prepare_timeseries_dataframe",
    "resolve_single_run_column_name",
    "runs_to_dataframe",
]
