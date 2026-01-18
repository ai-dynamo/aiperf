# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Swept parameter detection for multi-run plots.

Automatically detects which configuration parameters vary across runs
and provides heuristics for selecting group_by and label_by parameters.
"""

import pandas as pd

# Default parameters to ignore when detecting swept parameters
DEFAULT_IGNORE_PARAMS = {
    "profiling.start_time",
    "profiling.end_time",
    "loadgen.random_seed",
    "endpoint.url",
    "run_id",
    "run_path",
}


def detect_swept_parameters(
    df: pd.DataFrame, ignore_params: set[str] | None = None
) -> list[str]:
    """
    Detect which parameters vary across runs in the DataFrame.

    Args:
        df: DataFrame with configuration columns
        ignore_params: Set of parameter names to ignore (uses defaults if None)

    Returns:
        List of column names that have more than one unique value
    """
    if ignore_params is None:
        ignore_params = DEFAULT_IGNORE_PARAMS

    swept_params = []

    for col in df.columns:
        # Skip ignored parameters
        if col in ignore_params:
            continue

        # Check if this column has more than one unique value
        unique_values = df[col].dropna().unique()
        if len(unique_values) > 1:
            swept_params.append(col)

    return swept_params


def auto_select_group_by(
    df: pd.DataFrame, swept_params: list[str] | None = None
) -> str | None:
    """
    Automatically select the best parameter for grouping (coloring) in plots.

    Heuristics:
    1. Prefer "model" if it varies
    2. Prefer parameters with fewer unique values (better for distinct colors)
    3. Avoid parameters with too many unique values (>10)

    Args:
        df: DataFrame with configuration columns
        swept_params: List of swept parameter names (auto-detected if None)

    Returns:
        Column name to use for group_by, or None if no swept parameters exist
    """
    if swept_params is None:
        swept_params = detect_swept_parameters(df)

    if not swept_params:
        return None

    # Prefer "model" if it's in swept params
    if "model" in swept_params:
        return "model"

    # Find parameter with fewest unique values (but > 1)
    best_param = None
    best_count = float("inf")

    for param in swept_params:
        unique_count = df[param].nunique()
        # Skip parameters with too many unique values
        if unique_count > 10:
            continue
        if unique_count < best_count:
            best_count = unique_count
            best_param = param

    return best_param if best_param else swept_params[0]


def auto_select_label_by(
    df: pd.DataFrame, swept_params: list[str] | None = None, group_by: str | None = None
) -> str | None:
    """
    Automatically select the best parameter for labeling points in plots.

    Heuristics:
    1. Prefer "concurrency" if it varies
    2. Avoid using the same parameter as group_by
    3. Prefer parameters with moderate number of unique values (2-20)

    Args:
        df: DataFrame with configuration columns
        swept_params: List of swept parameter names (auto-detected if None)
        group_by: The parameter being used for grouping (to avoid duplicates)

    Returns:
        Column name to use for label_by, or None if no swept parameters exist
    """
    if swept_params is None:
        swept_params = detect_swept_parameters(df)

    if not swept_params:
        return None

    # Prefer "concurrency" if it's in swept params
    if "concurrency" in swept_params:
        return "concurrency"

    # Find a good parameter for labeling (not the same as group_by)
    for param in swept_params:
        if param == group_by:
            continue

        unique_count = df[param].nunique()
        # Good range for point labels
        if 2 <= unique_count <= 20:
            return param

    # Fallback: use first swept param that's not group_by
    for param in swept_params:
        if param != group_by:
            return param

    return swept_params[0] if swept_params else None
