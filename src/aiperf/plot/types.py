# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared types for the plot subpackage."""

from __future__ import annotations

from typing import Any, NamedTuple


class ParsedMetricSpec(NamedTuple):
    """Parsed server metric specification with optional filters."""

    metric_name: str
    endpoint_url: str | None
    labels: dict[str, str] | None


class FilteredMetrics(NamedTuple):
    """Filtered server metrics DataFrame with metadata."""

    dataframe: Any  # pandas DataFrame
    unit: str
    metric_type: str
