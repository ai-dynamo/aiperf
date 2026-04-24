# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.common.enums import ListMetricAggregationMode
from aiperf.config._base import BaseConfig

__all__ = ["MetricsConfig"]


class MetricsConfig(BaseConfig):
    """Configuration for benchmark metric aggregation behavior."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    list_metric_aggregation: Annotated[
        ListMetricAggregationMode,
        Field(
            default=ListMetricAggregationMode.EXACT,
            description="Aggregation mode for list-valued metrics in benchmark summaries.",
        ),
    ]
