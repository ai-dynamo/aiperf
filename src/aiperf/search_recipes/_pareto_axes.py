# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pareto axes specification declared by recipes that opt into console Pareto plots."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig


class ParetoAxesSpec(BaseConfig):
    """Declares the 2D axes a recipe wants visualized as a Pareto plot.

    Recipes set this as a class-level attribute to opt into the live and
    end-of-sweep Pareto plot path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_metric: str = Field(
        description="Flat metric key for the x axis (e.g. 'request_latency')."
    )
    x_stat: str = Field(description="Stat name on the x metric (e.g. 'p95', 'avg').")
    x_minimize: bool = Field(
        default=True,
        description="True iff lower x is dominant (e.g. latency is lower-is-better).",
    )
    y_metric: str = Field(
        description="Flat metric key for the y axis (e.g. 'output_token_throughput')."
    )
    y_stat: str = Field(description="Stat name on the y metric (e.g. 'avg', 'p95').")
    y_maximize: bool = Field(
        default=True,
        description="True iff higher y is dominant (e.g. throughput is higher-is-better).",
    )
    series_keys: tuple[str, ...] = Field(
        default=(),
        description=(
            "Parameter keys whose values define a series in the plot. "
            "Empty tuple = a single growing curve (e.g. concurrency-ramp)."
        ),
    )
