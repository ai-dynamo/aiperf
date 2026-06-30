# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-rate time series configuration."""

from __future__ import annotations

import math
from typing import Any

import orjson
from pydantic import ConfigDict, Field, model_validator

from aiperf.common.path_safety import safe_read_template_path
from aiperf.config.base import BaseConfig


class RateSeriesPoint(BaseConfig):
    """One request-rate control point."""

    model_config = ConfigDict(extra="forbid")

    time_s: float = Field(
        ge=0.0,
        description="Elapsed phase time in seconds for this request-rate point.",
    )
    qps: float = Field(
        gt=0.0,
        description="Request rate in queries per second at this point.",
    )


class RateSeriesConfig(BaseConfig):
    """Piecewise-linear request-rate schedule."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(
        default=None,
        description="JSON file path containing request-rate control points.",
    )
    points: list[RateSeriesPoint] = Field(
        default_factory=list,
        description="Strictly increasing request-rate control points.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_shorthand(cls, data: Any) -> Any:
        """Allow `rateSeries: path.json` and `rateSeries: [{...}, ...]` shorthand."""
        if isinstance(data, str):
            return {"path": data}
        if isinstance(data, list):
            return {"points": data}
        return data

    @model_validator(mode="after")
    def load_path_points(self) -> RateSeriesConfig:
        """Load JSON points when a path is supplied."""
        if self.path is None:
            _validate_points(self.points)
            return self
        if self.points:
            raise ValueError(
                "rate_series.path and rate_series.points are mutually exclusive"
            )
        self.points = read_rate_series_json(self.path)
        self.path = None
        return self

    @property
    def initial_qps(self) -> float:
        """Return the first configured request rate."""
        return self.points[0].qps


def read_rate_series_json(path: str) -> list[RateSeriesPoint]:
    """Read and validate a JSON request-rate series file."""
    text = safe_read_template_path(path)
    if text is None:
        raise ValueError(f"Cannot read request-rate series JSON '{path}'")

    try:
        document = orjson.loads(text)
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"Invalid request-rate series JSON '{path}'") from exc

    if isinstance(document, list):
        points_data = document
    elif isinstance(document, dict):
        keys = set(document)
        if keys != {"points"}:
            raise ValueError(
                "Request-rate series JSON must contain exactly one top-level key: points"
            )
        points_data = document["points"]
    else:
        raise ValueError(
            "Request-rate series JSON must be an object with points or a points array"
        )

    points: list[RateSeriesPoint] = []
    if not isinstance(points_data, list):
        raise ValueError("Request-rate series JSON points must be an array")

    for index, point_data in enumerate(points_data):
        try:
            points.append(RateSeriesPoint.model_validate(point_data))
        except ValueError as exc:
            raise ValueError(
                f"Invalid request-rate series point {index}: {exc}"
            ) from exc

    _validate_points(points)
    return points


def _validate_points(points: list[RateSeriesPoint]) -> None:
    if len(points) < 2:
        raise ValueError("Request-rate series requires at least two points")
    for point in points:
        if not math.isfinite(point.time_s) or not math.isfinite(point.qps):
            raise ValueError("Request-rate series values must be finite")
    previous_time = points[0].time_s
    for point in points[1:]:
        if point.time_s <= previous_time:
            raise ValueError(
                "Request-rate series time_s values must be strictly increasing"
            )
        previous_time = point.time_s
