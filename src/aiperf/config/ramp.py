# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ramp configuration shared by phase configuration types."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field

from aiperf.config.base import BaseConfig
from aiperf.config.loader.duration import _parse_duration
from aiperf.plugin.enums import RampType


class RampConfig(BaseConfig):
    """
    Configuration for gradual value ramping.

    Controls how a value (concurrency, rate, etc.) transitions from
    start to target over time.
    """

    model_config = ConfigDict(extra="forbid")

    duration: Annotated[
        float,
        Field(
            gt=0.0,
            description="Seconds to ramp from start to target value.",
        ),
    ]

    strategy: Annotated[
        RampType,
        Field(
            default=RampType.LINEAR,
            description="Ramp curve shape: "
            "linear (constant rate), "
            "exponential (slow start, fast finish), "
            "poisson (stochastic with guaranteed completion).",
        ),
    ]


def _normalize_ramp(v: Any) -> Any:
    """Normalize ramp shorthand to RampConfig dict."""
    if v is None:
        return None
    if isinstance(v, (int, float, str)):
        duration = _parse_duration(v)
        return {"duration": duration}
    if isinstance(v, dict) and "duration" in v:
        v["duration"] = _parse_duration(v["duration"])
    return v


# Type alias for ramp that supports shorthand (just duration as number or string)
RampSpec = Annotated[RampConfig | None, BeforeValidator(_normalize_ramp)]
