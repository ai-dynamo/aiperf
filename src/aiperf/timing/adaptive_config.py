# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.config.sweep.adaptive import SLAFilter

ADAPTIVE_TIMING_FIELDS = frozenset(
    {
        "adaptive_sustain_duration_sec",
        "adaptive_assessment_period_sec",
        "adaptive_control_variable",
        "adaptive_scale_min_concurrency",
        "adaptive_scale_strategy_type",
        "adaptive_scale_step_policy",
        "adaptive_scale_base_step",
        "adaptive_scale_max_step_multiplier",
        "adaptive_scale_step_percent",
        "adaptive_min_completed_requests",
        "adaptive_sla_filters",
    }
)


class AdaptiveTimingConfig(AIPerfBaseModel):
    """Adaptive scale timing settings for a credit phase."""

    model_config = ConfigDict(frozen=True)

    adaptive_sustain_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to sustain load after adaptive scale discovery.",
    )
    adaptive_assessment_period_sec: float = Field(
        default=30.0,
        ge=1.0,
        description="Duration in seconds for each adaptive scale SLA assessment window.",
    )
    adaptive_control_variable: Literal["concurrency"] = Field(
        default="concurrency",
        description="Adaptive scale control variable.",
    )
    adaptive_scale_min_concurrency: int = Field(
        default=1,
        ge=1,
        description="Minimum concurrency used by adaptive scale discovery.",
    )
    adaptive_scale_strategy_type: Literal["ramp_until_fail"] = Field(
        default="ramp_until_fail",
        description="Adaptive scale strategy type.",
    )
    adaptive_scale_step_policy: Literal["sla_margin", "fixed_percent_step"] = Field(
        default="sla_margin",
        description="Adaptive scale step policy.",
    )
    adaptive_scale_base_step: int = Field(
        default=10,
        ge=1,
        description="Minimum adaptive scale step for SLA-margin policy.",
    )
    adaptive_scale_max_step_multiplier: int = Field(
        default=4,
        ge=1,
        description="Maximum base-step multiplier for SLA-margin policy.",
    )
    adaptive_scale_step_percent: float = Field(
        default=25.0,
        gt=0,
        description="Percent of current concurrency used by fixed-percent adaptive scaling.",
    )
    adaptive_min_completed_requests: int = Field(
        default=1,
        ge=1,
        description="Minimum completed requests needed before an adaptive SLA decision.",
    )
    adaptive_sla_filters: tuple[SLAFilter, ...] = Field(
        default_factory=tuple,
        description="SLA filters used by adaptive scale.",
    )
