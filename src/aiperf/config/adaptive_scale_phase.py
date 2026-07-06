# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive scale YAML lowering helpers for phase config."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from aiperf.config.sweep.adaptive import SLAFilter


def normalize_adaptive_sla(sla: dict[str, object]) -> list[SLAFilter]:
    """Lower compact metric/stat/op SLA YAML into SLAFilter objects."""
    filters: list[SLAFilter] = []
    for metric_tag, stats in sla.items():
        if not isinstance(stats, dict):
            raise ValueError("adaptive_scale.sla entries must map metric tags to stats")
        for stat, ops in stats.items():
            if not isinstance(ops, dict):
                raise ValueError(
                    "adaptive_scale.sla stats must map operators to thresholds"
                )
            for op, threshold in ops.items():
                filters.append(
                    SLAFilter(
                        metric_tag=metric_tag,
                        stat=stat,
                        op=op,
                        threshold=threshold,
                    )
                )
    return filters


_ADAPTIVE_SCALE_FIELD_MAP = {
    "control_variable": "adaptive_control_variable",
    "controlVariable": "adaptive_control_variable",
    "min_concurrency": "adaptive_scale_min_concurrency",
    "minConcurrency": "adaptive_scale_min_concurrency",
    "window": "adaptive_assessment_period",
    "assessment_period": "adaptive_assessment_period",
    "assessmentPeriod": "adaptive_assessment_period",
    "min_completed_requests": "adaptive_min_completed_requests",
    "minCompletedRequests": "adaptive_min_completed_requests",
    "sustain_duration": "adaptive_sustain_duration",
    "sustainDuration": "adaptive_sustain_duration",
}


_ADAPTIVE_SCALE_STRATEGY_FIELD_MAP = {
    "type": "adaptive_scale_strategy_type",
    "step_policy": "adaptive_scale_step_policy",
    "stepPolicy": "adaptive_scale_step_policy",
    "base_step": "adaptive_scale_base_step",
    "baseStep": "adaptive_scale_base_step",
    "max_step_multiplier": "adaptive_scale_max_step_multiplier",
    "maxStepMultiplier": "adaptive_scale_max_step_multiplier",
    "step_percent": "adaptive_scale_step_percent",
    "stepPercent": "adaptive_scale_step_percent",
}


def _copy_mapped_fields(
    lowered: dict[str, object],
    source_data: dict[str, object],
    field_map: dict[str, str],
) -> None:
    for source, target in field_map.items():
        if source in source_data:
            lowered[target] = source_data[source]


def _parse_enabled(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
        raise ValueError("adaptive_scale.enabled must be a boolean")
    return bool(value)


def lower_adaptive_scale_details(
    lowered: dict[str, object], block: dict[str, object]
) -> None:
    """Lower nested adaptive-scale YAML settings into flat phase fields."""
    lowered["adaptive_scale"] = _parse_enabled(block.get("enabled"))
    _copy_mapped_fields(lowered, block, _ADAPTIVE_SCALE_FIELD_MAP)

    strategy = block.get("strategy", {})
    if isinstance(strategy, dict):
        _copy_mapped_fields(lowered, strategy, _ADAPTIVE_SCALE_STRATEGY_FIELD_MAP)

    sla = block.get("sla")
    if isinstance(sla, list):
        lowered["sla"] = sla
    elif isinstance(sla, dict):
        lowered["sla"] = normalize_adaptive_sla(sla)


class AdaptiveScalePhaseMixin:
    """Adaptive scale fields and validation for concurrency phases."""

    adaptive_scale: Annotated[
        bool,
        Field(
            default=False,
            description="Enable single-run adaptive scale control for this phase.",
        ),
    ]

    adaptive_sustain_duration: Annotated[
        float | None,
        Field(
            gt=0,
            default=None,
            description="Duration in seconds to sustain load near the discovered adaptive scale boundary.",
        ),
    ]

    adaptive_assessment_period: Annotated[
        float | None,
        Field(
            ge=1.0,
            default=None,
            description="Duration in seconds for each adaptive scale SLA assessment window.",
        ),
    ]

    adaptive_min_completed_requests: Annotated[
        int,
        Field(
            ge=1,
            default=1,
            description="Minimum completed requests needed before an adaptive SLA window can make a decision.",
        ),
    ]

    adaptive_control_variable: Annotated[
        Literal["concurrency"],
        Field(
            default="concurrency",
            description="Named adaptive control variable. Only concurrency is supported in v1.",
        ),
    ]

    adaptive_scale_min_concurrency: Annotated[
        int,
        Field(
            ge=1,
            default=1,
            description="Minimum concurrency used by adaptive scale discovery.",
        ),
    ]

    adaptive_scale_strategy_type: Annotated[
        Literal["ramp_until_fail"],
        Field(
            default="ramp_until_fail",
            description="Adaptive scale controller strategy. v1 supports ramp_until_fail.",
        ),
    ]

    adaptive_scale_step_policy: Annotated[
        Literal["sla_margin", "fixed_percent_step"],
        Field(
            default="sla_margin",
            description=(
                "Adaptive scale increase policy. sla_margin uses normalized SLA "
                "margin to choose larger steps when far from the boundary; "
                "fixed_percent_step uses a fixed percentage of the current control value."
            ),
        ),
    ]

    adaptive_scale_base_step: Annotated[
        int,
        Field(
            ge=1,
            default=10,
            description="Minimum adaptive scale step for SLA-margin policy.",
        ),
    ]

    adaptive_scale_max_step_multiplier: Annotated[
        int,
        Field(
            ge=1,
            default=4,
            description="Maximum base-step multiplier for SLA-margin policy.",
        ),
    ]

    adaptive_scale_step_percent: Annotated[
        float,
        Field(
            gt=0,
            default=25.0,
            description="Percent of current concurrency used by fixed-percent adaptive scaling.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def _lower_adaptive_scale_block(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        lowered = dict(data)
        if isinstance(lowered.get("sla"), dict):
            lowered["sla"] = normalize_adaptive_sla(lowered["sla"])

        block = data.get("adaptive_scale")
        if not isinstance(block, dict):
            return lowered

        lower_adaptive_scale_details(lowered, block)
        return lowered

    @model_validator(mode="after")
    def _validate_adaptive_scale(self) -> Self:
        if not self.adaptive_scale:
            return self
        if self.duration is None:
            raise ValueError("adaptive_scale requires duration")
        if self.adaptive_sustain_duration is None:
            raise ValueError("adaptive_scale requires adaptive_sustain_duration")
        if not self.sla:
            raise ValueError("adaptive_scale requires sla filters")
        if self.concurrency_ramp is not None:
            raise ValueError(
                "adaptive_scale cannot be combined with concurrency_ramp. "
                "adaptive_scale already adjusts concurrency during the phase to "
                "discover an SLA boundary. Use concurrency_ramp only when you know "
                "the target concurrency and want to ease into it over a fixed duration."
            )
        # TODO: AIP-967 - Add adaptive scale control-backend abstraction.
        if self.adaptive_control_variable != "concurrency":
            raise ValueError(
                "adaptive_scale control variable must be 'concurrency' in this release"
            )
        if self.adaptive_scale_min_concurrency > self.concurrency:
            raise ValueError("adaptive_scale_min_concurrency must be <= concurrency")
        return self
