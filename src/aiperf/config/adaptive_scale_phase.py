# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive scale YAML lowering helpers for phase config."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.plugin.enums import PhaseType
from aiperf.timing.adaptive_types import AdaptiveControlVariable

_FIRST_TOKEN_SLA_METRICS = frozenset({"time_to_first_token", "ttft"})


def is_first_token_sla_metric(metric_tag: str) -> bool:
    """Return True when an SLA metric requires first-token observations."""
    return metric_tag in _FIRST_TOKEN_SLA_METRICS


def sla_filters_require_first_token_observation(
    sla_filters: list[SLAFilter] | tuple[SLAFilter, ...],
) -> bool:
    """Return True when any SLA filter needs first-token observations."""
    return any(is_first_token_sla_metric(sla.metric_tag) for sla in sla_filters)


def normalize_adaptive_sla(sla: dict[str, object]) -> list[SLAFilter]:
    """Lower compact metric/stat/op SLA YAML into SLAFilter objects."""
    if not isinstance(sla, dict):
        raise ValueError("adaptive_scale.sla must be a mapping or list of filters")
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
    "control_min": "adaptive_control_min",
    "controlMin": "adaptive_control_min",
    "control_max": "adaptive_control_max",
    "controlMax": "adaptive_control_max",
    "min_concurrency": "adaptive_control_min",
    "minConcurrency": "adaptive_control_min",
    "max_concurrency": "adaptive_control_max",
    "maxConcurrency": "adaptive_control_max",
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


_ADAPTIVE_SCALE_FIELD_ALIASES = {
    "adaptive_scale": "adaptiveScale",
    "adaptive_sustain_duration": "adaptiveSustainDuration",
    "adaptive_assessment_period": "adaptiveAssessmentPeriod",
    "adaptive_min_completed_requests": "adaptiveMinCompletedRequests",
    "adaptive_control_variable": "adaptiveControlVariable",
    "adaptive_control_min": "adaptiveControlMin",
    "adaptive_control_max": "adaptiveControlMax",
    "adaptive_scale_strategy_type": "adaptiveScaleStrategyType",
    "adaptive_scale_step_policy": "adaptiveScaleStepPolicy",
    "adaptive_scale_base_step": "adaptiveScaleBaseStep",
    "adaptive_scale_max_step_multiplier": "adaptiveScaleMaxStepMultiplier",
    "adaptive_scale_step_percent": "adaptiveScaleStepPercent",
}

_ADAPTIVE_SCALE_BLOCK_KEYS = {"enabled", "control", "strategy", "sla"} | set(
    _ADAPTIVE_SCALE_FIELD_MAP
)
_ADAPTIVE_SCALE_CONTROL_KEYS = {"variable", "min", "max"}
_ADAPTIVE_SCALE_STRATEGY_KEYS = set(_ADAPTIVE_SCALE_STRATEGY_FIELD_MAP)


def _copy_mapped_fields(
    lowered: dict[str, object],
    source_data: dict[str, object],
    field_map: dict[str, str],
) -> None:
    for source, target in field_map.items():
        if source in source_data:
            lowered[target] = source_data[source]


def _reject_unknown_keys(
    data: dict[str, object], allowed: set[str], scope: str
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"{scope} contains unsupported field(s): {joined}")


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
    raise ValueError("adaptive_scale.enabled must be a boolean")


def _use_adaptive_scale_aliases(lowered: dict[str, object]) -> None:
    for field_name, alias in _ADAPTIVE_SCALE_FIELD_ALIASES.items():
        if field_name in lowered:
            lowered[alias] = lowered.pop(field_name)


def lower_adaptive_scale_details(
    lowered: dict[str, object], block: dict[str, object]
) -> None:
    """Lower nested adaptive-scale YAML settings into flat phase fields."""
    _reject_unknown_keys(block, _ADAPTIVE_SCALE_BLOCK_KEYS, "adaptive_scale")
    lowered["adaptive_scale"] = _parse_enabled(block.get("enabled"))
    _copy_mapped_fields(lowered, block, _ADAPTIVE_SCALE_FIELD_MAP)

    control = block.get("control", {})
    if not isinstance(control, dict):
        raise ValueError("adaptive_scale.control must be a mapping")
    _reject_unknown_keys(
        control, _ADAPTIVE_SCALE_CONTROL_KEYS, "adaptive_scale.control"
    )
    if "variable" in control:
        lowered["adaptive_control_variable"] = control["variable"]
    if "min" in control:
        lowered["adaptive_control_min"] = control["min"]
    if "max" in control:
        lowered["adaptive_control_max"] = control["max"]

    strategy = block.get("strategy", {})
    if not isinstance(strategy, dict):
        raise ValueError("adaptive_scale.strategy must be a mapping")
    _reject_unknown_keys(
        strategy, _ADAPTIVE_SCALE_STRATEGY_KEYS, "adaptive_scale.strategy"
    )
    _copy_mapped_fields(lowered, strategy, _ADAPTIVE_SCALE_STRATEGY_FIELD_MAP)

    sla = block.get("sla")
    if sla is None:
        return
    if isinstance(sla, list):
        lowered["sla"] = sla
    elif isinstance(sla, dict):
        lowered["sla"] = normalize_adaptive_sla(sla)
    else:
        raise ValueError("adaptive_scale.sla must be a mapping or list of filters")


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
        AdaptiveControlVariable,
        Field(
            default="concurrency",
            description="Named adaptive control variable.",
        ),
    ]

    adaptive_control_min: Annotated[
        float,
        Field(
            gt=0,
            default=1,
            description="Minimum adaptive control value.",
        ),
    ]

    adaptive_control_max: Annotated[
        float | None,
        Field(
            gt=0,
            default=None,
            description="Maximum adaptive control value. Inferred from the phase target when omitted.",
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

        uses_alias = "adaptiveScale" in data and "adaptive_scale" not in data
        block = (
            data["adaptive_scale"]
            if "adaptive_scale" in data
            else data.get("adaptiveScale")
        )
        if not isinstance(block, dict):
            if (
                "adaptive_scale" in data or "adaptiveScale" in data
            ) and block is not None:
                lowered["adaptive_scale"] = _parse_enabled(block)
                if uses_alias:
                    _use_adaptive_scale_aliases(lowered)
            return lowered

        lower_adaptive_scale_details(lowered, block)
        if uses_alias:
            _use_adaptive_scale_aliases(lowered)
        return lowered

    @model_validator(mode="after")
    def _validate_adaptive_scale(self) -> Self:
        if not self.adaptive_scale:
            return self
        self._validate_adaptive_scale_required_fields()
        self._validate_adaptive_scale_phase_type()
        variable = self.adaptive_control_variable
        max_value = self._adaptive_control_max_value(variable)
        self._validate_adaptive_control_bounds(variable, max_value)
        return self

    def _validate_adaptive_scale_required_fields(self) -> None:
        if self.duration is None:
            raise ValueError("adaptive_scale requires duration")
        if self.adaptive_sustain_duration is None:
            raise ValueError("adaptive_scale requires adaptive_sustain_duration")
        if not self.sla:
            raise ValueError("adaptive_scale requires sla filters")

    def _validate_adaptive_scale_phase_type(self) -> None:
        if self.type == PhaseType.FIXED_SCHEDULE:
            raise ValueError(
                "adaptive_scale cannot be combined with fixed_schedule phases"
            )

    def _adaptive_control_max_value(self, variable: str) -> float:
        inferred_max = self._infer_adaptive_control_max(variable)
        max_value = (
            self.adaptive_control_max
            if self.adaptive_control_max is not None
            else inferred_max
        )
        if max_value is None:
            raise ValueError("adaptive_scale control.max could not be inferred")
        return max_value

    def _infer_adaptive_control_max(self, variable: str) -> float | None:
        if variable == "concurrency":
            return self._infer_concurrency_control_max()
        if variable == "prefill_concurrency":
            return self._infer_prefill_control_max()
        if variable == "request_rate":
            return self._infer_request_rate_control_max()
        if variable == "users":
            return self._infer_users_control_max()
        raise ValueError(f"unsupported adaptive control variable {variable!r}")

    def _infer_concurrency_control_max(self) -> float | None:
        if self.concurrency_ramp is not None:
            raise ValueError(
                "adaptive_scale cannot be combined with concurrency_ramp; "
                "control.variable=concurrency already adjusts concurrency"
            )
        return self.concurrency

    def _infer_prefill_control_max(self) -> float | None:
        if self.prefill_ramp is not None:
            raise ValueError(
                "adaptive_scale control.variable=prefill_concurrency cannot be "
                "combined with prefill_ramp"
            )
        if self.concurrency is None:
            raise ValueError("adaptive_scale prefill_concurrency requires concurrency")
        return self.prefill_concurrency

    def _infer_request_rate_control_max(self) -> float | None:
        inferred_max = getattr(self, "rate", None)
        if getattr(self, "rate_ramp", None) is not None:
            raise ValueError(
                "adaptive_scale control.variable=request_rate cannot be combined "
                "with rate_ramp"
            )
        if getattr(self, "rate_series", None) is not None:
            raise ValueError(
                "adaptive_scale control.variable=request_rate cannot be combined "
                "with rate_series"
            )
        if inferred_max is None:
            raise ValueError(
                "adaptive_scale request_rate requires a rate-controlled phase"
            )
        return inferred_max

    def _infer_users_control_max(self) -> float | None:
        inferred_max = getattr(self, "users", None)
        if inferred_max is None:
            raise ValueError(
                "adaptive_scale users is only valid on user_centric phases"
            )
        return inferred_max

    def _validate_adaptive_control_bounds(
        self, variable: str, max_value: float
    ) -> None:
        if max_value <= self.adaptive_control_min:
            raise ValueError("adaptive_scale control.max must be > control.min")
        self._validate_integer_adaptive_bounds(variable, max_value)
        if self._concurrency_min_exceeds_target(variable):
            raise ValueError("adaptive_scale control.min must be <= concurrency")
        if variable == "prefill_concurrency" and max_value > self.concurrency:
            raise ValueError(
                "adaptive_scale prefill_concurrency control.max must be <= concurrency"
            )

    def _validate_integer_adaptive_bounds(
        self, variable: str, max_value: float
    ) -> None:
        if variable in {"concurrency", "prefill_concurrency", "users"}:
            for field, value in (
                ("control.min", self.adaptive_control_min),
                ("control.max", max_value),
            ):
                if int(value) != value:
                    raise ValueError(
                        f"adaptive_scale {field} must be an integer for {variable}"
                    )

    def _concurrency_min_exceeds_target(self, variable: str) -> bool:
        return (
            variable == "concurrency"
            and self.concurrency is not None
            and self.adaptive_control_min > self.concurrency
        )
