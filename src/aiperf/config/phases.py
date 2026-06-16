# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Phase Configuration

Discriminated union of phase types. Each concrete phase type only exposes
fields it supports; ``extra="forbid"`` rejects unknown fields structurally,
making invalid states unrepresentable.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    model_validator,
)
from typing_extensions import Self

from aiperf.config.base import BaseConfig
from aiperf.config.cancellation import CancellationConfig
from aiperf.config.loader.duration import (
    DurationSpec,
    _normalize_duration,
    _parse_duration,
)
from aiperf.config.ramp import RampConfig, RampSpec, _normalize_ramp
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.plugin.enums import PhaseType, PhaseTypeStr, RampType

__all__ = [
    "BasePhaseConfig",
    "CancellationConfig",
    "ConcurrencyPhase",
    "ConstantPhase",
    "DurationSpec",
    "FixedSchedulePhase",
    "GammaPhase",
    "PhaseConfig",
    "PhaseType",
    "PhaseTypeStr",
    "PoissonPhase",
    "RampConfig",
    "RampSpec",
    "RampType",
    "RatePhaseConfig",
    "UserCentricPhase",
    "_normalize_duration",
    "_normalize_ramp",
    "_parse_duration",
]


def _normalize_adaptive_sla(sla: dict[str, object]) -> list[SLAFilter]:
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


# =============================================================================
# PHASE HIERARCHY
# =============================================================================


class BasePhaseConfig(BaseConfig):
    """Base configuration shared by all phase types.

    Not instantiated directly -- use a concrete type via the
    :data:`PhaseConfig` discriminated union.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        Literal["warmup", "profiling"],
        Field(
            description="Phase identifier — must be 'warmup' or 'profiling'. "
            "The credit pipeline only distinguishes these two phase kinds. "
            "Used in logs, status, sweep targeting, and result file naming.",
        ),
    ]

    # Narrowed to Literal in each concrete class; declared here so that
    # code holding a BasePhaseConfig reference can always access .type.
    type: Annotated[
        PhaseType,
        Field(
            description="Load generation type. "
            "concurrency: concurrency-controlled immediate dispatch, "
            "poisson/gamma/constant: rate-controlled with arrival distribution, "
            "user_centric: N users sharing global rate, "
            "fixed_schedule: replay from timestamps.",
        ),
    ]

    # =========================================================================
    # UNIVERSAL FIELDS
    # =========================================================================

    exclude_from_results: Annotated[
        bool,
        Field(
            default=False,
            description="Exclude this phase's metrics from final results. "
            "Forced by phase name: 'warmup' is always excluded, "
            "'profiling' is always included. Explicitly setting this "
            "field to a value inconsistent with the phase name is rejected.",
        ),
    ]

    # -------------------------------------------------------------------------
    # Stop Conditions (at least one required unless _stop_condition_required=False)
    # -------------------------------------------------------------------------

    requests: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Stop after this many requests sent (must be >= 1).",
        ),
    ]

    duration: Annotated[
        DurationSpec,
        Field(
            gt=0,
            default=None,
            description="Stop after this time elapsed (must be > 0). Supports: 300, '5m', '2h'.",
        ),
    ]

    sessions: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Stop after this many sessions completed (must be >= 1).",
        ),
    ]

    # -------------------------------------------------------------------------
    # Concurrency Control
    # -------------------------------------------------------------------------

    concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Max concurrent in-flight requests (must be >= 1). "
            "For concurrency type: primary control. "
            "For rate types: acts as a cap.",
        ),
    ]

    concurrency_ramp: Annotated[
        RampSpec,
        Field(
            default=None,
            description="Ramp concurrency from lower value. "
            "Can be number (seconds) or {duration, strategy}.",
        ),
    ]

    prefill_concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Max concurrent requests in prefill stage (must be >= 1). "
            "Limits requests before first token received.",
        ),
    ]

    prefill_ramp: Annotated[
        RampSpec,
        Field(
            default=None,
            description="Ramp prefill_concurrency from lower value. "
            "Can be number (seconds) or {duration, strategy}.",
        ),
    ]

    # -------------------------------------------------------------------------
    # Transition Settings
    # -------------------------------------------------------------------------

    grace_period: Annotated[
        DurationSpec,
        Field(
            ge=0,
            default=None,
            description="Seconds to wait for in-flight requests after duration expires (must be >= 0). "
            "Requires 'duration' to be set. Supports: 30, '30s', '2m'.",
        ),
    ]

    cancellation: Annotated[
        CancellationConfig | None,
        Field(
            default=None,
            description="Request cancellation testing configuration.",
        ),
    ]

    sla: Annotated[
        list[SLAFilter],
        Field(
            default_factory=list,
            description="SLA filters evaluated by adaptive load controllers.",
        ),
    ]

    seamless: Annotated[
        bool,
        Field(
            default=False,
            description="Start this phase immediately when previous phase stops, "
            "without waiting for in-flight requests to complete. "
            "Cannot be True for the first phase.",
        ),
    ]

    # Subclasses set False to opt out (e.g. FixedSchedulePhase, where the
    # stop condition is inferred from the dataset). Otherwise CLI users
    # get autodefaults applied in the CLI->YAML converter (see
    # ``aiperf.config.flags._converter_profiling``); YAML users must be
    # explicit.
    _stop_condition_required: ClassVar[bool] = True

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @model_validator(mode="after")
    def _validate_phase_constraints(self) -> Self:
        """Validate stop condition and cross-field constraints."""
        required = {"warmup": True, "profiling": False}.get(self.name)
        if required is not None:
            if (
                "exclude_from_results" in self.model_fields_set
                and self.exclude_from_results != required
            ):
                raise ValueError(
                    f"Phase '{self.name}': exclude_from_results must be "
                    f"{required} (warmup is always excluded; profiling is "
                    f"always included)"
                )
            if self.exclude_from_results != required:
                self.exclude_from_results = required
        if (
            self._stop_condition_required
            and self.requests is None
            and self.duration is None
            and self.sessions is None
        ):
            raise ValueError(
                f"Phase '{self.name}': at least one of "
                "'requests', 'duration', or 'sessions' must be specified"
            )
        if (
            self.prefill_concurrency is not None
            and self.concurrency is not None
            and self.prefill_concurrency > self.concurrency
        ):
            raise ValueError(
                f"Phase '{self.name}': prefill_concurrency must be <= concurrency"
            )
        if self.grace_period is not None and self.duration is None:
            raise ValueError(
                f"Phase '{self.name}': grace_period requires duration to be set"
            )
        return self


# =============================================================================
# CONCURRENCY PHASE
# =============================================================================


class ConcurrencyPhase(BasePhaseConfig):
    """Concurrency-controlled load: dispatch immediately when a slot opens.

    Primary control is ``concurrency`` (defaults to 1).
    No rate limiting -- pure concurrency-based throughput.
    """

    type: Annotated[
        Literal[PhaseType.CONCURRENCY],
        Field(description="Concurrency-controlled immediate dispatch."),
    ]

    concurrency: Annotated[
        int,
        Field(
            ge=1,
            default=1,
            description="Max concurrent in-flight requests (must be >= 1). "
            "Primary control for concurrency phases.",
        ),
    ]

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
            lowered["sla"] = _normalize_adaptive_sla(lowered["sla"])

        block = data.get("adaptive_scale")
        if not isinstance(block, dict):
            return lowered

        lowered["adaptive_scale"] = bool(block.get("enabled", True))

        field_map = {
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
        for source, target in field_map.items():
            if source in block:
                lowered[target] = block[source]

        strategy = block.get("strategy", {})
        if isinstance(strategy, dict):
            strategy_map = {
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
            for source, target in strategy_map.items():
                if source in strategy:
                    lowered[target] = strategy[source]

        sla = block.get("sla")
        if isinstance(sla, list):
            lowered["sla"] = sla
        elif isinstance(sla, dict):
            lowered["sla"] = _normalize_adaptive_sla(sla)
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
        # v1 keeps the controller concrete: it drives ConcurrencyManager's
        # session limit directly. Add a control-backend abstraction when a
        # second variable, such as user count, is implemented end to end.
        if self.adaptive_control_variable != "concurrency":
            raise ValueError(
                "adaptive_scale control variable must be 'concurrency' in this release"
            )
        if self.adaptive_scale_min_concurrency > self.concurrency:
            raise ValueError("adaptive_scale_min_concurrency must be <= concurrency")
        return self


# =============================================================================
# RATE-CONTROLLED PHASES
# =============================================================================


class RatePhaseConfig(BasePhaseConfig):
    """Base for rate-controlled phases. Not instantiated directly."""

    rate: Annotated[
        float,
        Field(
            gt=0,
            description="Target request rate in requests per second (must be > 0).",
        ),
    ]

    rate_ramp: Annotated[
        RampSpec,
        Field(
            default=None,
            description="Ramp rate from lower value. "
            "Can be number (seconds) or {duration, strategy}.",
        ),
    ]


class PoissonPhase(RatePhaseConfig):
    """Poisson-distributed request arrivals at the target rate."""

    type: Annotated[
        Literal[PhaseType.POISSON],
        Field(description="Poisson-distributed rate-controlled arrivals."),
    ]


class GammaPhase(RatePhaseConfig):
    """Gamma-distributed request arrivals with configurable smoothness."""

    type: Annotated[
        Literal[PhaseType.GAMMA],
        Field(description="Gamma-distributed rate-controlled arrivals."),
    ]

    smoothness: Annotated[
        float | None,
        Field(
            gt=0,
            default=None,
            description="Gamma distribution shape parameter (must be > 0). "
            "1.0 = Poisson, <1 = bursty, >1 = regular.",
        ),
    ]


class ConstantPhase(RatePhaseConfig):
    """Constant-rate request arrivals (fixed inter-arrival time)."""

    type: Annotated[
        Literal[PhaseType.CONSTANT],
        Field(description="Constant rate-controlled arrivals."),
    ]


class UserCentricPhase(RatePhaseConfig):
    """N concurrent users sharing a global request rate.

    Requires multi-turn conversations. Each user gets a proportional
    share of the global ``rate``.
    """

    type: Annotated[
        Literal[PhaseType.USER_CENTRIC],
        Field(description="N users sharing a global request rate."),
    ]

    users: Annotated[
        int,
        Field(
            ge=1,
            description="Number of simulated concurrent users (must be >= 1). "
            "Requests distributed across users to achieve global rate.",
        ),
    ]

    @model_validator(mode="after")
    def validate_user_centric_constraints(self) -> UserCentricPhase:
        """Validate user-centric mode constraints."""
        if self.sessions is not None and self.sessions < self.users:
            raise ValueError(
                f"Phase '{self.name}': --num-sessions ({self.sessions}) must be "
                f">= --num-users ({self.users}). Each user needs at least one session."
            )

        if self.requests is not None and self.requests < self.users:
            raise ValueError(
                f"Phase '{self.name}': --request-count ({self.requests}) must be "
                f">= --num-users ({self.users}). Each user needs at least one request."
            )

        return self


# =============================================================================
# FIXED SCHEDULE PHASE
# =============================================================================


class FixedSchedulePhase(BasePhaseConfig):
    """Replay requests at predetermined timestamps from a trace dataset.

    Stop condition not required -- the trace dataset determines when the
    phase ends.
    """

    _stop_condition_required: ClassVar[bool] = False

    type: Annotated[
        Literal[PhaseType.FIXED_SCHEDULE],
        Field(description="Replay requests at trace timestamps."),
    ]

    auto_offset: Annotated[
        bool,
        Field(
            default=True,
            description="Normalize trace timestamps to start at 0. "
            "Subtracts minimum timestamp from all entries.",
        ),
    ]

    start_offset: Annotated[
        int | None,
        Field(
            ge=0,
            default=None,
            description="Filter out trace requests before this timestamp in ms (must be >= 0).",
        ),
    ]

    end_offset: Annotated[
        int | None,
        Field(
            ge=0,
            default=None,
            description="Filter out trace requests after this timestamp in ms (must be >= 0).",
        ),
    ]

    @model_validator(mode="after")
    def _validate_fixed_schedule_constraints(self) -> Self:
        if self.auto_offset and self.start_offset is not None:
            raise ValueError("auto_offset cannot be True when start_offset is set")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.start_offset > self.end_offset
        ):
            raise ValueError("start_offset must be <= end_offset")
        return self


# =============================================================================
# DISCRIMINATED UNION
# =============================================================================

PhaseConfig = Annotated[
    ConcurrencyPhase
    | PoissonPhase
    | GammaPhase
    | ConstantPhase
    | UserCentricPhase
    | FixedSchedulePhase,
    Discriminator("type"),
]
