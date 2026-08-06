# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
AIPerf Configuration v2.0 - Phase Configuration

Discriminated union of phase types. Each concrete phase type only exposes
fields it supports; ``extra="forbid"`` rejects unknown fields structurally,
making invalid states unrepresentable.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    model_validator,
)

from aiperf.common.phase import infer_legacy_phase_kind
from aiperf.common.types import PhaseKind
from aiperf.config.adaptive_scale_phase import AdaptiveScalePhaseMixin
from aiperf.config.base import BaseConfig
from aiperf.config.cancellation import CancellationConfig
from aiperf.config.loader.duration import (
    DurationSpec,
    _normalize_duration,
    _parse_duration,
)
from aiperf.config.ramp import RampConfig, RampSpec, _normalize_ramp
from aiperf.config.rate_series import RateSeriesConfig
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.plugin.enums import PhaseType, PhaseTypeStr, RampType, TimingMode

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
    "PhaseKind",
    "PoissonPhase",
    "RampConfig",
    "RampSpec",
    "RampType",
    "RateSeriesConfig",
    "RatePhaseConfig",
    "UserCentricPhase",
    "_normalize_duration",
    "_normalize_ramp",
    "_parse_duration",
    "get_phase_rate",
]


# =============================================================================
# PHASE HIERARCHY
# =============================================================================


class BasePhaseConfig(AdaptiveScalePhaseMixin, BaseConfig):
    """Base configuration shared by all phase types.

    Not instantiated directly -- use a concrete type via the
    :data:`PhaseConfig` discriminated union.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str,
        Field(
            pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$",
            description="Unique workflow label for this phase, such as "
            "'baseline_traffic', 'cancellation_stress', or "
            "'recovery_traffic'. This is distinct from phase kind: "
            "multiple phases may share kind='profiling' while each keeps a "
            "different name. Used in logs, status, sweep targeting, artifact "
            "paths, and result file naming. Must be a strict identifier: "
            "letters, numbers, underscores, and hyphens; must start with a "
            "letter or underscore.",
        ),
    ]

    kind: Annotated[
        PhaseKind | None,
        Field(
            default=None,
            description="Semantic runtime role for the phase. Only 'warmup' "
            "and 'profiling' are valid kinds because the credit/results "
            "pipeline distinguishes those two roles. This field is nullable "
            "only as an input compatibility bridge: legacy canonical names "
            "('warmup' and 'profiling') infer kind during normalization, "
            "while validated phases always carry a concrete kind.",
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
            "Forced by phase kind: kind='warmup' is always excluded, "
            "kind='profiling' is always included. Explicitly setting this "
            "field to a value inconsistent with the phase kind is rejected.",
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

    # -------------------------------------------------------------------------
    # Agentic-replay timing (AGENTIC_REPLAY timing mode only)
    # -------------------------------------------------------------------------

    timing_mode: Annotated[
        TimingMode | None,
        Field(
            default=None,
            description="Explicit timing-strategy override for this phase. When "
            "set, it WINS over the timing mode derived from ``type`` in "
            "``aiperf.timing.config._phase_timing_mode`` (read via "
            "``_is_agentic_replay`` and ``_build_profiling_config``). This is "
            "how a benchmark-scenario invariant-lock stamps AGENTIC_REPLAY onto "
            "the profiling phase(s): the scenario validator sets "
            "``phase.timing_mode = TimingMode.AGENTIC_REPLAY`` and the credit "
            "pipeline then routes the phase through the agentic-replay strategy. "
            "Leave None for normal/dag_jsonl runs so the phase-type mapping "
            "applies (REQUEST_RATE / FIXED_SCHEDULE / USER_CENTRIC_RATE). "
            "Distinct from the sweep ``scenarios`` strategy.",
        ),
    ]

    failed_request_threshold: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Abort the run early when (failed_records / total_records) exceeds this "
            "ratio. Default None disables the check. Only PROFILING-phase records "
            "count toward the ratio. A grace floor of max(concurrency, 10) records "
            "must accumulate before the check is armed, so a single early failure "
            "cannot kill the run. When the threshold is exceeded a "
            "ProfileCancelCommand is broadcast: in-flight requests drain via the "
            "normal cancel path, partial results are still aggregated, and the run "
            "exits non-zero. Pairs with the AGENTIC_REPLAY context-overflow drop "
            "in record_processor_service so the rate measures real failures only.",
        ),
    ]

    trajectory_start_min_ratio: Annotated[
        float,
        Field(
            default=0.25,
            ge=0.0,
            le=1.0,
            description="AGENTIC_REPLAY only: lower bound (inclusive) on the random start "
            "position within each trajectory, expressed as a fraction of the "
            "trace's recorded wall-clock duration (timestamped traces) or its "
            "total turn count (legacy timestamp-less traces). Sampled per "
            "trajectory at trajectory-build "
            "time; deterministic given --random-seed.",
        ),
    ]

    trajectory_start_max_ratio: Annotated[
        float,
        Field(
            default=0.75,
            ge=0.0,
            le=1.0,
            description="AGENTIC_REPLAY only: upper bound (inclusive) on the random start "
            "position within each trajectory, expressed as a fraction of the "
            "trace's recorded wall-clock duration (timestamped traces) or its "
            "total turn count (legacy timestamp-less traces). For the "
            "timestamp-less path the effective per-trace ceiling is "
            "min(int(max_ratio * n), n - 2) so at least one profile turn remains "
            "after warmup.",
        ),
    ]

    burst_phase_starts: Annotated[
        bool,
        Field(
            default=False,
            description="AGENTIC_REPLAY only: collapse the WARMUP-start and "
            "PROFILING-start dispatches into synchronized bursts instead of "
            "spreading them by each request's recorded offset from t*. By "
            "default (False) the phase starts are SPREAD: WARMUP requests are "
            "aligned globally so every trajectory reaches its t* at the same "
            "instant (the warmup end), and each lane's first PROFILING request "
            "waits out its recorded gap after t* -- reproducing the recorded "
            "arrival pattern at both phase boundaries. The rest of the replay "
            "(inter-turn delays) is timing-faithful regardless of this flag; "
            "it governs ONLY the burst-vs-spread of the two phase starts. Pass "
            "--burst-phase-starts to fire each phase's first requests together "
            "(faster concurrency ramp, synchronized start), e.g. for a "
            "throughput-oriented run rather than a faithful arrival replay.",
        ),
    ]

    system_idle_gap_cap_seconds: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description="AGENTIC_REPLAY only: maximum time in seconds the "
            "replay may remain globally idle while future requests are "
            "scheduled. When no requests are in flight or ready, all pending "
            "request timers shift earlier by the same amount so the next "
            "request arrives within this limit. Per-trace timing, timer order, "
            "and relative spacing are otherwise preserved. None disables the "
            "global idle cap.",
        ),
    ]

    agentic_cache_warmup_duration: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="AGENTIC_REPLAY only: additional cache-pressure warmup "
            "duration in seconds. After the normal snapshot warmup drains, AIPerf "
            "continues the live trajectories without recorded idle delays and with "
            "one-token outputs for this long, then drains and resumes profiling "
            "from the resulting trajectory state. Read off the profiling phase by "
            "``timing.config._build_agentic_warmup_config``. Mutually exclusive "
            "with warmup_requests_per_lane. None disables it.",
        ),
    ]

    warmup_requests_per_lane: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="AGENTIC_REPLAY only: deterministic cache-pressure "
            "warmup request budget per concurrency lane, additional to "
            "mandatory snapshot primers. The total warmup request target is "
            "the primer count plus this value multiplied by the number of live "
            "trajectory lanes. Mutually exclusive with "
            "agentic_cache_warmup_duration.",
        ),
    ]

    agentic_warmup_grace_period: Annotated[
        float | None,
        Field(
            default=None,
            ge=0,
            description="AGENTIC_REPLAY only: grace period in seconds the "
            "auto-synthesized warmup barrier waits for in-flight priming "
            "requests after the warmup burst sends. Read off the profiling phase "
            "by ``timing.config._build_agentic_warmup_config`` (the agentic "
            "warmup is not a user-declared phase, so it does not inherit "
            "``--warmup-grace-period``, which requires ``--warmup-duration``). "
            "None waits indefinitely (the agentic warmup must complete every "
            "primed trajectory before profiling starts).",
        ),
    ]

    _failed_request_threshold_explicitly_set: bool = False
    _trajectory_start_min_ratio_explicitly_set: bool = False
    _trajectory_start_max_ratio_explicitly_set: bool = False
    _burst_phase_starts_explicitly_set: bool = False
    _system_idle_gap_cap_seconds_explicitly_set: bool = False

    # Subclasses set False to opt out (e.g. FixedSchedulePhase, where the
    # stop condition is inferred from the dataset). Otherwise CLI users
    # get autodefaults applied in the CLI->YAML converter (see
    # ``aiperf.config.flags._converter_profiling``); YAML users must be
    # explicit.
    _stop_condition_required: ClassVar[bool] = True
    _windows_reserved_phase_names: ClassVar[frozenset[str]] = frozenset(
        {"CON", "PRN", "AUX", "NUL"}
        | {f"COM{idx}" for idx in range(1, 10)}
        | {f"LPT{idx}" for idx in range(1, 10)}
    )

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @model_validator(mode="after")
    def _validate_phase_constraints(self) -> Self:
        """Validate stop condition and cross-field constraints."""
        windows_basename = self.name.split(".", 1)[0].upper()
        if windows_basename in self._windows_reserved_phase_names:
            raise ValueError(
                f"Phase name '{self.name}' is reserved by Windows and cannot "
                "be used as an artifact directory name."
            )

        self.kind = infer_legacy_phase_kind(self.name, self.kind)
        if self.kind is None:
            raise ValueError(
                f"Phase '{self.name}': kind is required for non-canonical phase "
                "names. Set kind to 'warmup' or 'profiling'."
            )
        if self.name in {"warmup", "profiling"} and self.kind != self.name:
            raise ValueError(
                f"Phase name '{self.name}' is reserved for kind '{self.name}'; "
                f"got kind '{self.kind}'."
            )

        required = self.kind == "warmup"
        if (
            "exclude_from_results" in self.model_fields_set
            and self.exclude_from_results != required
        ):
            raise ValueError(
                f"Phase '{self.name}': exclude_from_results must be "
                f"{required} for kind '{self.kind}' (warmup is always "
                "excluded; profiling is always included)"
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

    @model_validator(mode="after")
    def _record_agentic_explicit_set_flags(self) -> Self:
        """Snapshot which AGENTIC_REPLAY fields were explicitly provided.

        Scenario validation distinguishes "user explicitly set the value to a
        non-required value" (raise) from "value is at default; auto-fill from
        scenario spec" (info log). Surface stable underscore flags for the
        validator's defensive `getattr`.
        """
        self._failed_request_threshold_explicitly_set = (
            "failed_request_threshold" in self.model_fields_set
        )
        self._trajectory_start_min_ratio_explicitly_set = (
            "trajectory_start_min_ratio" in self.model_fields_set
        )
        self._trajectory_start_max_ratio_explicitly_set = (
            "trajectory_start_max_ratio" in self.model_fields_set
        )
        self._burst_phase_starts_explicitly_set = (
            "burst_phase_starts" in self.model_fields_set
        )
        self._system_idle_gap_cap_seconds_explicitly_set = (
            "system_idle_gap_cap_seconds" in self.model_fields_set
        )
        return self

    @model_validator(mode="after")
    def validate_trajectory_start_range(self) -> Self:
        """Ensure trajectory_start_min_ratio <= trajectory_start_max_ratio."""
        if self.trajectory_start_min_ratio > self.trajectory_start_max_ratio:
            raise ValueError(
                f"--trajectory-start-min-ratio ({self.trajectory_start_min_ratio}) "
                f"must be <= --trajectory-start-max-ratio "
                f"({self.trajectory_start_max_ratio})."
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


# =============================================================================
# RATE-CONTROLLED PHASES
# =============================================================================


class RatePhaseConfig(BasePhaseConfig):
    """Base for rate-controlled phases. Not instantiated directly."""

    rate: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Target request rate in requests per second. Required unless rate_series is set.",
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

    rate_series: Annotated[
        RateSeriesConfig | None,
        Field(
            default=None,
            description="Piecewise-linear request-rate schedule.",
        ),
    ]

    @model_validator(mode="after")
    def validate_rate_source(self) -> Self:
        """Require exactly one of a scalar rate or a rate series."""
        if self.rate is None and self.rate_series is None:
            raise ValueError("rate-controlled phases require rate or rate_series")
        if self.rate is not None and self.rate_series is not None:
            raise ValueError("rate and rate_series are mutually exclusive")
        return self


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
        if self.rate_series is not None:
            raise ValueError("user-centric phases do not support rate_series")

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


def get_phase_rate(phase: BasePhaseConfig) -> float | None:
    """Return the configured request rate for a phase, or None for non-rate phases.

    Single accessor for the ``rate`` field so a future rename fails fast here
    instead of being silently swallowed by scattered ``getattr(..., None)`` reads.
    """
    if not isinstance(phase, RatePhaseConfig):
        return None
    if phase.rate is not None:
        return phase.rate
    if phase.rate_series is not None:
        return phase.rate_series.initial_qps
    return None
