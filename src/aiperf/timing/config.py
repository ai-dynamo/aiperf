# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.enums import CreditPhase
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import PhaseKind
from aiperf.config.dataset.defaults import InputDefaults
from aiperf.config.rate_series import RateSeriesConfig
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.plugin.enums import (
    ArrivalPattern,
    PhaseType,
    TimingMode,
    URLSelectionStrategy,
)
from aiperf.timing.adaptive_config import (
    ADAPTIVE_TIMING_FIELDS,
    AdaptiveControlVariable,
    AdaptiveTimingConfig,
)
from aiperf.timing.request_cancellation import RequestCancellationConfig

if TYPE_CHECKING:
    from aiperf.config.phases import PhaseConfig
    from aiperf.config.resolution.plan import BenchmarkRun


# Map ``PhaseType`` values onto the ``ArrivalPattern`` values consumed by the
# timing strategies. Concurrency / fixed_schedule phases don't use an arrival
# pattern; we still set a sensible default so downstream code paths remain
# uniform when they consult this field.
_PHASE_TYPE_TO_ARRIVAL_PATTERN: dict[PhaseType, ArrivalPattern] = {
    PhaseType.POISSON: ArrivalPattern.POISSON,
    PhaseType.GAMMA: ArrivalPattern.GAMMA,
    PhaseType.CONSTANT: ArrivalPattern.CONSTANT,
    PhaseType.USER_CENTRIC: ArrivalPattern.POISSON,
    PhaseType.CONCURRENCY: ArrivalPattern.CONCURRENCY_BURST,
    PhaseType.FIXED_SCHEDULE: ArrivalPattern.CONCURRENCY_BURST,
}


def _phase_timing_mode(phase: PhaseConfig) -> TimingMode:
    """Map a phase to the timing strategy used for credit issuance."""
    if getattr(phase, "adaptive_scale", False):
        return TimingMode.ADAPTIVE_SCALE
    if phase.type == PhaseType.FIXED_SCHEDULE:
        return TimingMode.FIXED_SCHEDULE
    if phase.type == PhaseType.USER_CENTRIC:
        return TimingMode.USER_CENTRIC_RATE
    return TimingMode.REQUEST_RATE


class TimingConfig(AIPerfBaseModel):
    """Configuration for TimingManager and timing strategies.

    Controls timing mode (REQUEST_RATE, FIXED_SCHEDULE, or USER_CENTRIC_RATE),
    rate/concurrency settings, warmup/profiling phase stop conditions, and
    request cancellation behavior.
    """

    model_config = ConfigDict(frozen=True)

    phase_configs: list[CreditPhaseConfig] = Field(
        ...,
        description="List of phase configs to execute in order. These specify the exact behavior of each phase.",
    )
    request_cancellation: RequestCancellationConfig = Field(
        default_factory=RequestCancellationConfig,
        description="Configuration for request cancellation policy.",
    )
    urls: list[str] = Field(
        default_factory=list,
        description="List of endpoint URLs for load balancing. If multiple URLs provided, "
        "requests are distributed according to url_selection_strategy.",
    )
    url_selection_strategy: URLSelectionStrategy = Field(
        default=URLSelectionStrategy.ROUND_ROBIN,
        description="Strategy for selecting URLs when multiple URLs are provided.",
    )

    @classmethod
    def from_run(cls, run: BenchmarkRun) -> TimingConfig:
        """Build ordered list of credit-phase configs from a ``BenchmarkRun``.

        Preserves the ordered ``cfg.phases`` list. Each executable phase gets
        stable identity metadata: absolute ``phase_index``, profiling-only
        ``profiling_index``, ``phase_name``, and ``phase_kind``. A phase-local
        cancellation config is attached to each timing phase; omitted per-phase
        cancellation inherits the run default.
        """
        cfg = run.cfg

        artifact_dir = cfg.artifacts.dir

        profiling_default_cancellation = _default_cancellation_config(cfg.phases)
        warmup_default_cancellation = RequestCancellationConfig()

        configs: list[CreditPhaseConfig] = []
        profiling_index = 0
        for phase_index, phase in enumerate(cfg.phases):
            current_profiling_index = None
            if phase.kind == "profiling":
                current_profiling_index = profiling_index
                profiling_index += 1
            default_cancellation = (
                profiling_default_cancellation
                if phase.kind == "profiling"
                else warmup_default_cancellation
            )
            configs.append(
                _build_phase_config(
                    phase,
                    artifact_dir=artifact_dir,
                    default_cancellation=default_cancellation,
                    phase_index=phase_index,
                    profiling_index=current_profiling_index,
                )
            )

        return cls(
            phase_configs=configs,
            request_cancellation=profiling_default_cancellation,
            urls=list(cfg.endpoint.urls),
            url_selection_strategy=cfg.endpoint.url_strategy,
        )


class CreditPhaseConfig(AIPerfBaseModel):
    """Model for credit phase config. This is used to configure a credit phase.

    Stop conditions (first one reached wins):
    - total_expected_requests: Stop after sending this many total requests
    - expected_num_sessions: Stop starting NEW user sessions after this many (complete ongoing ones)
    - expected_duration_sec: Stop after this time
    """

    model_config = ConfigDict(frozen=True)

    phase: CreditPhase = Field(..., description="The phase of the credit phase.")
    phase_index: int | None = Field(
        default=None, ge=0, description="Absolute index in the ordered phases list."
    )
    profiling_index: int | None = Field(
        default=None,
        ge=0,
        description="Index among profiling-kind phases; None for warmup.",
    )
    phase_name: str | None = Field(
        default=None, description="User-provided unique phase name."
    )
    phase_kind: PhaseKind | None = Field(
        default=None, description="Phase semantic kind: warmup or profiling."
    )
    request_cancellation: RequestCancellationConfig = Field(
        default_factory=RequestCancellationConfig,
        description="Phase-local request cancellation policy.",
    )
    timing_mode: TimingMode = Field(
        ...,
        description="The timing mode of the credit phase. Used to determine "
        "how to send requests to the workers.",
    )
    total_expected_requests: int | None = Field(
        default=None, gt=0, description="The total number of expected requests to send."
    )
    expected_num_sessions: int | None = Field(
        default=None, gt=0, description="The total number of expected sessions to send."
    )
    expected_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="The expected duration of the credit phase in seconds.",
    )
    seamless: bool = Field(
        default=False,
        description="Whether the credit phase should be seamless. "
        "Seamless phases start immediately after the previous phase sends all credits, "
        "without waiting for all credits to return. This can be used to maintain concurrency "
        "during phase transitions.",
    )
    concurrency: int | None = Field(
        default=None,
        gt=0,
        description="The max concurrency of the credit phase. "
        "This is the max number of requests that can be in flight at once. "
        "If None, the concurrency is unlimited.",
    )
    prefill_concurrency: int | None = Field(
        default=None,
        gt=0,
        description="The max concurrency of the prefill phase. "
        "This is the max number of requests that can be waiting for the first token at once. "
        "If None, the prefill concurrency is unlimited.",
    )
    request_rate: float | None = Field(
        default=None, gt=0, description="The request rate of the credit phase."
    )
    arrival_pattern: ArrivalPattern = Field(
        default=ArrivalPattern.POISSON,
        description="The arrival pattern of the credit phase.",
    )
    arrival_smoothness: float | None = Field(
        default=None,
        gt=0,
        description="The smoothness parameter for gamma distribution arrivals. "
        "Only used when arrival_pattern is GAMMA. Controls the shape of the distribution: "
        "1.0 = Poisson-like (exponential), <1.0 = bursty, >1.0 = smooth/regular. "
        "If None, defaults to 1.0 when using GAMMA arrival pattern.",
    )
    grace_period_sec: float | None = Field(
        default=None,
        ge=0,
        description="The grace period of the credit phase in seconds. "
        "This is the time to wait after the expected duration of the phase has elapsed "
        "before the phase is considered complete. This can be used to ensure that all requests "
        "have returned before the phase is considered complete. "
        "If None, the grace period is disabled.",
    )
    num_users: int | None = Field(
        default=None,
        ge=1,
        description="The number of concurrent users to use for the credit phase. "
        "This is only applicable when using user-centric rate limiting mode. ",
    )
    concurrency_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp session concurrency from 1 to target. "
        "If None, concurrency starts at target immediately.",
    )
    prefill_concurrency_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp prefill concurrency from 1 to target. "
        "If None, prefill concurrency starts at target immediately.",
    )
    request_rate_ramp_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="Duration in seconds to ramp request rate from 1 QPS to target. "
        "If None, request rate starts at target immediately.",
    )
    request_rate_series: RateSeriesConfig | None = Field(
        default=None,
        description="Piecewise-linear request-rate schedule, if enabled.",
    )
    auto_offset_timestamps: bool = Field(
        default=InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET,
        description="The auto offset timestamps of the timing manager.",
    )
    fixed_schedule_start_offset: int | None = Field(
        default=None,
        ge=0,
        description="The fixed schedule start offset of the timing manager.",
    )
    fixed_schedule_end_offset: int | None = Field(
        default=None,
        ge=0,
        description="The fixed schedule end offset of the timing manager.",
    )

    artifact_dir: Path | None = Field(
        default=None,
        description="Directory for phase-owned timing artifacts.",
    )
    adaptive: AdaptiveTimingConfig = Field(
        default_factory=AdaptiveTimingConfig,
        description="Adaptive scale timing settings.",
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_adaptive_timing_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        folded = dict(data)
        adaptive = dict(folded.get("adaptive") or {})
        for field in ADAPTIVE_TIMING_FIELDS:
            if field in folded:
                adaptive[field] = folded.pop(field)
        if adaptive:
            folded["adaptive"] = adaptive
        return folded

    def model_copy(
        self, *, update: dict[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if update:
            update = self._fold_adaptive_update(update)
        return super().model_copy(update=update, deep=deep)

    def _fold_adaptive_update(self, update: dict[str, Any]) -> dict[str, Any]:
        folded = dict(update)
        adaptive_update = {
            field: folded.pop(field)
            for field in list(folded)
            if field in ADAPTIVE_TIMING_FIELDS
        }
        if adaptive_update:
            adaptive_payload = self.adaptive.model_dump(mode="python")
            adaptive_payload.update(adaptive_update)
            folded["adaptive"] = AdaptiveTimingConfig.model_validate(adaptive_payload)
        return folded

    @property
    def adaptive_sustain_duration_sec(self) -> float | None:
        return self.adaptive.adaptive_sustain_duration_sec

    @property
    def adaptive_assessment_period_sec(self) -> float:
        return self.adaptive.adaptive_assessment_period_sec

    @property
    def adaptive_control_variable(self) -> AdaptiveControlVariable:
        return self.adaptive.adaptive_control_variable

    @property
    def adaptive_control_min(self) -> float:
        return self.adaptive.adaptive_control_min

    @property
    def adaptive_control_max(self) -> float | None:
        return self.adaptive.adaptive_control_max

    @property
    def adaptive_scale_strategy_type(self) -> Literal["ramp_until_fail"]:
        return self.adaptive.adaptive_scale_strategy_type

    @property
    def adaptive_scale_step_policy(self) -> Literal["sla_margin", "fixed_percent_step"]:
        return self.adaptive.adaptive_scale_step_policy

    @property
    def adaptive_scale_base_step(self) -> int:
        return self.adaptive.adaptive_scale_base_step

    @property
    def adaptive_scale_max_step_multiplier(self) -> int:
        return self.adaptive.adaptive_scale_max_step_multiplier

    @property
    def adaptive_scale_step_percent(self) -> float:
        return self.adaptive.adaptive_scale_step_percent

    @property
    def adaptive_min_completed_requests(self) -> int:
        return self.adaptive.adaptive_min_completed_requests

    @property
    def adaptive_sla_filters(self) -> tuple[SLAFilter, ...]:
        return self.adaptive.adaptive_sla_filters


def _ramp_duration(ramp: object | None) -> float | None:
    """Extract the ramp duration in seconds from a ``RamperConfig`` (or None)."""
    if ramp is None:
        return None
    return getattr(ramp, "duration", None)


def _phase_request_rate(phase: PhaseConfig) -> float | None:
    """Return the configured request rate for a phase, if any."""
    # Lazy import: aiperf.config.phases is only a TYPE_CHECKING import here.
    from aiperf.config.phases import get_phase_rate

    return get_phase_rate(phase)


def _phase_arrival_pattern(phase: PhaseConfig) -> ArrivalPattern:
    """Map a phase type to its arrival pattern."""
    return _PHASE_TYPE_TO_ARRIVAL_PATTERN.get(phase.type, ArrivalPattern.POISSON)


def _default_cancellation_config(
    phases: list[PhaseConfig],
) -> RequestCancellationConfig:
    for phase in phases:
        if getattr(phase, "kind", None) != "profiling":
            continue
        cancellation = getattr(phase, "cancellation", None)
        if cancellation is not None:
            return RequestCancellationConfig(
                rate=cancellation.rate, delay=cancellation.delay
            )
    return RequestCancellationConfig()


def _phase_cancellation_config(
    phase: PhaseConfig, default_cancellation: RequestCancellationConfig
) -> RequestCancellationConfig:
    cancellation = getattr(phase, "cancellation", None)
    if cancellation is None:
        return default_cancellation
    return RequestCancellationConfig(rate=cancellation.rate, delay=cancellation.delay)


def _build_phase_config(
    phase: PhaseConfig,
    *,
    artifact_dir: Path | None = None,
    default_cancellation: RequestCancellationConfig,
    phase_index: int,
    profiling_index: int | None,
) -> CreditPhaseConfig:
    if phase.kind == "warmup":
        return _build_warmup_config(
            phase,
            artifact_dir=artifact_dir,
            default_cancellation=default_cancellation,
            phase_index=phase_index,
            profiling_index=profiling_index,
        )
    return _build_profiling_config(
        phase,
        artifact_dir=artifact_dir,
        default_cancellation=default_cancellation,
        phase_index=phase_index,
        profiling_index=profiling_index,
    )


def _build_warmup_config(
    phase: PhaseConfig,
    *,
    artifact_dir: Path | None = None,
    default_cancellation: RequestCancellationConfig,
    phase_index: int,
    profiling_index: int | None,
) -> CreditPhaseConfig:
    """Build a warmup CreditPhaseConfig from a warmup PhaseConfig.

    Warmup triggers JIT compilation, memory allocation, and connection pool
    initialization so profiling measurements aren't polluted by cold-start effects.

    When the phase doesn't set ``grace_period``, default to infinity (wait
    forever for in-flight requests). This differs from the CreditPhaseConfig
    field default of None (disabled) because warmup should always complete all
    in-flight requests before transitioning to profiling.
    """
    grace_period = phase.grace_period
    if grace_period is None:
        grace_period = float("inf")

    return CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        phase_index=phase_index,
        profiling_index=profiling_index,
        phase_name=phase.name,
        phase_kind=phase.kind,
        request_cancellation=_phase_cancellation_config(phase, default_cancellation),
        # Warmup phase is always request rate timing mode
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=phase.requests,
        expected_duration_sec=phase.duration,
        expected_num_sessions=phase.sessions,
        concurrency=phase.concurrency,
        prefill_concurrency=phase.prefill_concurrency,
        request_rate=_phase_request_rate(phase),
        arrival_pattern=_phase_arrival_pattern(phase),
        arrival_smoothness=getattr(phase, "smoothness", None),
        seamless=False,
        grace_period_sec=grace_period,
        concurrency_ramp_duration_sec=_ramp_duration(phase.concurrency_ramp),
        prefill_concurrency_ramp_duration_sec=_ramp_duration(phase.prefill_ramp),
        request_rate_ramp_duration_sec=_ramp_duration(
            getattr(phase, "rate_ramp", None)
        ),
        artifact_dir=artifact_dir,
        request_rate_series=getattr(phase, "rate_series", None),
    )


def _build_profiling_config(
    phase: PhaseConfig,
    *,
    artifact_dir: Path | None = None,
    default_cancellation: RequestCancellationConfig,
    phase_index: int,
    profiling_index: int | None,
) -> CreditPhaseConfig:
    """Build a profiling CreditPhaseConfig from a profiling PhaseConfig.

    Main benchmark phase where all performance metrics are collected.
    Grace period allows in-flight requests to complete after the stop condition
    is met, ensuring metrics include requests that were sent before the deadline.
    """
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        phase_index=phase_index,
        profiling_index=profiling_index,
        phase_name=phase.name,
        phase_kind=phase.kind,
        request_cancellation=_phase_cancellation_config(phase, default_cancellation),
        timing_mode=_phase_timing_mode(phase),
        expected_duration_sec=phase.duration,
        total_expected_requests=phase.requests,
        expected_num_sessions=phase.sessions,
        concurrency=phase.concurrency,
        prefill_concurrency=phase.prefill_concurrency,
        request_rate=_phase_request_rate(phase),
        arrival_pattern=_phase_arrival_pattern(phase),
        arrival_smoothness=getattr(phase, "smoothness", None),
        seamless=phase.seamless,
        grace_period_sec=phase.grace_period,
        num_users=getattr(phase, "users", None),
        concurrency_ramp_duration_sec=_ramp_duration(phase.concurrency_ramp),
        prefill_concurrency_ramp_duration_sec=_ramp_duration(phase.prefill_ramp),
        request_rate_ramp_duration_sec=_ramp_duration(
            getattr(phase, "rate_ramp", None)
        ),
        request_rate_series=getattr(phase, "rate_series", None),
        # Fixed schedule config
        auto_offset_timestamps=getattr(
            phase, "auto_offset", InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET
        ),
        fixed_schedule_start_offset=getattr(phase, "start_offset", None),
        fixed_schedule_end_offset=getattr(phase, "end_offset", None),
        artifact_dir=artifact_dir,
        adaptive_sustain_duration_sec=getattr(phase, "adaptive_sustain_duration", None),
        adaptive_assessment_period_sec=getattr(
            phase, "adaptive_assessment_period", None
        )
        or 30.0,
        adaptive_control_variable=getattr(
            phase, "adaptive_control_variable", "concurrency"
        ),
        adaptive_control_min=getattr(phase, "adaptive_control_min", 1),
        adaptive_control_max=getattr(phase, "adaptive_control_max", None),
        adaptive_scale_strategy_type=getattr(
            phase, "adaptive_scale_strategy_type", "ramp_until_fail"
        ),
        adaptive_scale_step_policy=getattr(
            phase, "adaptive_scale_step_policy", "sla_margin"
        ),
        adaptive_scale_base_step=getattr(phase, "adaptive_scale_base_step", 10),
        adaptive_scale_max_step_multiplier=getattr(
            phase, "adaptive_scale_max_step_multiplier", 4
        ),
        adaptive_scale_step_percent=getattr(phase, "adaptive_scale_step_percent", 25.0),
        adaptive_min_completed_requests=getattr(
            phase, "adaptive_min_completed_requests", 1
        ),
        adaptive_sla_filters=tuple(getattr(phase, "sla", ()) or ()),
    )
