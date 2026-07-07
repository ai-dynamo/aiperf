# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict, Field

from aiperf.common.enums import CreditPhase
from aiperf.config import InputDefaults

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig
    from aiperf.config.phases import BasePhaseConfig
    from aiperf.config.sweep.adaptive import SLAFilter

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.plugin.enums import (
    ArrivalPattern,
    PhaseType,
    TimingMode,
    URLSelectionStrategy,
)
from aiperf.timing.adaptive_config import ADAPTIVE_TIMING_FIELDS, AdaptiveTimingConfig
from aiperf.timing.request_cancellation import RequestCancellationConfig


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
    def from_user_config(cls, config: BenchmarkConfig) -> TimingConfig:
        """Alias for from_config (backward compatibility)."""
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: BenchmarkConfig) -> TimingConfig:
        """Build TimingConfig from AIPerfConfig phases in config order.

        Each phase uses its `name` field as the phase name and preserves
        exclude_from_results from the config.
        """
        phase_configs: list[CreditPhaseConfig] = []
        cancellation = RequestCancellationConfig()
        artifact_dir = config.artifacts.artifact_directory

        for phase in config.phases:
            phase_config = _build_credit_phase_config(
                phase,
                phase_name=phase.name,
                exclude_from_results=phase.exclude_from_results,
                artifact_dir=artifact_dir,
            )
            phase_configs.append(phase_config)

            # Use first non-excluded phase's cancellation as global cancellation
            if (
                not phase.exclude_from_results
                and phase.cancellation
                and cancellation.rate is None
            ):
                cancellation = RequestCancellationConfig(
                    rate=phase.cancellation.rate,
                    delay=phase.cancellation.delay,
                )

        return cls(
            phase_configs=phase_configs,
            request_cancellation=cancellation,
            urls=config.endpoint.urls,
            url_selection_strategy=config.endpoint.url_strategy,
        )


@dataclass(slots=True, kw_only=True, frozen=True)
class CreditPhaseConfig:
    """Config for a single credit phase.

    Slotted dataclass — shared type for both msgspec envelopes (e.g.
    ``CreditPhaseStartMessage.config``) and the Pydantic ``TimingConfig``
    parent that hosts ``list[CreditPhaseConfig]``. Fields are primitives or
    enums except the nested ``adaptive`` (``AdaptiveTimingConfig``), which
    the message codec's ``_msgspec_enc_hook`` / ``_msgspec_dec_hook`` project
    to/from a dict on the wire.

    Stop conditions (first one reached wins):
    - total_expected_requests: Stop after sending this many total requests
    - expected_num_sessions: Stop starting NEW user sessions after this many (complete ongoing ones)
    - expected_duration_sec: Stop after this time
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    phase: CreditPhase
    timing_mode: TimingMode
    exclude_from_results: bool = False
    total_expected_requests: int | None = None
    expected_num_sessions: int | None = None
    expected_duration_sec: float | None = None
    seamless: bool = False
    concurrency: int | None = None
    prefill_concurrency: int | None = None
    request_rate: float | None = None
    arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON
    # Only used when arrival_pattern is GAMMA. Controls the shape of the
    # distribution: 1.0 = Poisson-like (exponential), <1.0 = bursty,
    # >1.0 = smooth/regular. If None, defaults to 1.0 when using GAMMA.
    arrival_smoothness: float | None = None
    grace_period_sec: float | None = None
    # Only applicable for user-centric rate-limiting mode.
    num_users: int | None = None
    concurrency_ramp_duration_sec: float | None = None
    prefill_concurrency_ramp_duration_sec: float | None = None
    request_rate_ramp_duration_sec: float | None = None
    auto_offset_timestamps: bool = InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET
    fixed_schedule_start_offset: int | None = None
    fixed_schedule_end_offset: int | None = None

    # Directory for phase-owned timing artifacts (adaptive-scale decision log,
    # etc.). None disables artifact writing.
    artifact_dir: Path | None = None
    # Adaptive-scale timing settings. Only consulted when timing_mode is
    # ADAPTIVE_SCALE; defaults are inert for every other mode.
    adaptive: AdaptiveTimingConfig = field(default_factory=AdaptiveTimingConfig)

    @property
    def adaptive_sustain_duration_sec(self) -> float | None:
        return self.adaptive.adaptive_sustain_duration_sec

    @property
    def adaptive_assessment_period_sec(self) -> float:
        return self.adaptive.adaptive_assessment_period_sec

    @property
    def adaptive_control_variable(self) -> Literal["concurrency"]:
        return self.adaptive.adaptive_control_variable

    @property
    def adaptive_scale_min_concurrency(self) -> int:
        return self.adaptive.adaptive_scale_min_concurrency

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

    def model_copy(
        self, *, update: dict[str, Any] | None = None, deep: bool = False
    ) -> CreditPhaseConfig:
        """Return a copy with ``update`` applied.

        Flat ``adaptive_*`` keys are folded into the nested ``adaptive``
        sub-config so callers can tweak individual adaptive settings without
        rebuilding the whole ``AdaptiveTimingConfig``. Named ``model_copy`` to
        mirror the Pydantic ergonomics the rest of the config layer exposes.
        """
        updates = dict(update or {})
        adaptive_updates = {
            key: updates.pop(key)
            for key in list(updates)
            if key in ADAPTIVE_TIMING_FIELDS
        }
        if adaptive_updates:
            updates["adaptive"] = self.adaptive.model_copy(update=adaptive_updates)
        return replace(self, **updates)


def _phase_type_to_timing(phase_type: PhaseType) -> tuple[TimingMode, ArrivalPattern]:
    """Map PhaseType to (TimingMode, ArrivalPattern).

    Delegates to the shared resolution function in config.resolved.
    """
    from aiperf.config.resolution.predicates import get_phase_timing

    return get_phase_timing(phase_type)


def _build_adaptive_timing_config(phase: BasePhaseConfig) -> AdaptiveTimingConfig:
    """Fold a phase's flat ``adaptive_*`` fields into an AdaptiveTimingConfig.

    Uses getattr defaults so non-concurrency phase types (which do not mix in
    ``AdaptiveScalePhaseMixin``) build an inert config.
    """
    return AdaptiveTimingConfig(
        adaptive_sustain_duration_sec=getattr(phase, "adaptive_sustain_duration", None),
        adaptive_assessment_period_sec=getattr(
            phase, "adaptive_assessment_period", None
        )
        or 30.0,
        adaptive_control_variable=getattr(
            phase, "adaptive_control_variable", "concurrency"
        ),
        adaptive_scale_min_concurrency=getattr(
            phase, "adaptive_scale_min_concurrency", 1
        ),
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


def _build_credit_phase_config(
    phase: BasePhaseConfig,
    *,
    phase_name: str,
    exclude_from_results: bool,
    artifact_dir: Path | None = None,
) -> CreditPhaseConfig:
    """Build a CreditPhaseConfig from a phase config.

    Maps the AIPerfConfig phase structure to the internal
    CreditPhaseConfig used by the timing system. Uses getattr for
    fields that only exist on specific phase types.

    For excluded phases (exclude_from_results=True), grace_period defaults to infinity
    to ensure all in-flight requests complete before the next phase begins.

    When the phase enables ``adaptive_scale``, the timing mode is overridden to
    ADAPTIVE_SCALE regardless of the phase type's default mapping.
    """
    timing_mode, arrival_pattern = _phase_type_to_timing(phase.type)

    if getattr(phase, "adaptive_scale", False):
        timing_mode = TimingMode.ADAPTIVE_SCALE

    grace_period = phase.grace_period
    if exclude_from_results and grace_period is None:
        grace_period = float("inf")

    rate_ramp = getattr(phase, "rate_ramp", None)

    return CreditPhaseConfig(
        phase=CreditPhase(phase_name),
        exclude_from_results=exclude_from_results,
        timing_mode=timing_mode,
        arrival_pattern=arrival_pattern,
        total_expected_requests=phase.requests,
        expected_duration_sec=phase.duration,
        expected_num_sessions=phase.sessions,
        concurrency=phase.concurrency,
        prefill_concurrency=phase.prefill_concurrency,
        request_rate=getattr(phase, "rate", None),
        arrival_smoothness=getattr(phase, "smoothness", None),
        num_users=getattr(phase, "users", None),
        grace_period_sec=grace_period,
        seamless=phase.seamless,
        auto_offset_timestamps=getattr(phase, "auto_offset", True),
        fixed_schedule_start_offset=getattr(phase, "start_offset", None),
        fixed_schedule_end_offset=getattr(phase, "end_offset", None),
        concurrency_ramp_duration_sec=phase.concurrency_ramp.duration if phase.concurrency_ramp else None,
        prefill_concurrency_ramp_duration_sec=phase.prefill_ramp.duration if phase.prefill_ramp else None,
        request_rate_ramp_duration_sec=rate_ramp.duration if rate_ramp else None,
        artifact_dir=artifact_dir,
        adaptive=_build_adaptive_timing_config(phase),
    )  # fmt: skip
