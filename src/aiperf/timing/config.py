# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from pydantic import ConfigDict, Field

from aiperf.common.enums import CreditPhase
from aiperf.config import InputDefaults

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig
    from aiperf.config.phases import BasePhaseConfig

from aiperf.common.models.base_models import AIPerfBaseModel, PydanticStructMixin
from aiperf.plugin.enums import (
    ArrivalPattern,
    PhaseType,
    TimingMode,
    URLSelectionStrategy,
)
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

        Each phase uses its dict key as the phase name and preserves
        exclude_from_results from the config.
        """
        phase_configs: list[CreditPhaseConfig] = []
        cancellation = RequestCancellationConfig()

        for name, phase in config.phases.items():
            phase_config = _build_credit_phase_config(
                phase, phase_name=name, exclude_from_results=phase.exclude_from_results
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


class CreditPhaseConfig(
    PydanticStructMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Config for a single credit phase.

    Stop conditions (first one reached wins):
    - total_expected_requests: Stop after sending this many total requests
    - expected_num_sessions: Stop starting NEW user sessions after this many (complete ongoing ones)
    - expected_duration_sec: Stop after this time
    """

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


def _phase_type_to_timing(phase_type: PhaseType) -> tuple[TimingMode, ArrivalPattern]:
    """Map PhaseType to (TimingMode, ArrivalPattern).

    Delegates to the shared resolution function in config.resolved.
    """
    from aiperf.config.resolved import get_phase_timing

    return get_phase_timing(phase_type)


def _build_credit_phase_config(
    phase: BasePhaseConfig,
    *,
    phase_name: str,
    exclude_from_results: bool,
) -> CreditPhaseConfig:
    """Build a CreditPhaseConfig from a phase config.

    Maps the AIPerfConfig phase structure to the internal
    CreditPhaseConfig used by the timing system. Uses getattr for
    fields that only exist on specific phase types.

    For excluded phases (exclude_from_results=True), grace_period defaults to infinity
    to ensure all in-flight requests complete before the next phase begins.
    """
    timing_mode, arrival_pattern = _phase_type_to_timing(phase.type)

    grace_period = phase.grace_period
    if exclude_from_results and grace_period is None:
        grace_period = float("inf")

    rate_ramp = getattr(phase, "rate_ramp", None)

    return CreditPhaseConfig(
        phase=phase_name,
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
    )  # fmt: skip
