# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import ConfigDict, Field

from aiperf.common.config import InputDefaults, UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.plugin.enums import (
    ArrivalPattern,
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
    def from_user_config(cls, user_config: UserConfig) -> TimingConfig:
        """Build ordered list of phase configs based on user config: [warmup?, profiling].

        Warmup (if enabled) executes first to prepare system,
        then profiling for actual measurement.
        """
        loadgen = user_config.loadgen
        configs: list[CreditPhaseConfig] = []

        warmup = _build_warmup_config(user_config)
        if warmup:
            configs.append(warmup)

        configs.append(_build_profiling_config(user_config))

        return cls(
            phase_configs=configs,
            request_cancellation=RequestCancellationConfig(
                rate=loadgen.request_cancellation_rate,
                delay=loadgen.request_cancellation_delay,
            ),
            urls=user_config.endpoint.urls,
            url_selection_strategy=user_config.endpoint.url_selection_strategy,
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


def _build_warmup_config(user_config: UserConfig) -> CreditPhaseConfig | None:
    """Build warmup phase config if any warmup stop condition is set.

    Returns None if warmup disabled (no stop conditions).
    Warmup triggers JIT compilation, memory allocation, and connection pool
    initialization so profiling measurements aren't polluted by cold-start effects.

    Note:
        When warmup_grace_period is not specified, defaults to infinity (wait forever
        for in-flight requests). This differs from the CreditPhaseConfig field default
        of None (disabled) because warmup should always complete all requests.
    """
    loadgen = user_config.loadgen
    if not (
        loadgen.warmup_request_count
        or loadgen.warmup_duration
        or loadgen.warmup_num_sessions
    ):
        return None

    request_rate = loadgen.warmup_request_rate or loadgen.request_rate
    arrival_pattern = loadgen.warmup_arrival_pattern or loadgen.arrival_pattern
    concurrency = loadgen.warmup_concurrency or loadgen.concurrency
    prefill_concurrency = (
        loadgen.warmup_prefill_concurrency or loadgen.prefill_concurrency
    )
    if request_rate is None or arrival_pattern is None:
        arrival_pattern = ArrivalPattern.CONCURRENCY_BURST
        if concurrency is None and prefill_concurrency is None:
            concurrency = 1
            # TODO: We should add a warning here

    return CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        # Warmup phase is always request rate timing mode
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=loadgen.warmup_request_count,
        expected_duration_sec=loadgen.warmup_duration,
        expected_num_sessions=loadgen.warmup_num_sessions,
        concurrency=concurrency,
        prefill_concurrency=prefill_concurrency,
        request_rate=request_rate,
        arrival_pattern=arrival_pattern,
        arrival_smoothness=loadgen.arrival_smoothness,
        seamless=False,
        grace_period_sec=loadgen.warmup_grace_period if loadgen.warmup_grace_period is not None else float('inf'),
        concurrency_ramp_duration_sec=loadgen.warmup_concurrency_ramp_duration or loadgen.concurrency_ramp_duration,
        prefill_concurrency_ramp_duration_sec=loadgen.warmup_prefill_concurrency_ramp_duration or loadgen.prefill_concurrency_ramp_duration,
        request_rate_ramp_duration_sec=loadgen.warmup_request_rate_ramp_duration or loadgen.request_rate_ramp_duration,
    )  # fmt: skip


def _build_profiling_config(user_config: UserConfig) -> CreditPhaseConfig:
    """Build profiling phase config (always created).

    Main benchmark phase where all performance metrics are collected.
    Grace period allows in-flight requests to complete after the stop condition
    is met, ensuring metrics include requests that were sent before the deadline.
    """

    loadgen = user_config.loadgen
    input = user_config.input

    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=user_config.timing_mode,
        expected_duration_sec=loadgen.benchmark_duration,
        total_expected_requests=loadgen.request_count,
        expected_num_sessions=input.conversation.num,
        concurrency=loadgen.concurrency,
        prefill_concurrency=loadgen.prefill_concurrency,
        request_rate=loadgen.request_rate or loadgen.user_centric_rate,
        arrival_pattern=loadgen.arrival_pattern,
        arrival_smoothness=loadgen.arrival_smoothness,
        grace_period_sec=loadgen.benchmark_grace_period,
        num_users=loadgen.num_users,
        concurrency_ramp_duration_sec=loadgen.concurrency_ramp_duration,
        prefill_concurrency_ramp_duration_sec=loadgen.prefill_concurrency_ramp_duration,
        request_rate_ramp_duration_sec=loadgen.request_rate_ramp_duration,
        # Fixed schedule config
        auto_offset_timestamps=input.fixed_schedule_auto_offset,
        fixed_schedule_start_offset=input.fixed_schedule_start_offset,
        fixed_schedule_end_offset=input.fixed_schedule_end_offset,
    )  # fmt: skip
