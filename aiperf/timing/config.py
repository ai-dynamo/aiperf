# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import (
    InputDefaults,
    LoadGeneratorDefaults,
    UserConfig,
)
from aiperf.common.enums import RequestRateMode, TimingMode
from aiperf.common.models import AIPerfBaseModel


class TimingManagerConfig(AIPerfBaseModel):
    """Configuration for the timing manager."""

    timing_mode: TimingMode = LoadGeneratorDefaults.TIMING_MODE
    concurrency: int | None = LoadGeneratorDefaults.CONCURRENCY
    request_rate: float | None = LoadGeneratorDefaults.REQUEST_RATE
    request_rate_mode: RequestRateMode = LoadGeneratorDefaults.REQUEST_RATE_MODE
    request_count: int = LoadGeneratorDefaults.REQUEST_COUNT
    warmup_request_count: int = LoadGeneratorDefaults.WARMUP_REQUEST_COUNT
    benchmark_duration: float | None = LoadGeneratorDefaults.BENCHMARK_DURATION
    benchmark_grace_period: float = LoadGeneratorDefaults.BENCHMARK_GRACE_PERIOD
    random_seed: int | None = None
    auto_offset_timestamps: bool = InputDefaults.FIXED_SCHEDULE_AUTO_OFFSET
    fixed_schedule_start_offset: int | None = InputDefaults.FIXED_SCHEDULE_START_OFFSET
    fixed_schedule_end_offset: int | None = InputDefaults.FIXED_SCHEDULE_END_OFFSET

    @classmethod
    def from_user_config(cls, user_config: UserConfig) -> "TimingManagerConfig":
        """Create a TimingManagerConfig from a UserConfig."""

        return cls(
            timing_mode=user_config.timing_mode,
            concurrency=user_config.loadgen.concurrency,
            request_rate=user_config.loadgen.request_rate,
            request_rate_mode=user_config.loadgen.request_rate_mode,
            request_count=user_config.get_effective_request_count(),  # Use effective request count for custom datasets
            warmup_request_count=user_config.loadgen.warmup_request_count,
            benchmark_duration=user_config.loadgen.benchmark_duration,
            benchmark_grace_period=user_config.loadgen.benchmark_grace_period,
            random_seed=user_config.input.random_seed,
            auto_offset_timestamps=user_config.input.fixed_schedule_auto_offset,
            fixed_schedule_start_offset=user_config.input.fixed_schedule_start_offset,
            fixed_schedule_end_offset=user_config.input.fixed_schedule_end_offset,
        )
