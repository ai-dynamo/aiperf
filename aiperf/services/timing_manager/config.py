# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.config_defaults import LoadGeneratorDefaults, ServiceDefaults
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import RequestRateMode, TimingMode
from aiperf.common.pydantic_utils import AIPerfBaseModel


class TimingManagerConfig(AIPerfBaseModel):
    """Configuration for the timing manager."""

    timing_mode: TimingMode = LoadGeneratorDefaults.TIMING_MODE
    concurrency: int = LoadGeneratorDefaults.CONCURRENCY
    request_rate: float | None = LoadGeneratorDefaults.REQUEST_RATE
    request_rate_mode: RequestRateMode = LoadGeneratorDefaults.REQUEST_RATE_MODE
    request_count: int = LoadGeneratorDefaults.REQUEST_COUNT
    warmup_request_count: int = LoadGeneratorDefaults.WARMUP_REQUEST_COUNT
    random_seed: int | None = None
    progress_report_interval_sec: float = (
        ServiceDefaults.PROGRESS_REPORT_INTERVAL_SECONDS
    )

    @classmethod
    def from_user_config(cls, user_config: UserConfig) -> "TimingManagerConfig":
        """Create a TimingManagerConfig from a UserConfig."""

        if user_config.input.fixed_schedule:
            timing_mode = TimingMode.FIXED_SCHEDULE
        elif user_config.loadgen.request_rate is not None:
            timing_mode = TimingMode.REQUEST_RATE
        else:
            # Default to concurrency mode if no request rate or schedule is provided
            timing_mode = TimingMode.CONCURRENCY

        return cls(
            timing_mode=timing_mode,
            concurrency=user_config.loadgen.concurrency,
            request_rate=user_config.loadgen.request_rate,
            request_rate_mode=user_config.loadgen.request_rate_mode,
            request_count=user_config.loadgen.request_count,
            warmup_request_count=user_config.loadgen.warmup_request_count,
            random_seed=user_config.input.random_seed,
        )
