# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.config_defaults import LoadGeneratorDefaults
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
    concurrency_ramp_up_time: float | None = (
        LoadGeneratorDefaults.CONCURRENCY_RAMP_UP_TIME
    )
    random_seed: int | None = None

    @classmethod
    def from_user_config(cls, user_config: UserConfig) -> "TimingManagerConfig":
        """Create a TimingManagerConfig from a UserConfig."""

        if user_config.input.file is not None:
            timing_mode = TimingMode.FIXED_SCHEDULE
        elif user_config.load.request_rate is not None:
            timing_mode = TimingMode.REQUEST_RATE
        else:
            # Default to concurrency mode if no request rate or schedule is provided
            timing_mode = TimingMode.CONCURRENCY

        return cls(
            timing_mode=timing_mode,
            concurrency=user_config.load.concurrency,
            request_rate=user_config.load.request_rate,
            request_rate_mode=user_config.load.request_rate_mode,
            request_count=user_config.load.request_count,
            warmup_request_count=user_config.load.warmup_request_count,
            # concurrency_ramp_up_time=user_config.load.concurrency_ramp_up_time,
            random_seed=user_config.input.random_seed,
        )
