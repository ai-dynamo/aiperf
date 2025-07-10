# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base import CaseInsensitiveStrEnum


class TimingMode(CaseInsensitiveStrEnum):
    """The different ways the TimingManager should generate requests."""

    FIXED_SCHEDULE = "fixed_schedule"
    """A mode where the TimingManager will send requests according to a fixed schedule."""

    CONCURRENCY = "concurrency"
    """A mode where the TimingManager will maintain a continuous stream of concurrent requests."""

    REQUEST_RATE = "request_rate"
    """A mode where the TimingManager will send requests at either a constant request rate or based on a poisson distribution."""


class RequestRateMode(CaseInsensitiveStrEnum):
    """The different ways the RequestRateStrategy should generate requests."""

    CONSTANT = "constant"
    """Generate requests at a constant rate."""

    POISSON = "poisson"
    """Generate requests using a poisson distribution."""


class CreditPhase(CaseInsensitiveStrEnum):
    """The type of credit phase. This is used to identify which phase of the
    benchmark the credit is being used in, for tracking and reporting purposes."""

    WARMUP = "warmup"
    """The credit phase is the warmup phase. This is used to warm up the model
    before the benchmark starts."""

    RAMP_UP = "ramp_up"
    """The credit phase is the ramp up phase. This is used to ramp up the request
    rate before the benchmark starts."""

    STABILIZING = "stabilizing"
    """Used with stabilization based measurement modes to indicate that the
    benchmark is still stabilizing."""

    STEADY_STATE = "steady_state"
    """The credit phase is the steady state phase. This is the primary phase of the
    benchmark, and what is used to calculate the final results."""
