# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class TimingMode(CaseInsensitiveStrEnum):
    """The different ways the TimingManager should generate requests."""

    FIXED_SCHEDULE = "fixed_schedule"
    """A mode where the TimingManager will send requests according to a fixed schedule."""

    REQUEST_RATE = "request_rate"
    """A mode where the TimingManager will send requests using a request rate generator based on various modes.
    Optionally, a max concurrency limit can be specified as well.
    """


class RequestRateMode(CaseInsensitiveStrEnum):
    """The different ways the RequestRateStrategy should generate requests."""

    CONSTANT = "constant"
    """Generate requests at a constant rate."""

    POISSON = "poisson"
    """Generate requests using a poisson process."""

    CONCURRENCY_BURST = "concurrency_burst"
    """Generate requests as soon as possible, up to a max concurrency limit. Only allowed when a request rate is not specified."""


class CreditPhase(CaseInsensitiveStrEnum):
    """The type of credit phase. This is used to identify which phase of the
    benchmark the credit is being used in, for tracking and reporting purposes."""

    WARMUP = "warmup"
    """The credit phase while the warmup is active. This is used to warm up the model and
    ensure that the model is ready to be profiled."""

    PROFILING = "profiling"
    """The credit phase while profiling is active. This is the primary phase of the
    benchmark, and what is used to calculate the final results."""
