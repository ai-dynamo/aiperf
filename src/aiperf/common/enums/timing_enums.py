# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    USER_CENTRIC_RATE = "user_centric_rate"
    """A mode where each session acts as a separate user with gap = num_users / request_rate between turns.
    Users block on their previous turn (no interleaving within a user).
    Matches LMBenchmark behavior for KV cache benchmarking.
    """


class ArrivalPattern(CaseInsensitiveStrEnum):
    """The different ways the IntervalGenerator should generate intervals."""

    CONSTANT = "constant"
    """Generate intervals at a constant rate."""

    POISSON = "poisson"
    """Generate intervals using a poisson process."""

    GAMMA = "gamma"
    """Generate intervals using a gamma distribution with tunable smoothness.
    Use --arrival-smoothness to control the shape parameter:
    - smoothness = 1.0: Equivalent to Poisson (exponential inter-arrivals)
    - smoothness < 1.0: More bursty/clustered arrivals
    - smoothness > 1.0: More regular/smooth arrivals
    """

    CONCURRENCY_BURST = "concurrency_burst"
    """Generate intervals as soon as possible, up to a max concurrency limit. Only allowed when a request rate is not specified."""


class CreditPhase(CaseInsensitiveStrEnum):
    """The type of credit phase. This is used to identify which phase of the
    benchmark the credit is being used in, for tracking and reporting purposes."""

    WARMUP = "warmup"
    """The credit phase while the warmup is active. This is used to warm up the model and
    ensure that the model is ready to be profiled."""

    PROFILING = "profiling"
    """The credit phase while profiling is active. This is the primary phase of the
    benchmark, and what is used to calculate the final results."""
