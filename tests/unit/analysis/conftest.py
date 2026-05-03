# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for steady-state detection accuracy tests."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.analysis.profiles import (
    SyntheticBenchmark,
    asymmetric_ramp,
    burst_arrivals,
    clean_ramp,
    flat_profile,
    high_concurrency,
    noisy_ramp,
    short_benchmark,
    slow_warmup,
    step_function,
)

# Profiles where a well-defined ramp exists and boundaries are meaningful
RAMP_PROFILES = [
    clean_ramp,
    slow_warmup,
    noisy_ramp,
    step_function,
    short_benchmark,
    asymmetric_ramp,
    burst_arrivals,
    high_concurrency,
]

ALL_PROFILES = [
    clean_ramp,
    slow_warmup,
    noisy_ramp,
    step_function,
    short_benchmark,
    flat_profile,
    asymmetric_ramp,
    burst_arrivals,
    high_concurrency,
]


@pytest.fixture(params=RAMP_PROFILES, ids=lambda m: m.__name__.split(".")[-1])
def ramp_benchmark(request) -> SyntheticBenchmark:
    """Synthetic benchmark with a ramp (excludes flat_profile and no_steady_state)."""
    return request.param.generate(rng=np.random.default_rng(42))


@pytest.fixture(params=ALL_PROFILES, ids=lambda m: m.__name__.split(".")[-1])
def synthetic_benchmark(request) -> SyntheticBenchmark:
    """All synthetic benchmarks except no_steady_state (which has no plateau)."""
    return request.param.generate(rng=np.random.default_rng(42))
