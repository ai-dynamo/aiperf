# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High concurrency, narrow ramp: 50K requests with <1% ramp."""

from __future__ import annotations

import numpy as np

from tests.unit.analysis.profiles import SyntheticBenchmark

_NS = 1_000_000_000


def generate(rng: np.random.Generator, n: int = 50_000) -> SyntheticBenchmark:
    """High volume with very short ramp — nearly all requests in steady state."""
    total_dur = 100 * _NS
    ramp_end = 0.5 * _NS
    drain_start = 99 * _NS

    n_ramp = int(n * 0.005)
    n_drain = int(n * 0.005)
    n_steady = n - n_ramp - n_drain

    ramp_starts = np.linspace(0, ramp_end, n_ramp, endpoint=False)
    ramp_latency = rng.normal(20, 2, n_ramp) * 1e6
    ramp_ends = ramp_starts + ramp_latency

    steady_starts = rng.uniform(ramp_end, drain_start - 30e6, n_steady)
    steady_latency = rng.normal(20, 2, n_steady) * 1e6
    steady_ends = steady_starts + steady_latency

    drain_starts = rng.uniform(drain_start - 30e6, drain_start, n_drain)
    drain_latency = rng.normal(20, 2, n_drain) * 1e6
    drain_ends = np.linspace(drain_start, total_dur, n_drain)

    start_ns = np.concatenate([ramp_starts, steady_starts, drain_starts])
    end_ns = np.concatenate([ramp_ends, steady_ends, drain_ends])
    latency = np.concatenate([ramp_latency, steady_latency, drain_latency])
    ttft = latency * rng.uniform(0.05, 0.15, n)

    generation_start_ns = start_ns + ttft
    output_tokens = rng.integers(50, 200, n).astype(np.float64)

    return SyntheticBenchmark(
        start_ns=start_ns,
        end_ns=end_ns,
        latency=latency,
        ttft=ttft,
        true_ramp_up_end_ns=float(ramp_end),
        true_ramp_down_start_ns=float(drain_start),
        true_steady_state_mean_latency=20e6,
        profile_name="high_concurrency",
        generation_start_ns=generation_start_ns,
        output_tokens=output_tokens,
    )
