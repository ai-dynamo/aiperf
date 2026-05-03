# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-function ramp: concurrency increases in discrete steps."""

from __future__ import annotations

import numpy as np

from tests.unit.analysis.profiles import SyntheticBenchmark

_NS = 1_000_000_000


def generate(rng: np.random.Generator, n: int = 10_000) -> SyntheticBenchmark:
    """Steps of 10 concurrent, stabilizing at each level before the next."""
    total_dur = 100 * _NS
    ramp_end = 20 * _NS
    drain_start = 80 * _NS

    n_ramp = int(n * 0.20)
    n_steady = int(n * 0.60)
    n_drain = n - n_ramp - n_steady

    # Step ramp: 4 steps of 5s each
    ramp_starts = np.zeros(n_ramp)
    per_step = n_ramp // 4
    for step in range(4):
        t0 = step * 5 * _NS
        t1 = (step + 1) * 5 * _NS
        s = step * per_step
        e = (step + 1) * per_step if step < 3 else n_ramp
        ramp_starts[s:e] = rng.uniform(t0, t1, e - s)

    ramp_latency = rng.normal(60, 10, n_ramp) * 1e6
    ramp_latency = np.maximum(ramp_latency, 10e6)
    ramp_ends = ramp_starts + ramp_latency

    # Steady
    steady_starts = rng.uniform(ramp_end, drain_start - 50e6, n_steady)
    steady_latency = rng.normal(50, 3, n_steady) * 1e6
    steady_ends = steady_starts + steady_latency

    # Drain
    drain_starts = rng.uniform(drain_start - 50e6, drain_start, n_drain)
    drain_latency = rng.normal(50, 5, n_drain) * 1e6
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
        true_steady_state_mean_latency=50e6,
        profile_name="step_function",
        generation_start_ns=generation_start_ns,
        output_tokens=output_tokens,
    )
