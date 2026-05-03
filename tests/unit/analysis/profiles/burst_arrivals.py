# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Burst arrivals: periodic bursts with inter-burst gaps."""

from __future__ import annotations

import numpy as np

from tests.unit.analysis.profiles import SyntheticBenchmark

_NS = 1_000_000_000


def generate(rng: np.random.Generator, n: int = 10_000) -> SyntheticBenchmark:
    """Periodic bursts of requests, creating a bursty concurrency pattern."""
    total_dur = 100 * _NS
    ramp_end = 10 * _NS
    drain_start = 85 * _NS

    n_ramp = int(n * 0.10)
    n_steady = int(n * 0.75)
    n_drain = n - n_ramp - n_steady

    # Ramp
    ramp_starts = np.linspace(0, ramp_end, n_ramp, endpoint=False)
    ramp_latency = rng.normal(50, 8, n_ramp) * 1e6
    ramp_latency = np.maximum(ramp_latency, 10e6)
    ramp_ends = ramp_starts + ramp_latency

    # Steady with bursts: 15 burst windows, requests cluster at burst centers
    n_bursts = 15
    burst_centers = np.linspace(ramp_end + 2 * _NS, drain_start - 2 * _NS, n_bursts)
    burst_width = 1.5 * _NS
    steady_starts = np.zeros(n_steady)
    per_burst = n_steady // n_bursts
    for i in range(n_bursts):
        s = i * per_burst
        e = (i + 1) * per_burst if i < n_bursts - 1 else n_steady
        steady_starts[s:e] = rng.normal(burst_centers[i], burst_width / 4, e - s)

    steady_starts = np.clip(steady_starts, ramp_end, drain_start - 60e6)
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
        profile_name="burst_arrivals",
        generation_start_ns=generation_start_ns,
        output_tokens=output_tokens,
    )
