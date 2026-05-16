# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slow server warm-up: concurrency stabilizes at 10%, latency until 25%."""

from __future__ import annotations

import numpy as np

from tests.unit.analysis.profiles import SyntheticBenchmark

_NS = 1_000_000_000


def generate(rng: np.random.Generator, n: int = 10_000) -> SyntheticBenchmark:
    """Concurrency stable at 10%, but latency only stabilizes at 25%."""
    total_dur = 100 * _NS
    latency_stable = 25 * _NS  # latency stable at 25%
    drain_start = 80 * _NS

    n_warmup = int(n * 0.25)
    n_steady = int(n * 0.55)
    n_drain = n - n_warmup - n_steady

    # Warm-up: latency decreases exponentially from 200ms to 50ms
    warmup_starts = rng.uniform(0, latency_stable, n_warmup)
    warmup_starts.sort()
    t_frac = warmup_starts / latency_stable
    warmup_latency = (200 - 150 * t_frac + rng.normal(0, 10, n_warmup)) * 1e6
    warmup_latency = np.maximum(warmup_latency, 10e6)
    warmup_ends = warmup_starts + warmup_latency

    # Steady
    steady_starts = rng.uniform(latency_stable, drain_start - 50e6, n_steady)
    steady_latency = rng.normal(50, 3, n_steady) * 1e6
    steady_ends = steady_starts + steady_latency

    # Drain
    drain_starts = rng.uniform(drain_start - 50e6, drain_start, n_drain)
    drain_latency = rng.normal(50, 5, n_drain) * 1e6
    drain_ends = np.linspace(drain_start, total_dur, n_drain)

    start_ns = np.concatenate([warmup_starts, steady_starts, drain_starts])
    end_ns = np.concatenate([warmup_ends, steady_ends, drain_ends])
    latency = np.concatenate([warmup_latency, steady_latency, drain_latency])
    ttft = latency * rng.uniform(0.05, 0.15, n)

    generation_start_ns = start_ns + ttft
    output_tokens = rng.integers(50, 200, n).astype(np.float64)

    return SyntheticBenchmark(
        start_ns=start_ns,
        end_ns=end_ns,
        latency=latency,
        ttft=ttft,
        true_ramp_up_end_ns=float(latency_stable),
        true_ramp_down_start_ns=float(drain_start),
        true_steady_state_mean_latency=50e6,
        profile_name="slow_warmup",
        generation_start_ns=generation_start_ns,
        output_tokens=output_tokens,
    )
