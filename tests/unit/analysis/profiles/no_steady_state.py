# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No steady state: continuously drifting latency (no plateau)."""

from __future__ import annotations

import numpy as np

from tests.unit.analysis.profiles import SyntheticBenchmark

_NS = 1_000_000_000


def generate(rng: np.random.Generator, n: int = 5_000) -> SyntheticBenchmark:
    """Latency continuously increases — no true steady state.

    The 'true' boundaries are set to mark the middle 50% for reference,
    but stationarity_warning should fire on this profile.
    """
    total_dur = 100 * _NS

    starts = rng.uniform(0, total_dur - 200e6, n)
    starts.sort()

    # Latency increases linearly with time
    t_frac = starts / total_dur
    latency = (30 + 70 * t_frac + rng.normal(0, 5, n)) * 1e6
    latency = np.maximum(latency, 5e6)
    ends = starts + latency
    ttft = latency * rng.uniform(0.05, 0.15, n)

    generation_start_ns = starts + ttft
    output_tokens = rng.integers(50, 200, n).astype(np.float64)

    return SyntheticBenchmark(
        start_ns=starts,
        end_ns=ends,
        latency=latency,
        ttft=ttft,
        true_ramp_up_end_ns=25 * _NS,
        true_ramp_down_start_ns=75 * _NS,
        true_steady_state_mean_latency=65e6,
        profile_name="no_steady_state",
        generation_start_ns=generation_start_ns,
        output_tokens=output_tokens,
    )
