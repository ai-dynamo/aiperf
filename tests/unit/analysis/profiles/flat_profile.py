# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flat profile: constant concurrency throughout (no ramp)."""

from __future__ import annotations

import numpy as np

from tests.unit.analysis.profiles import SyntheticBenchmark

_NS = 1_000_000_000


def generate(rng: np.random.Generator, n: int = 10_000) -> SyntheticBenchmark:
    """All requests overlap the full duration — no ramp at all."""
    total_dur = 100 * _NS

    starts = rng.uniform(0, total_dur - 100e6, n)
    latency = rng.normal(50, 3, n) * 1e6
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
        true_ramp_up_end_ns=0.0,
        true_ramp_down_start_ns=float(total_dur),
        true_steady_state_mean_latency=50e6,
        profile_name="flat_profile",
        generation_start_ns=generation_start_ns,
        output_tokens=output_tokens,
    )
