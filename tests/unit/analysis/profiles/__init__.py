# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic benchmark profiles for steady-state detection validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticBenchmark:
    """Ground-truth benchmark profile for detection accuracy testing."""

    start_ns: NDArray[np.float64]
    end_ns: NDArray[np.float64]
    latency: NDArray[np.float64]
    ttft: NDArray[np.float64]
    true_ramp_up_end_ns: float
    true_ramp_down_start_ns: float
    true_steady_state_mean_latency: float
    profile_name: str
    generation_start_ns: NDArray[np.float64] | None = None
    output_tokens: NDArray[np.float64] | None = None

    @property
    def total_duration(self) -> float:
        return float(np.nanmax(self.end_ns) - np.nanmin(self.start_ns))

    @property
    def true_window_duration(self) -> float:
        return self.true_ramp_down_start_ns - self.true_ramp_up_end_ns

    def true_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of requests in the true steady-state window."""
        return (self.start_ns >= self.true_ramp_up_end_ns) & (
            self.end_ns <= self.true_ramp_down_start_ns
        )
