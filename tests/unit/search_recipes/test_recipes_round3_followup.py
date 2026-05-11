# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-3 follow-up regressions for the search-recipe post-processors.

This file tracks the fixes applied after the initial round-3 cleanup landed.
The original round-3 commit ``ab5f07a16`` taught :class:`DegradationKneeDetect`
to reject negative and non-finite baselines, but the strictly-zero baseline was
still accepted. A zero baseline produces ``cutoff = 0 * (1 + threshold) = 0`` so
any positive value is flagged as an "infinite" degradation, which is a
meaningless threshold; reject it up front instead.
"""

from __future__ import annotations

import pytest

from aiperf.search_recipes.post_process import DegradationKneeDetect


def test_degradation_knee_rejects_zero_baseline() -> None:
    """A zero baseline yields cutoff=0 — every positive y trips it. Reject it."""
    agg = {
        "per_combination_metrics": [
            {
                "parameters": {"c": 1},
                "metrics": {"request_latency_p99": {"mean": 0.0}},
            },
            {
                "parameters": {"c": 100},
                "metrics": {"request_latency_p99": {"mean": 1000.0}},
            },
        ]
    }
    with pytest.raises(ValueError, match="must be positive"):
        DegradationKneeDetect().process(
            agg,
            {
                "threshold_pct": 0.2,
                "metric_tag": "request_latency",
                "stat": "p99",
                "swept_param": "c",
            },
        )
