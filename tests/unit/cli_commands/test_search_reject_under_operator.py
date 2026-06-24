# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator-managed-pod handling of adaptive outer-loop plans.

v2: adaptive search is allowed under the operator — the controller pod owns the
BO loop in cluster mode. Only deterministic grid sweeps still need the
AIPerfSweep CRD.
"""

from __future__ import annotations

from aiperf.cli_runner import _reject_in_process_sweep_under_operator
from aiperf.common.enums import OptimizationDirection
from aiperf.config import BenchmarkPlan
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import (
    AdaptiveObjective,
    AdaptiveSearchSweep,
    SweepVariation,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension


def _bo_plan() -> BenchmarkPlan:
    sweep = AdaptiveSearchSweep(
        planner="bayesian",
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objectives=[
            AdaptiveObjective(
                metric="m", stat="avg", direction=OptimizationDirection.MAXIMIZE
            )
        ],
        max_iterations=10,
    )
    return BenchmarkPlan(
        configs=[
            BenchmarkConfig.model_validate(
                {
                    "models": ["m"],
                    "endpoint": {"urls": ["http://x"], "type": "chat"},
                    "datasets": [{"name": "default", "type": "synthetic"}],
                    "phases": [
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "requests": 1,
                            "concurrency": 1,
                        }
                    ],
                }
            )
        ],
        variations=[SweepVariation(index=0, label="base", values={})],
        sweep=sweep,
    )


def test_bo_allowed_under_operator(monkeypatch):
    """v2: adaptive search is allowed under AIPERF_OPERATOR_MANAGED=1.

    The controller pod owns the BO loop in cluster mode; the in-process
    rejection that existed in v1 has been lifted.
    """
    monkeypatch.setenv("AIPERF_OPERATOR_MANAGED", "1")
    # Should NOT raise — controller pod owns the BO loop in cluster mode.
    _reject_in_process_sweep_under_operator(_bo_plan())


def test_bo_allowed_outside_operator(monkeypatch):
    monkeypatch.delenv("AIPERF_OPERATOR_MANAGED", raising=False)
    # Should not raise: BO is fine in-process when not under the operator.
    _reject_in_process_sweep_under_operator(_bo_plan())
