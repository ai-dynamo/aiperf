# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator-managed-pod rejection of adaptive outer-loop plans."""

from __future__ import annotations

import pytest

from aiperf.cli_runner import _reject_in_process_sweep_under_operator
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.config.benchmark import BenchmarkPlan
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection


def _bo_plan() -> BenchmarkPlan:
    cfg = AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objective_metric="m",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
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
        adaptive_search=cfg,
    )


def test_reject_bo_under_operator(monkeypatch):
    monkeypatch.setenv("AIPERF_OPERATOR_MANAGED", "1")
    with pytest.raises(SystemExit, match="adaptive outer loop"):
        _reject_in_process_sweep_under_operator(_bo_plan())


def test_bo_allowed_outside_operator(monkeypatch):
    monkeypatch.delenv("AIPERF_OPERATOR_MANAGED", raising=False)
    # Should not raise: BO is fine in-process when not under the operator.
    _reject_in_process_sweep_under_operator(_bo_plan())
