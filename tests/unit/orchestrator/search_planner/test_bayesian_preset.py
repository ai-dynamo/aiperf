# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the curated `BayesianSearchPlanner` preset.

The preset is a thin subclass of `OptunaSearchPlanner` that locks
`optuna_sampler=botorch` and selects `optuna_acquisition` based on
n_objectives (qlognei single-obj, qlognehvi multi-obj). Multi-objective
support is part of the contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

optuna = pytest.importorskip("optuna")

from aiperf.config.config import BenchmarkConfig  # noqa: E402
from aiperf.config.sweep import (  # noqa: E402
    AdaptiveSearchSweep,
    Objective,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension  # noqa: E402
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection  # noqa: E402
from aiperf.orchestrator.search_planner.bayesian import (  # noqa: E402
    BayesianSearchPlanner,
)
from aiperf.orchestrator.search_planner.optuna_planner import (  # noqa: E402
    OptunaSearchPlanner,
)


def _base_config() -> BenchmarkConfig:
    """Minimal validated BenchmarkConfig the preset's super().__init__ will deep-copy."""
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "profiling", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 10,
                }
            ],
        }
    )


def _adaptive_cfg(*, n_objectives: int, **overrides) -> AdaptiveSearchSweep:
    objectives = [
        Objective(
            metric="output_token_throughput",
            stat="avg",
            direction=OptimizationDirection.MAXIMIZE,
        )
    ]
    if n_objectives == 2:
        objectives.append(
            Objective(
                metric="time_to_first_token",
                stat="p95",
                direction=OptimizationDirection.MINIMIZE,
            )
        )
    kwargs = {
        "search_space": [
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=16, kind="int"
            )
        ],
        "objectives": objectives,
        "max_iterations": 4,
        "n_initial_points": 2,
    }
    kwargs.update(overrides)
    return AdaptiveSearchSweep(**kwargs)


def test_preset_is_subclass_of_optuna_planner():
    assert issubclass(BayesianSearchPlanner, OptunaSearchPlanner)


def test_preset_single_objective_locks_botorch_qlognei():
    pytest.importorskip("botorch")
    cfg = _adaptive_cfg(n_objectives=1)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    # The preset stamps the curated cfg fields before delegating to super().
    assert planner._cfg.optuna_sampler == "botorch"
    assert planner._cfg.optuna_acquisition == "qlognei"


def test_preset_multi_objective_locks_botorch_qlognehvi():
    pytest.importorskip("botorch")
    cfg = _adaptive_cfg(n_objectives=2)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    assert planner._cfg.optuna_sampler == "botorch"
    assert planner._cfg.optuna_acquisition == "qlognehvi"


def test_preset_overrides_user_optuna_flags():
    """User passes --optuna-sampler=tpe but selects --search-planner=bayesian.
    The curated preset overrides; user wanting flexibility must pick =optuna.
    """
    pytest.importorskip("botorch")
    cfg = _adaptive_cfg(
        n_objectives=1,
        optuna_sampler="tpe",
        optuna_acquisition="qnei",
    )
    planner = BayesianSearchPlanner(_base_config(), cfg)
    assert planner._cfg.optuna_sampler == "botorch"
    assert planner._cfg.optuna_acquisition == "qlognei"


def test_preset_falls_back_to_tpe_when_implicit_botorch_is_unavailable():
    real_build_sampler = __import__(
        "aiperf.orchestrator.search_planner.optuna_planner",
        fromlist=["build_sampler"],
    ).build_sampler

    def fake_build_sampler(cfg):
        if cfg.optuna_sampler == "botorch":
            raise ImportError("simulated missing botorch")
        return real_build_sampler(cfg)

    cfg = _adaptive_cfg(n_objectives=1)
    with (
        patch(
            "aiperf.orchestrator.search_planner.optuna_planner.build_sampler",
            side_effect=fake_build_sampler,
        ),
        pytest.warns(RuntimeWarning, match="falling back to optuna_sampler='tpe'"),
    ):
        planner = BayesianSearchPlanner(_base_config(), cfg)

    assert planner._cfg.optuna_sampler == "tpe"
    assert planner._cfg.optuna_acquisition is None


def test_preset_accepts_multi_objective():
    """Pre-collapse, BayesianSearchPlanner raised for len(objectives) > 1.
    Post-collapse, the preset must accept multi-obj (it now backs the
    qlognehvi path through Optuna).
    """
    cfg = _adaptive_cfg(n_objectives=2)
    # Should not raise.
    BayesianSearchPlanner(_base_config(), cfg)


def test_preset_ask_returns_proposal():
    """End-to-end smoke: instantiate, ask once, get back (cfg, variation)."""
    cfg = _adaptive_cfg(n_objectives=1)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    proposal = planner.ask()
    assert proposal is not None
    benchmark_cfg, variation = proposal
    assert variation.index == 0
    assert "phases.profiling.concurrency" in variation.values
