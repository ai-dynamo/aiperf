# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-objective search_history.json shape tests."""

from __future__ import annotations

import json
from pathlib import Path

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    Objective,
    SearchSpaceDimension,
)
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.search_planner.base import SearchIteration


def _single_obj_cfg() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(path="concurrency", lo=1, hi=100, kind="int")
        ],
        objectives=[Objective(metric="x", direction=OptimizationDirection.MAXIMIZE)],
        max_iterations=10,
    )


def _two_obj_cfg() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(path="concurrency", lo=1, hi=100, kind="int")
        ],
        objectives=[
            Objective(metric="throughput", direction=OptimizationDirection.MAXIMIZE),
            Objective(metric="latency", direction=OptimizationDirection.MINIMIZE),
        ],
        max_iterations=10,
        optuna_sampler="botorch",
        optuna_acquisition="qlognehvi",
    )


def test_single_objective_emits_length_one_best_trials(tmp_path: Path):
    cfg = _single_obj_cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 5},
            objective_value=10.0,
            objective_values=[10.0],
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"concurrency": 10},
            objective_value=20.0,
            objective_values=[20.0],
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())
    assert "best" not in payload
    assert isinstance(payload["best_trials"], list)
    assert len(payload["best_trials"]) == 1
    assert payload["best_trials"][0]["iteration_idx"] == 1
    assert payload["best_trials"][0]["objective_values"] == [20.0]


def test_multi_objective_emits_pareto_front(tmp_path: Path):
    cfg = _two_obj_cfg()
    # 3 points: (10, 5), (20, 8), (15, 3). Maximize first, minimize second.
    # Pareto front: (20, 8) dominates nothing on lat>=8; (15, 3) dominates nothing
    # on tput<=15. (10, 5) dominated by (15, 3)? tput 15>10, lat 3<5 -> yes.
    # Front: {(20, 8), (15, 3)}
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 5},
            objective_value=10.0,
            objective_values=[10.0, 5.0],
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"concurrency": 50},
            objective_value=20.0,
            objective_values=[20.0, 8.0],
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=2,
            variation_values={"concurrency": 30},
            objective_value=15.0,
            objective_values=[15.0, 3.0],
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())
    front = payload["best_trials"]
    iters_on_front = sorted(p["iteration_idx"] for p in front)
    assert iters_on_front == [1, 2]
    # All on-front points have pareto_rank == 0
    assert all(p["pareto_rank"] == 0 for p in front)


def test_config_block_emits_objectives_list(tmp_path: Path):
    cfg = _two_obj_cfg()
    write_search_history(tmp_path, [], cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())
    objs = payload["config"]["objectives"]
    assert len(objs) == 2
    assert objs[0]["metric"] == "throughput"
    assert objs[0]["direction"] == "MAXIMIZE"
    assert objs[1]["metric"] == "latency"
    assert objs[1]["direction"] == "MINIMIZE"


def test_iterations_emit_objective_values_vector(tmp_path: Path):
    cfg = _two_obj_cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 5},
            objective_value=10.0,
            objective_values=[10.0, 5.0],
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())
    assert payload["iterations"][0]["objective_values"] == [10.0, 5.0]
