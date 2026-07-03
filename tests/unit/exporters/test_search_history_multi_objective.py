# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-objective Pareto-front shape tests for search_history.json.

Covers the ``best_trials`` restoration: single-objective stays a length-1
argmax/argmin list; multi-objective becomes the non-dominated (Pareto) front,
every member carrying ``pareto_rank == 0``. Direction handling follows the
planner convention verified in ``_optuna_helpers`` (MAXIMIZE = larger-is-better,
MINIMIZE = smaller-is-better); the Pareto math uses real ``SearchIteration``
data so dominance is exercised end-to-end through the exporter.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.search_planner.base import SearchIteration


def _single_obj_cfg(
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
) -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(path="concurrency", lo=1, hi=100, kind="int")
        ],
        objectives=[Objective(metric="throughput", direction=direction)],
        max_iterations=10,
    )


def _two_obj_cfg(
    d0: OptimizationDirection = OptimizationDirection.MAXIMIZE,
    d1: OptimizationDirection = OptimizationDirection.MINIMIZE,
) -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(path="concurrency", lo=1, hi=100, kind="int")
        ],
        objectives=[
            Objective(metric="throughput", direction=d0),
            Objective(metric="latency", direction=d1),
        ],
        max_iterations=10,
        optuna_sampler="botorch",
        optuna_acquisition="qlognehvi",
    )


def _iter(idx: int, values: list[float], *, feasible: bool = True) -> SearchIteration:
    """Build a SearchIteration with a real objective vector (scalar mirrors [0])."""
    return SearchIteration(
        iteration_idx=idx,
        variation_values={"concurrency": idx + 1},
        objective_value=values[0],
        objective_values=list(values),
        feasible=feasible,
    )


def _read(base_dir: Path) -> dict:
    return orjson.loads((base_dir / "search_history.json").read_bytes())


def test_compute_best_trials_single_objective_maximize_picks_argmax(tmp_path: Path):
    """Single objective, MAXIMIZE: length-1 list with the argmax and pareto_rank 0."""
    cfg = _single_obj_cfg(OptimizationDirection.MAXIMIZE)
    history = [_iter(0, [10.0]), _iter(1, [20.0]), _iter(2, [15.0])]
    write_search_history(tmp_path, history, cfg)
    trials = _read(tmp_path)["best_trials"]

    assert len(trials) == 1
    assert trials[0]["iteration_idx"] == 1
    assert trials[0]["objective_values"] == [20.0]
    assert trials[0]["pareto_rank"] == 0


def test_compute_best_trials_single_objective_minimize_picks_argmin(tmp_path: Path):
    """Single objective, MINIMIZE: length-1 list with the argmin (behavior unchanged)."""
    cfg = _single_obj_cfg(OptimizationDirection.MINIMIZE)
    history = [_iter(0, [10.0]), _iter(1, [8.0]), _iter(2, [15.0])]
    write_search_history(tmp_path, history, cfg)
    trials = _read(tmp_path)["best_trials"]

    assert len(trials) == 1
    assert trials[0]["iteration_idx"] == 1
    assert trials[0]["objective_values"] == [8.0]
    assert trials[0]["pareto_rank"] == 0


def test_compute_best_trials_multi_objective_returns_exact_pareto_front(
    tmp_path: Path,
):
    """Known 2-objective set (MAX, MIN) resolves to the exact non-dominated front.

    Points: (10, 5), (20, 8), (15, 3). Maximize obj0, minimize obj1.
    (15, 3) dominates (10, 5) (>= on obj0, <= on obj1, strictly better on both);
    (20, 8) and (15, 3) are mutually non-dominated. Front == {iter 1, iter 2}.
    """
    cfg = _two_obj_cfg()
    history = [
        _iter(0, [10.0, 5.0]),
        _iter(1, [20.0, 8.0]),
        _iter(2, [15.0, 3.0]),
    ]
    write_search_history(tmp_path, history, cfg)
    front = _read(tmp_path)["best_trials"]

    assert sorted(p["iteration_idx"] for p in front) == [1, 2]
    assert all(p["pareto_rank"] == 0 for p in front)
    assert all(len(p["objective_values"]) == 2 for p in front)


@pytest.mark.parametrize(
    ("d0", "d1", "points", "expected_front"),
    [
        param(
            OptimizationDirection.MAXIMIZE,
            OptimizationDirection.MINIMIZE,
            [[10.0, 5.0], [20.0, 8.0], [15.0, 3.0]],
            [1, 2],
            id="max_min",
        ),
        param(
            OptimizationDirection.MINIMIZE,
            OptimizationDirection.MINIMIZE,
            [[1.0, 4.0], [4.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [0, 1, 2],
            id="min_min",
        ),
        param(
            OptimizationDirection.MAXIMIZE,
            OptimizationDirection.MAXIMIZE,
            [[1.0, 4.0], [4.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [0, 1, 3],
            id="max_max",
        ),
    ],
)  # fmt: skip
def test_compute_best_trials_multi_objective_respects_directions(
    tmp_path: Path,
    d0: OptimizationDirection,
    d1: OptimizationDirection,
    points: list[list[float]],
    expected_front: list[int],
):
    """Dominance flips with each objective's direction, not a hardcoded minimize."""
    cfg = _two_obj_cfg(d0, d1)
    history = [_iter(i, pt) for i, pt in enumerate(points)]
    write_search_history(tmp_path, history, cfg)
    front = _read(tmp_path)["best_trials"]

    assert sorted(p["iteration_idx"] for p in front) == expected_front
    assert all(p["pareto_rank"] == 0 for p in front)


def test_compute_best_trials_multi_objective_feasibility_first(tmp_path: Path):
    """Infeasible dominators are excluded before the front is computed."""
    cfg = _two_obj_cfg()
    # iter 0 dominates everything on raw values but is infeasible; the front is
    # computed over the feasible subset {1, 2}, where (15, 4) dominates (10, 6).
    history = [
        _iter(0, [20.0, 3.0], feasible=False),
        _iter(1, [15.0, 4.0], feasible=True),
        _iter(2, [10.0, 6.0], feasible=True),
    ]
    write_search_history(tmp_path, history, cfg)
    front = _read(tmp_path)["best_trials"]

    assert [p["iteration_idx"] for p in front] == [1]
    assert all(p["feasible_count"] == 2 for p in front)
    assert all(p["feasible"] is True for p in front)


def test_compute_best_trials_multi_objective_all_infeasible_falls_back(
    tmp_path: Path,
):
    """No feasible iterations: front over the full pool, feasible_count == 0."""
    cfg = _two_obj_cfg()
    history = [
        _iter(0, [20.0, 3.0], feasible=False),
        _iter(1, [15.0, 4.0], feasible=False),
    ]
    write_search_history(tmp_path, history, cfg)
    front = _read(tmp_path)["best_trials"]

    assert [p["iteration_idx"] for p in front] == [0]
    assert all(p["feasible_count"] == 0 for p in front)


def test_compute_best_trials_unscored_history_is_none(tmp_path: Path):
    """History with no scored iterations yields best_trials == null."""
    cfg = _two_obj_cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 1},
            objective_value=None,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    assert _read(tmp_path)["best_trials"] is None
