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


def test_multi_objective_config_with_length_one_vectors(tmp_path: Path):
    """A 1-D planner under a 2-objective config must not crash the sweep.

    Only the Optuna planner writes a full objective vector; monotonic.py,
    smooth_isotonic.py and multi_tier_planner.py all hard-code a length-1
    objective_values. _pareto_front indexed objective_values[i] blindly, so
    the IndexError escaped orchestrator.py -- which has no try/except around
    it -- and killed the adaptive sweep after its first scored iteration,
    losing the trajectory file with it.
    """
    cfg = _two_obj_cfg()
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
            variation_values={"concurrency": 50},
            objective_value=20.0,
            objective_values=[20.0],
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())
    # The comparable objective still orders them: 20 beats 10 on maximize.
    assert [t["iteration_idx"] for t in payload["best_trials"]] == [1]


def test_single_objective_nan_first_does_not_win_max(tmp_path: Path):
    """A NaN-scoring iteration must never win max()/min() by list order.

    Reproduces the reported bug directly: with objective values
    [NaN, 10.0, 50.0] and the NaN trial evaluated first, an unfiltered
    max() returns the NaN trial as "best" because every comparison against
    NaN is False. The legitimately-best iteration (idx=2, value=50.0) must
    win regardless of where the NaN trial sits in history.
    """
    cfg = _single_obj_cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 5},
            objective_value=float("nan"),
            objective_values=[float("nan")],
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"concurrency": 10},
            objective_value=10.0,
            objective_values=[10.0],
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=2,
            variation_values={"concurrency": 20},
            objective_value=50.0,
            objective_values=[50.0],
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())
    assert len(payload["best_trials"]) == 1
    assert payload["best_trials"][0]["iteration_idx"] == 2
    assert payload["best_trials"][0]["objective_values"] == [50.0]


def test_all_nan_trial_excluded_from_pareto_front(tmp_path: Path):
    """An all-NaN trial can never be dominated and must not survive on the front.

    Since every comparison against NaN is False, an all-NaN trial would
    never lose a domination check and would be stuck on the Pareto front
    forever with pareto_rank=0. It must be filtered out upstream instead.
    """
    cfg = _two_obj_cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 5},
            objective_value=float("nan"),
            objective_values=[float("nan"), float("nan")],
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"concurrency": 50},
            objective_value=20.0,
            objective_values=[20.0, 8.0],
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())
    front_idxs = [p["iteration_idx"] for p in payload["best_trials"]]
    assert 0 not in front_idxs
    assert front_idxs == [1]


def test_multi_objective_with_absent_vectors(tmp_path: Path):
    """objective_values is Optional; the scalar mirror must carry index 0."""
    cfg = _two_obj_cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 5},
            objective_value=10.0,
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"concurrency": 50},
            objective_value=20.0,
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())
    assert [t["iteration_idx"] for t in payload["best_trials"]] == [1]


def test_non_finite_secondary_objective_cannot_dominate(tmp_path: Path):
    """A NaN on a *secondary* objective must not win the Pareto front.

    The primary-objective filter upstream of the front only rejects a
    non-finite index 0, so a trial scoring [100.0, NaN] survived it. Inside
    _dominates the NaN read back as "unavailable" and the objective was
    skipped, leaving the trial to win on throughput alone -- it dominated
    [50.0, 8.0], a trial strictly better on the very objective that got
    dropped, and then serialized as [100.0, null], indistinguishable from a
    trial that was never scored on latency at all.
    """
    cfg = _two_obj_cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 5},
            objective_values=[50.0, 8.0],
            objective_value=50.0,
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"concurrency": 50},
            objective_values=[100.0, float("nan")],
            objective_value=100.0,
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())

    ranked = [t["iteration_idx"] for t in payload["best_trials"]]
    assert 0 in ranked, (
        "the fully-scored trial must survive: it was strictly better on latency"
    )
    assert ranked == [0], (
        "the poisoned trial must be dropped from the pool, not merely made "
        "incomparable -- incomparable is also undominatable, which would "
        "*guarantee* it a slot in the reported front"
    )
    assert all(
        t["objective_values"] is None or None not in t["objective_values"]
        for t in payload["best_trials"]
    ), "no best trial may serialize a scrubbed non-finite objective as null"


def test_poisoned_trial_is_not_reported_as_pareto_best(tmp_path: Path):
    """Front membership, not just domination, must reject a poisoned trial.

    Distinct from the domination test above: here the poisoned trial is the
    only one that could plausibly top the front, so a fix that merely stops it
    *dominating* still leaves it undominated and therefore reported as
    Pareto-best with pareto_rank=0 -- serialized as [500.0, null], which a
    consumer cannot distinguish from a trial never scored on latency.
    """
    cfg = _two_obj_cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"concurrency": 5},
            objective_values=[10.0, 1.0],
            objective_value=10.0,
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"concurrency": 50},
            objective_values=[20.0, 2.0],
            objective_value=20.0,
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=2,
            variation_values={"concurrency": 99},
            objective_values=[500.0, float("inf")],
            objective_value=500.0,
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = json.loads((tmp_path / "search_history.json").read_text())

    ranked = [t["iteration_idx"] for t in payload["best_trials"]]
    assert 2 not in ranked, (
        "a trial with a non-finite objective must never be reported as best"
    )
    assert ranked, "dropping the poisoned trial must not empty the front"
    assert set(ranked) <= {0, 1}
