# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the search_history.json incremental exporter."""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner.base import SearchIteration


def _cfg() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objectives=[
            Objective(
                metric="m",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            ),
        ],
        max_iterations=10,
    )


def test_write_search_history_creates_file(tmp_path: Path):
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"x": 5},
            objective_value=10.0,
            objective_values=[10.0],
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"x": 7},
            objective_value=15.0,
            objective_values=[15.0],
        ),
    ]
    write_search_history(tmp_path, history, _cfg())
    out = tmp_path / "search_history.json"
    assert out.exists()
    data = orjson.loads(out.read_bytes())
    assert len(data["iterations"]) == 2
    assert data["iterations"][1]["objective_values"] == [15.0]
    assert data["best_trials"][0]["objective_values"][0] == 15.0  # MAXIMIZE picks 15
    assert data["best_trials"][0]["iteration_idx"] == 1
    assert data["config"]["objectives"][0]["metric"] == "m"


def test_write_search_history_minimize_picks_smallest(tmp_path: Path):
    cfg = AdaptiveSearchSweep(
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objectives=[
            Objective(
                metric="m",
                stat="avg",
                direction=OptimizationDirection.MINIMIZE,
            ),
        ],
        max_iterations=10,
    )
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"x": 5},
            objective_value=10.0,
            objective_values=[10.0],
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"x": 7},
            objective_value=8.0,
            objective_values=[8.0],
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["best_trials"][0]["iteration_idx"] == 1
    assert data["best_trials"][0]["objective_values"][0] == 8.0


def test_write_search_history_skips_iterations_without_objective(tmp_path: Path):
    history = [
        SearchIteration(
            iteration_idx=0, variation_values={"x": 5}, objective_value=None
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"x": 7},
            objective_value=12.0,
            objective_values=[12.0],
        ),
    ]
    write_search_history(tmp_path, history, _cfg())
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["best_trials"][0]["iteration_idx"] == 1


def test_write_search_history_includes_all_adaptive_config_fields(tmp_path: Path):
    cfg = AdaptiveSearchSweep(
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            ),
        ],
        max_iterations=30,
        n_initial_points=7,
        random_seed=42,
        improvement_patience=8,
        plateau_window=4,
        plateau_threshold=0.025,
    )
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"x": 5},
            objective_value=10.0,
            objective_values=[10.0],
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    config_block = data["config"]
    assert config_block["planner"] == "bayesian"
    assert config_block["objectives"][0]["metric"] == "output_token_throughput"
    assert config_block["objectives"][0]["stat"] == "avg"
    assert config_block["objectives"][0]["direction"] == "MAXIMIZE"
    assert config_block["max_iterations"] == 30
    assert config_block["n_initial_points"] == 7
    assert config_block["random_seed"] == 42
    assert config_block["improvement_patience"] == 8
    assert config_block["plateau_window"] == 4
    assert config_block["plateau_threshold"] == 0.025
    assert config_block["search_space"] == [
        {"path": "x", "lo": 1.0, "hi": 10.0, "kind": "int"}
    ]
    # Field ordering: budget knobs sit between max_iterations and search_space.
    keys = list(config_block.keys())
    assert keys.index("max_iterations") < keys.index("n_initial_points")
    assert keys.index("plateau_threshold") < keys.index("search_space")


def test_write_search_history_random_seed_none_serializes_as_null(tmp_path: Path):
    cfg = _cfg()
    assert cfg.random_seed is None
    write_search_history(tmp_path, [], cfg)
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["config"]["random_seed"] is None


class _PlannerWithoutBoundary:
    """Planner stub whose ``boundary_summary()`` returns None.

    Mirrors the ``SearchPlanner`` ABC default — exporter must fall through
    to history-derived ``_compute_boundary_summary``.
    """

    def boundary_summary(self) -> dict | None:
        return None


class _PlannerWithBoundary:
    """Planner stub returning a precomputed boundary_summary dict."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def boundary_summary(self) -> dict | None:
        return self._payload


def test_write_search_history_planner_none_summary_falls_back_to_history(
    tmp_path: Path,
):
    """Planner returning None for boundary_summary defers to history derivation."""
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"x": 5},
            objective_value=10.0,
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, _cfg(), planner=_PlannerWithoutBoundary())
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    # History-derived path runs (1D space, non-empty history) and produces a
    # block with at least the swept_dim_path key.
    assert data["boundary_summary"] is not None
    assert data["boundary_summary"]["swept_dim_path"] == "x"


def test_write_search_history_planner_precomputed_summary_is_used(tmp_path: Path):
    """Non-None planner.boundary_summary() is used verbatim, no re-derivation."""
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"x": 5},
            objective_value=10.0,
            feasible=True,
        ),
    ]
    sentinel = {"swept_dim_path": "x", "feasible_max": {"sentinel": True}}
    write_search_history(
        tmp_path, history, _cfg(), planner=_PlannerWithBoundary(sentinel)
    )
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["boundary_summary"] == sentinel
