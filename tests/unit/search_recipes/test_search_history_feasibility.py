# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for write_search_history's lexicographic feasibility-first best.

Phase 2 of the search-recipes feature. write_search_history must:
- prefer the best feasible iteration when any feasible iterations exist;
- fall back to the best of the full pool when none are feasible (with
  feasible_count == 0 so readers can tell the two cases apart);
- surface ``recipe`` (the recipe name) and ``sla_filters`` for post-run audit.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.config.adaptive_search import (
    AdaptiveSearchConfig,
    SearchSpaceDimension,
    SLAFilter,
)
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner.base import SearchIteration


def _cfg(**overrides) -> AdaptiveSearchConfig:
    kwargs: dict = dict(
        algorithm="bayes",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            ),
        ],
        objective_metric="output_token_throughput",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=5,
        n_initial_points=2,
    )
    kwargs.update(overrides)
    return AdaptiveSearchConfig(**kwargs)


def _read(base_dir: Path) -> dict:
    return orjson.loads((base_dir / "search_history.json").read_bytes())


def test_write_search_history_picks_feasible_best_over_higher_objective(
    tmp_path: Path,
):
    """MAXIMIZE: a feasible point with low objective beats an infeasible high one."""
    cfg = _cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 10},
            objective_value=50.0,
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"phases.profiling.concurrency": 100},
            objective_value=1000.0,
            feasible=False,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["best"]["iteration_idx"] == 0
    assert payload["best"]["objective_value"] == 50.0
    assert payload["best"]["feasible"] is True
    assert payload["best"]["feasible_count"] == 1


def test_write_search_history_falls_back_to_best_infeasible_when_none_feasible(
    tmp_path: Path,
):
    """All-infeasible: pick the best of the full pool, feasible_count == 0."""
    cfg = _cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 10},
            objective_value=50.0,
            feasible=False,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"phases.profiling.concurrency": 100},
            objective_value=1000.0,
            feasible=False,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["best"]["iteration_idx"] == 1
    assert payload["best"]["objective_value"] == 1000.0
    assert payload["best"]["feasible"] is False
    assert payload["best"]["feasible_count"] == 0


def test_write_search_history_records_recipe_name_and_sla_filters(tmp_path: Path):
    """Recipe metadata lands in the top-level payload + config block."""
    cfg = _cfg(
        recipe_name="max-throughput-ttft-sla",
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=200.0,
            ),
        ],
    )
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 10},
            objective_value=50.0,
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["recipe"] == "max-throughput-ttft-sla"
    assert payload["config"]["sla_filters"] == [
        {
            "metric_tag": "time_to_first_token",
            "stat": "p95",
            "op": "lt",
            "threshold": 200.0,
        }
    ]


def test_write_search_history_recipe_is_none_for_explicit_search_flags(
    tmp_path: Path,
):
    """Without a recipe, payload['recipe'] is null and sla_filters is empty."""
    cfg = _cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 10},
            objective_value=50.0,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["recipe"] is None
    assert payload["config"]["sla_filters"] == []
