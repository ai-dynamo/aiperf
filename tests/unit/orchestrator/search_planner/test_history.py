# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the search_history.json incremental exporter."""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner.base import SearchIteration


def _cfg() -> AdaptiveSearchConfig:
    return AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objective_metric="m",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=10,
    )


def test_write_search_history_creates_file(tmp_path: Path):
    history = [
        SearchIteration(
            iteration_idx=0, variation_values={"x": 5}, objective_value=10.0
        ),
        SearchIteration(
            iteration_idx=1, variation_values={"x": 7}, objective_value=15.0
        ),
    ]
    write_search_history(tmp_path, history, _cfg())
    out = tmp_path / "search_history.json"
    assert out.exists()
    data = orjson.loads(out.read_bytes())
    assert len(data["iterations"]) == 2
    assert data["iterations"][1]["objective_value"] == 15.0
    assert data["best"]["objective_value"] == 15.0  # MAXIMIZE picks 15
    assert data["best"]["iteration_idx"] == 1
    assert data["config"]["objective_metric"] == "m"


def test_write_search_history_minimize_picks_smallest(tmp_path: Path):
    cfg = AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objective_metric="m",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MINIMIZE,
        max_iterations=10,
    )
    history = [
        SearchIteration(
            iteration_idx=0, variation_values={"x": 5}, objective_value=10.0
        ),
        SearchIteration(
            iteration_idx=1, variation_values={"x": 7}, objective_value=8.0
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["best"]["iteration_idx"] == 1
    assert data["best"]["objective_value"] == 8.0


def test_write_search_history_skips_iterations_without_objective(tmp_path: Path):
    history = [
        SearchIteration(
            iteration_idx=0, variation_values={"x": 5}, objective_value=None
        ),
        SearchIteration(
            iteration_idx=1, variation_values={"x": 7}, objective_value=12.0
        ),
    ]
    write_search_history(tmp_path, history, _cfg())
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["best"]["iteration_idx"] == 1
