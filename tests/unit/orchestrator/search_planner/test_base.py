# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SearchIteration dataclass and SearchPlanner ABC surface."""

from __future__ import annotations

from aiperf.orchestrator.search_planner.base import SearchIteration, SearchPlanner


def test_outer_iteration_dataclass_defaults():
    it = SearchIteration(iteration_idx=3, variation_values={"x": 42})
    assert it.iteration_idx == 3
    assert it.objective_value is None
    assert it.results == []


def test_outer_iteration_with_objective():
    it = SearchIteration(
        iteration_idx=0,
        variation_values={"x": 1},
        objective_value=12.5,
    )
    assert it.objective_value == 12.5


def test_search_planner_is_abstract():
    """ABC: cannot instantiate without concrete impls."""
    import pytest

    with pytest.raises(TypeError):
        SearchPlanner()  # type: ignore[abstract]
