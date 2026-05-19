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


class _MinimalPlanner(SearchPlanner):
    """Minimal SearchPlanner concrete subclass for ABC default-method tests."""

    def ask(self):  # type: ignore[override]
        return None

    def tell(self, variation, results):  # type: ignore[override]
        return None

    def is_converged(self) -> bool:
        return True

    def history(self) -> list[SearchIteration]:
        return []


def test_search_planner_boundary_summary_default_returns_none():
    """ABC's concrete default ``boundary_summary`` returns None.

    Lets planners with no single-boundary concept (e.g. N-D Bayesian) inherit
    a sentinel that the exporter treats as "fall back to history derivation".
    """
    planner = _MinimalPlanner()
    assert planner.boundary_summary() is None


def test_search_planner_convergence_reason_default_returns_none():
    """ABC's concrete default ``convergence_reason`` returns None."""
    planner = _MinimalPlanner()
    assert planner.convergence_reason() is None
