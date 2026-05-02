# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AdaptiveSearchConfig and SearchSpaceDimension."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection


def test_search_space_dimension_int():
    dim = SearchSpaceDimension(
        path="phases.profiling.concurrency", lo=1, hi=1000, kind="int"
    )
    assert dim.path == "phases.profiling.concurrency"
    assert dim.kind == "int"


def test_search_space_dimension_rejects_lo_gt_hi():
    with pytest.raises(ValidationError):
        SearchSpaceDimension(path="x", lo=10, hi=1, kind="int")


def test_adaptive_search_config_minimal():
    cfg = AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=1000, kind="int"
            ),
        ],
        objective_metric="output_token_throughput",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=20,
    )
    assert cfg.max_iterations == 20
    assert cfg.plateau_window == 5  # default


def test_adaptive_search_config_rejects_empty_search_space():
    with pytest.raises(ValidationError):
        AdaptiveSearchConfig(
            algorithm="bayes",
            search_space=[],
            objective_metric="x",
            objective_stat="avg",
            objective_direction=OptimizationDirection.MAXIMIZE,
            max_iterations=20,
        )


def test_adaptive_search_config_rejects_max_iterations_below_two():
    with pytest.raises(ValidationError):
        AdaptiveSearchConfig(
            algorithm="bayes",
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objective_metric="x",
            objective_stat="avg",
            objective_direction=OptimizationDirection.MAXIMIZE,
            max_iterations=1,  # below ge=2
        )


def test_adaptive_search_config_rejects_initial_points_at_or_above_max_iterations():
    with pytest.raises(ValidationError, match="n_initial_points"):
        AdaptiveSearchConfig(
            algorithm="bayes",
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objective_metric="x",
            objective_stat="avg",
            objective_direction=OptimizationDirection.MAXIMIZE,
            max_iterations=5,
            n_initial_points=5,  # not strictly less than max_iterations
        )
