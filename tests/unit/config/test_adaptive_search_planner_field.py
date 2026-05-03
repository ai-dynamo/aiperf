# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AdaptiveSearchConfig.planner field and --search-planner CLI flag."""

import pytest
from pydantic import ValidationError

from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.plugin.enums import SearchPlannerType


def _minimal_kwargs(**overrides):
    base = dict(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            )
        ],
        objective_metric="output_token_throughput",
        objective_direction="maximize",
        max_iterations=10,
    )
    base.update(overrides)
    return base


def test_adaptive_search_config_default_planner_is_bayesian():
    """Default planner is `bayesian` when --search-planner is not passed."""
    cfg = AdaptiveSearchConfig(**_minimal_kwargs())
    assert cfg.planner == SearchPlannerType.BAYESIAN


def test_adaptive_search_config_accepts_explicit_bayesian():
    """`planner='bayesian'` parses to SearchPlannerType.BAYESIAN."""
    cfg = AdaptiveSearchConfig(**_minimal_kwargs(planner="bayesian"))
    assert cfg.planner == SearchPlannerType.BAYESIAN


def test_adaptive_search_config_rejects_unknown_planner():
    """Names not registered under the `search_planner` plugin category are rejected."""
    with pytest.raises(ValidationError):
        AdaptiveSearchConfig(**_minimal_kwargs(planner="not-a-real-planner"))
