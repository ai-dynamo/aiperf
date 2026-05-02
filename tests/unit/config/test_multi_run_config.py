# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunConfig.adaptive_search field."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config._models_benchmark import MultiRunConfig
from aiperf.config.adaptive_search import AdaptiveSearchConfig


def test_multi_run_default_no_adaptive_search():
    cfg = MultiRunConfig()
    assert cfg.adaptive_search is None


def test_multi_run_accepts_adaptive_search_dict():
    cfg = MultiRunConfig.model_validate(
        {
            "num_runs": 2,
            "adaptive_search": {
                "algorithm": "bayes",
                "search_space": [{"path": "x", "lo": 1, "hi": 10, "kind": "int"}],
                "objective_metric": "m",
                "objective_stat": "avg",
                "objective_direction": "maximize",
                "max_iterations": 10,
            },
        }
    )
    assert isinstance(cfg.adaptive_search, AdaptiveSearchConfig)
    assert cfg.adaptive_search.max_iterations == 10


def test_multi_run_rejects_unknown_top_level_keys():
    """extra='forbid' protects MultiRunConfig from drift; verify it still fires."""
    with pytest.raises(ValidationError):
        MultiRunConfig.model_validate({"num_runs": 2, "adaptive_search_raw": {}})
