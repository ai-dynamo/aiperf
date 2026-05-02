# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for --search-recipe expansion through the v1->v2 converter."""

import pytest

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_optionals import build_multi_run
from aiperf.config.v1._loadgen import LoadGeneratorConfig


def _user_with_loadgen(**fields) -> UserConfig:
    return UserConfig(loadgen=LoadGeneratorConfig(**fields))


def test_recipe_populates_adaptive_search_block():
    user = _user_with_loadgen(
        search_recipe="max-throughput-ttft-sla", ttft_sla_ms=200.0
    )
    out = build_multi_run(user)
    assert out is not None
    assert "adaptive_search" in out
    ol = out["adaptive_search"]
    assert ol["algorithm"] == "bayes"
    assert ol["objective_metric"] == "output_token_throughput"
    assert ol["objective_direction"] == "maximize"
    assert ol["max_iterations"] == 30
    assert ol["n_initial_points"] == 5
    assert ol["search_space"] == [
        {
            "path": "phases.profiling.concurrency",
            "lo": 1.0,
            "hi": 1000.0,
            "kind": "int",
        },
    ]


def test_recipe_without_ttft_sla_ms_raises_value_error():
    user = _user_with_loadgen(search_recipe="max-throughput-ttft-sla")
    with pytest.raises(ValueError, match="--ttft-sla-ms"):
        build_multi_run(user)


def test_recipe_returns_none_when_unset():
    user = _user_with_loadgen(num_profile_runs=2)
    out = build_multi_run(user)
    assert out == {"num_runs": 2}
    assert "adaptive_search" not in out
