# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for --search-recipe vs explicit --search-* mutual exclusion."""

import pytest

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_optionals import build_multi_run
from aiperf.config.v1._loadgen import LoadGeneratorConfig


def _user_with_loadgen(**fields) -> UserConfig:
    return UserConfig(loadgen=LoadGeneratorConfig(**fields))


def test_recipe_with_explicit_search_space_raises_type_error():
    user = _user_with_loadgen(
        search_recipe="max-throughput-ttft-sla",
        ttft_sla_ms=200.0,
        search_space=["phases.profiling.concurrency:1,1000:int"],
    )
    with pytest.raises(TypeError, match="--search-recipe"):
        build_multi_run(user)


def test_recipe_with_explicit_search_metric_raises_type_error():
    user = _user_with_loadgen(
        search_recipe="max-throughput-ttft-sla",
        ttft_sla_ms=200.0,
        search_metric="some_metric",
    )
    with pytest.raises(TypeError, match="--search-recipe"):
        build_multi_run(user)


def test_recipe_alone_succeeds():
    user = _user_with_loadgen(
        search_recipe="max-throughput-ttft-sla",
        ttft_sla_ms=200.0,
    )
    out = build_multi_run(user)
    assert out is not None
    assert "adaptive_search" in out
