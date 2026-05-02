# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the v1 --search-* CLI flags (parsing only, no execution)."""

from __future__ import annotations

from aiperf.config.v1._loadgen import LoadGeneratorConfig


def test_loadgen_has_search_fields():
    fields = LoadGeneratorConfig.model_fields
    assert "search_space" in fields
    assert "search_metric" in fields
    assert "search_stat" in fields
    assert "search_direction" in fields
    assert "search_max_iterations" in fields
    assert "search_random_seed" in fields
    assert "search_initial_points" in fields


def test_loadgen_search_defaults_unset():
    """When the user supplies no --search-* flags, all fields are None/unset."""
    lg = LoadGeneratorConfig()
    assert lg.search_space is None
    assert lg.search_metric is None
    assert lg.search_stat is None
    assert lg.search_direction is None
    assert lg.search_max_iterations is None


def test_loadgen_accepts_search_space_list():
    lg = LoadGeneratorConfig(search_space=["phases.profiling.concurrency:1,1000:int"])
    assert lg.search_space == ["phases.profiling.concurrency:1,1000:int"]


def test_loadgen_accepts_search_objective_fields():
    lg = LoadGeneratorConfig(
        search_metric="output_token_throughput",
        search_stat="p99",
        search_direction="maximize",
    )
    assert lg.search_metric == "output_token_throughput"
    assert lg.search_stat == "p99"
    assert lg.search_direction == "maximize"
