# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.search_recipes._base import SearchRecipe, SearchRecipeOutput


def test_scenarios_branch_accepted() -> None:
    out = SearchRecipeOutput(
        scenarios=[
            {
                "name": "a",
                "benchmark": {"phases": [{"name": "profiling", "concurrency": 1}]},
            },
            {
                "name": "b",
                "benchmark": {"phases": [{"name": "profiling", "concurrency": 4}]},
            },
        ]
    )
    assert out.scenarios is not None
    assert out.adaptive_search is None
    assert out.sweep_parameters is None


def test_exactly_one_branch_zero_branches_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        SearchRecipeOutput()


def test_exactly_one_branch_two_branches_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        SearchRecipeOutput(
            scenarios=[{"name": "a"}],
            sweep_parameters={"x": [1, 2]},
        )


def test_consumed_magic_lists_default_empty() -> None:
    assert getattr(SearchRecipe, "consumed_magic_lists", frozenset()) == frozenset()
