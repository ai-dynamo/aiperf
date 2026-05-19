# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_isl_osl_pairs_flag_lives_on_sweeping() -> None:
    """The --isl-osl-pairs flag is a top-level field on CLIConfig (sweeping section)."""
    from aiperf.config.flags.cli_config import CLIConfig

    user = CLIConfig.model_validate({"isl_osl_pairs": "128/128,256/256"})
    assert user.isl_osl_pairs == "128/128,256/256"


def test_isl_osl_pairs_flag_default_none() -> None:
    from aiperf.config.flags.cli_config import CLIConfig

    user = CLIConfig.model_validate({})
    assert user.isl_osl_pairs is None


def test_recipe_output_to_dict_scenarios_branch() -> None:
    """_recipe_output_to_dict projects the scenarios branch verbatim."""
    from aiperf.config.flags.recipes import _recipe_output_to_dict
    from aiperf.search_recipes._base import SearchRecipeOutput

    output = SearchRecipeOutput(
        scenarios=[
            {
                "name": "a",
                "benchmark": {"phases": [{"name": "profiling", "concurrency": 1}]},
            },
            {
                "name": "b",
                "benchmark": {"phases": [{"name": "profiling", "concurrency": 2}]},
            },
        ]
    )
    out = _recipe_output_to_dict(output, "fake-recipe")
    assert "scenarios" in out
    assert len(out["scenarios"]) == 2
    assert out["scenarios"][0]["name"] == "a"


def test_reject_recipe_plus_magic_lists_honors_consumed() -> None:
    """A recipe whose consumed_magic_lists includes 'concurrency' is not rejected
    when the user passes --concurrency 1,2,4 alongside the recipe."""
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.config.flags.converter import _reject_recipe_plus_magic_lists

    class _StubRecipe:
        name = "stub"
        consumed_magic_lists = frozenset({"concurrency"})

        def expand(self, ctx):  # pragma: no cover - not invoked in this test
            raise NotImplementedError

    user = CLIConfig(
        **CLIConfig(concurrency=[1, 2, 4]).model_dump(exclude_unset=True),
        search_recipe="stub",
    )
    # No exception should be raised
    _reject_recipe_plus_magic_lists(user, recipe_cls=_StubRecipe)


def test_reject_recipe_plus_magic_lists_still_fires_for_non_consumed() -> None:
    """A recipe that does NOT consume 'concurrency' still rejects --concurrency lists."""
    import pytest

    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.config.flags.converter import _reject_recipe_plus_magic_lists

    class _StubRecipe:
        name = "stub"
        # Empty allowlist: any magic-list flag is an offender.
        consumed_magic_lists: frozenset[str] = frozenset()

        def expand(self, ctx):  # pragma: no cover
            raise NotImplementedError

    user = CLIConfig(
        **CLIConfig(concurrency=[1, 2, 4]).model_dump(exclude_unset=True),
        search_recipe="stub",
    )
    with pytest.raises(TypeError, match="magic-list flags"):
        _reject_recipe_plus_magic_lists(user, recipe_cls=_StubRecipe)


def test_reject_recipe_plus_magic_lists_no_recipe_cls_preserves_legacy() -> None:
    """Default recipe_cls=None preserves the pre-Task-4 reject-everything behavior."""
    import pytest

    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.config.flags.converter import _reject_recipe_plus_magic_lists

    user = CLIConfig(
        **CLIConfig(concurrency=[1, 2, 4]).model_dump(exclude_unset=True),
        search_recipe="stub",
    )
    with pytest.raises(TypeError, match="magic-list flags"):
        _reject_recipe_plus_magic_lists(user)
