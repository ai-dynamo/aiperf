# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution-rule tests for the v1 ``CLIConfig.auto_plot`` flag -> v2
``ArtifactsConfig.auto_plot``.

Covers C4 of the auto-plot design: the v1->v2 converter resolves the
tri-state CLI flag against the active recipe's ``auto_plot_default`` (read
via ``getattr`` so external plugin recipes that omit the attribute keep
working) before writing a plain bool into the v2 artifacts dict.

Resolution table (CLI x recipe):
    CLI=None,   no recipe                 -> False
    CLI=None,   recipe.default=True       -> True
    CLI=True,   recipe.default=False      -> True
    CLI=False,  recipe.default=True       -> False
    CLI=None,   recipe missing the attr   -> False (getattr fallback)

``plot_required`` is a pure pass-through.
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.config.flags._converter_optionals import resolve_auto_plot
from aiperf.config.flags.cli_config import CLIConfig


class _RecipeWithDefaultTrue:
    """Test double: recipe class exposing ``auto_plot_default = True``."""

    auto_plot_default: ClassVar[bool] = True


class _RecipeWithDefaultFalse:
    auto_plot_default: ClassVar[bool] = False


class _RecipeWithoutAttr:
    """Test double: recipe class that omits ``auto_plot_default`` entirely;
    the converter must fall back to False via ``getattr(..., False)``."""


def _make_user(
    *,
    auto_plot: bool | None = None,
    plot_required: bool = False,
    recipe: str | None = None,
) -> CLIConfig:
    """Build a v1 ``CLIConfig`` with only the auto-plot-relevant fields set.

    Avoids touching unrelated sections (endpoint/input/loadgen) since
    ``resolve_auto_plot`` only reads the flat output fields and the sweeping
    section fields.
    """
    kwargs: dict[str, object] = {
        "auto_plot": auto_plot,
        "plot_required": plot_required,
    }
    if recipe is not None:
        kwargs["search_recipe"] = recipe
    return CLIConfig(**kwargs)


def _patched_get_class(recipe_cls: type | None):
    """Patch ``get_class`` in the converter's late-import namespace so the
    test recipe class is returned regardless of plugin registration."""
    target = "aiperf.plugin.plugins.get_class"
    if recipe_cls is None:
        return patch(target, side_effect=KeyError("unknown"))
    return patch(target, return_value=recipe_cls)


@pytest.mark.parametrize(
    ("cli_value", "recipe_cls", "expected"),
    [
        param(None, None, False, id="cli-none-no-recipe"),
        param(None, _RecipeWithDefaultTrue, True, id="cli-none-recipe-default-true"),
        param(None, _RecipeWithDefaultFalse, False, id="cli-none-recipe-default-false"),
        param(True, None, True, id="cli-true-no-recipe"),
        param(True, _RecipeWithDefaultFalse, True, id="cli-true-overrides-recipe-false"),
        param(False, _RecipeWithDefaultTrue, False, id="cli-false-overrides-recipe-true"),
        param(None, _RecipeWithoutAttr, False, id="cli-none-recipe-missing-attr"),
    ],
)  # fmt: skip
def test_resolve_auto_plot_truth_table(
    cli_value: bool | None, recipe_cls: type | None, expected: bool
) -> None:
    """Truth table for the resolution rule."""
    recipe_name = "test-recipe" if recipe_cls is not None else None
    user = _make_user(auto_plot=cli_value, recipe=recipe_name)
    with _patched_get_class(recipe_cls):
        auto_plot, _ = resolve_auto_plot(user)
    assert auto_plot is expected


@pytest.mark.parametrize(
    "value",
    [param(True, id="strict"), param(False, id="warn-default")],
)  # fmt: skip
def test_resolve_auto_plot_plot_required_pass_through(value: bool) -> None:
    """``plot_required`` is independent of ``auto_plot`` resolution."""
    user = _make_user(auto_plot=None, plot_required=value)
    _, plot_required = resolve_auto_plot(user)
    assert plot_required is value


def test_resolve_auto_plot_default_cli_config() -> None:
    """A bare ``CLIConfig()`` (no auto-plot/plot-required flags set) resolves
    cleanly to (False, False) without crashing. Previously this guarded
    against ``user.output`` being None; now the flat fields default to
    None / False on CLIConfig directly."""
    user = CLIConfig()
    auto_plot, plot_required = resolve_auto_plot(user)
    assert auto_plot is False
    assert plot_required is False


def test_resolve_auto_plot_unknown_recipe_falls_back() -> None:
    """When ``get_class`` raises (recipe name unknown), the resolver
    treats it as 'no recipe' and returns the explicit CLI value or False
    -- the actual unknown-recipe error is raised later by ``_invoke_recipe``,
    which is the single source of truth for recipe-name validation."""
    user = _make_user(auto_plot=None, recipe="does-not-exist")
    with _patched_get_class(None):
        auto_plot, _ = resolve_auto_plot(user)
    assert auto_plot is False
