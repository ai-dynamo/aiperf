# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search Recipes: named, plugin-registered presets that compile to BO/grid configs.

Public entry points:
    - ``SearchRecipe``: Protocol implemented by each recipe.
    - ``SearchRecipeContext`` / ``SearchRecipeOutput``: I/O dataclasses.
    - ``SLAFilter`` / ``PostProcessSpec``: typed building blocks (Phase 2/3 wiring).

Built-in recipe implementations live in ``aiperf.search_recipes.builtins``.
"""

from aiperf.search_recipes._base import (
    PostProcessSpec,
    SearchRecipe,
    SearchRecipeContext,
    SearchRecipeOutput,
    SLAFilter,
)

__all__ = [
    "PostProcessSpec",
    "SLAFilter",
    "SearchRecipe",
    "SearchRecipeContext",
    "SearchRecipeOutput",
]
