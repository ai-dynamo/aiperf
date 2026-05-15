# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search Recipes: named, plugin-registered presets for BO/grid sweep configs.

``PostProcessSpec`` is imported eagerly because ``aiperf.config.sweep`` needs it
while the config package is still initializing. Heavier recipe base types are
loaded lazily because ``_base`` imports ``aiperf.config.config.BenchmarkConfig``,
which would cycle back through ``aiperf.config.sweep`` if pulled at package
import time.
"""

from typing import Any

from aiperf.search_recipes._post_process import PostProcessSpec

__all__ = [
    "PostProcessSpec",
    "SLAFilter",
    "SearchRecipe",
    "SearchRecipeContext",
    "SearchRecipeOutput",
]

_LAZY_EXPORTS = {
    "SLAFilter",
    "SearchRecipe",
    "SearchRecipeContext",
    "SearchRecipeOutput",
}


def __getattr__(name: str) -> Any:
    """Load recipe base exports lazily after config initialization."""
    if name in _LAZY_EXPORTS:
        from aiperf.search_recipes import _base

        return getattr(_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
