# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: out-of-tree custom Search Recipe loaded via the plugin
manifest path (the same path Python entry_points triggers).

Entry-points discovery (``importlib.metadata.entry_points(group="aiperf.plugins")``)
is one indirection thicker than the manifest load -- it resolves a
``module_name:filename`` reference to a real path, then calls
``load_manifest``. We test the load_manifest leg here because it's the
load-bearing side; the entry_points lookup itself is stdlib.

Verified manually with an installed package (``my-recipes-pkg``) earlier
in the session: ``aiperf plugins search-recipe`` lists the custom
``narrow-ttft-sla`` recipe, ``aiperf profile --search-recipe narrow-
ttft-sla --ttft-sla-ms 100`` runs it end-to-end against the mock server,
and ``search_history.json`` carries the recipe's ``[1, 200]`` search
range and 8-iter limit (vs. the built-ins' [1, 1000] / 30 iters).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1.converter import convert_user_to_aiperf
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


@pytest.fixture
def custom_recipe_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialize a tiny package directory + plugins.yaml for a custom
    recipe and put it on sys.path so ``load_manifest`` can resolve the
    class string ``my_test_recipes:CustomRecipe``."""
    pkg = tmp_path / "my_test_recipes"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        textwrap.dedent(
            """\
            from typing import ClassVar

            from aiperf.common.enums import OptimizationDirection
            from aiperf.config.adaptive_search import (
                AdaptiveSearchConfig,
                SearchSpaceDimension,
            )
            from aiperf.search_recipes._base import (
                SearchRecipe,
                SearchRecipeContext,
                SearchRecipeOutput,
                SLAFilter,
            )


            class CustomRecipe(SearchRecipe):
                name: ClassVar[str] = "custom-test-recipe"
                description: ClassVar[str] = "Test-only custom recipe."

                def expand(self, ctx):
                    threshold = ctx.sla_targets.get("ttft_sla_ms")
                    if threshold is None:
                        raise ValueError(
                            f"recipe {self.name!r} requires --ttft-sla-ms"
                        )
                    return SearchRecipeOutput(
                        adaptive_search=AdaptiveSearchConfig(
                            algorithm="bayes",
                            search_space=[
                                SearchSpaceDimension(
                                    path="phases.profiling.concurrency",
                                    lo=1, hi=99, kind="int",
                                ),
                            ],
                            objective_metric="output_token_throughput",
                            objective_stat="avg",
                            objective_direction=OptimizationDirection.MAXIMIZE,
                            max_iterations=7,
                            n_initial_points=2,
                        ),
                        sla_filters=[
                            SLAFilter(
                                metric_tag="time_to_first_token",
                                stat="p95", op="lt",
                                threshold=float(threshold),
                            ),
                        ],
                    )
            """
        )
    )
    manifest = pkg / "plugins.yaml"
    manifest.write_text(
        textwrap.dedent(
            """\
            schema_version: "1.0"
            search_recipe:
              custom-test-recipe:
                class: my_test_recipes:CustomRecipe
                description: Test-only custom recipe.
                metadata: {}
            """
        )
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    yield manifest


def test_custom_recipe_loaded_via_load_manifest(custom_recipe_module: Path) -> None:
    """The plugin loader's ``load_manifest`` API is what
    ``importlib.metadata.entry_points`` ultimately drives. Loading a
    YAML-described custom recipe must:

    1. Register the class under the ``search_recipe`` category, addressable
       by the recipe's ``name``.
    2. Survive ``aiperf plugins --validate`` (class path resolves).
    3. Be instantiable via ``plugins.get_class`` and produce the right
       ``SearchRecipeOutput`` from ``expand()``.
    """
    plugins.load_manifest(custom_recipe_module, plugin_name="my-test-recipes")
    try:
        recipe_cls = plugins.get_class(PluginType.SEARCH_RECIPE, "custom-test-recipe")
        assert recipe_cls.name == "custom-test-recipe"

        # Build a SearchRecipeContext and exercise expand().
        from aiperf.search_recipes._base import SearchRecipeContext

        user = UserConfig.model_validate(
            {
                "endpoint": {"streaming": True},
                "loadgen": {"ttft_sla_ms": 50.0},
            }
        )
        ctx = SearchRecipeContext(
            user_config=user,
            sla_targets={"ttft_sla_ms": 50.0},
            sweep_overrides={},
        )
        output = recipe_cls().expand(ctx)
        assert output.adaptive_search is not None
        assert output.adaptive_search.max_iterations == 7
        assert output.adaptive_search.search_space[0].hi == 99
        assert len(output.sla_filters) == 1
        assert output.sla_filters[0].threshold == 50.0
    finally:
        # Plugin registry is process-wide; unregister so the next test in the
        # suite sees a clean slate.
        plugins.unregister(PluginType.SEARCH_RECIPE, "custom-test-recipe")


def test_custom_recipe_drives_full_v1_to_v2_conversion(
    custom_recipe_module: Path,
) -> None:
    """End-to-end: a custom recipe loaded via load_manifest must flow
    through ``convert_user_to_aiperf`` exactly like the built-ins, ending
    up at ``AIPerfConfig.multi_run.adaptive_search`` with the recipe's
    custom search-space and SLA filter intact.

    Locks in the contract that there's no special-case for built-in
    recipes -- registry lookups treat in-tree and out-of-tree recipes
    identically.
    """
    plugins.load_manifest(custom_recipe_module, plugin_name="my-test-recipes")
    try:
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                    "streaming": True,
                },
                "loadgen": {
                    "search_recipe": "custom-test-recipe",
                    "ttft_sla_ms": 75.0,
                    "concurrency": 8,
                    "request_count": 30,
                },
            }
        )
        cfg = convert_user_to_aiperf(user, ServiceConfig())
        ad = cfg.multi_run.adaptive_search
        assert ad is not None
        assert ad.recipe_name == "custom-test-recipe"
        # Custom recipe's narrow [1, 99] range, NOT the built-in [1, 1000].
        space = ad.search_space[0]
        assert space.lo == 1
        assert space.hi == 99
        # Custom 7-iter cap, NOT the built-in 30.
        assert ad.max_iterations == 7
        assert ad.sla_filters[0].threshold == 75.0
    finally:
        plugins.unregister(PluginType.SEARCH_RECIPE, "custom-test-recipe")


def test_custom_recipe_missing_required_sla_arg_rejects(
    custom_recipe_module: Path,
) -> None:
    """Custom recipes' own validation paths fire just like the built-ins'
    -- no special handling. Calling without --ttft-sla-ms hits the
    recipe-author's ValueError mid-convert, not a generic crash."""
    plugins.load_manifest(custom_recipe_module, plugin_name="my-test-recipes")
    try:
        user = UserConfig.model_validate(
            {
                "endpoint": {
                    "model_names": ["m"],
                    "urls": ["http://localhost:8000"],
                    "type": "chat",
                    "streaming": True,
                },
                "loadgen": {
                    "search_recipe": "custom-test-recipe",
                    "concurrency": 8,
                    "request_count": 30,
                    # ttft_sla_ms intentionally omitted
                },
            }
        )
        with pytest.raises(ValueError, match="ttft-sla-ms"):
            convert_user_to_aiperf(user, ServiceConfig())
    finally:
        plugins.unregister(PluginType.SEARCH_RECIPE, "custom-test-recipe")
