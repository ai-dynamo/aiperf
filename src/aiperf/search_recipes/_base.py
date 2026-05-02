# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base types for the Search Recipe plugin category.

Search Recipes are named, plugin-registered presets that compile down to existing
v2 fields (``AdaptiveSearchConfig`` for BO, ``sweep.variables`` for grid) at the
v1->v2 converter boundary. The recipe NAME never reaches ``AIPerfConfig``: it is a
v1-CLI-only input that expands into existing canonical fields.

See ``aiperf.search_recipes.builtins`` for the first concrete implementation
(``MaxThroughputUnderTTFTSLA``).
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from aiperf.config._base import BaseConfig
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SLAFilter

__all__ = [
    "PostProcessSpec",
    "SLAFilter",
    "SearchRecipe",
    "SearchRecipeContext",
    "SearchRecipeOutput",
]


class PostProcessSpec(BaseConfig):
    """Post-aggregation hook spec emitted by a search recipe.

    Phase 1 declares the type only; Phase 3 wires it into
    ``aggregate_sweep_and_export`` via a ``SEARCH_RECIPE_POST_PROCESS`` plugin
    category.

    Example: ``PostProcessSpec(handler="ttft_sla_curve", params={"sla_ms": 200},
    output_filename="ttft_sla_curve.json")``.
    """

    model_config = ConfigDict(extra="forbid")

    handler: str = Field(
        description=(
            "Registered post-process handler name. Looked up via the "
            "search_recipe_post_process plugin category in Phase 3."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form parameters forwarded to the handler. Shape is handler-defined; "
            "Any is intentional because handlers ingest dynamic recipe-derived inputs."
        ),
    )
    output_filename: str = Field(
        description=(
            "Filename (relative to ``sweep_aggregate/`` under the artifact dir) "
            "to write the handler's output to. Must end in ``.json`` and contain "
            "no path separators -- the hook joins it with ``sweep_aggregate/`` "
            "and refuses to escape that directory."
        ),
    )

    @model_validator(mode="after")
    def _check_output_filename(self) -> PostProcessSpec:
        # Reject path separators outright so a recipe can't write to ``..`` or
        # an absolute path; the hook joins this filename with ``sweep_aggregate/``
        # and we want filename-only there.
        if "/" in self.output_filename or "\\" in self.output_filename:
            raise ValueError(
                f"PostProcessSpec.output_filename must be filename-only "
                f"(no path separators), got {self.output_filename!r}."
            )
        if not self.output_filename.endswith(".json"):
            raise ValueError(
                f"PostProcessSpec.output_filename must end in '.json' "
                f"(handlers emit JSON artifacts), got {self.output_filename!r}."
            )
        return self


class SearchRecipeContext(BaseConfig):
    """Inputs available to a recipe at expand time.

    Recipes read user-provided CLI inputs from this context (rather than reaching
    into ``UserConfig`` everywhere) so that converter-side wiring can pre-extract
    only the fields a recipe is allowed to depend on.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    user_config: Any = Field(
        description=(
            "The fully-populated v1 UserConfig the recipe is expanding under. Typed "
            "Any because aiperf.search_recipes lives downstream of the v1 import "
            "boundary (TID251); recipes treat it as a read-only dotted-path object."
        ),
    )
    sla_targets: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Recipe-specific SLA target values keyed by short name (e.g. 'ttft_sla_ms'). "
            "Populated from CLI flags like --ttft-sla-ms by the v1->v2 converter."
        ),
    )
    sweep_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Recipe-specific sweep overrides (e.g. concurrency bounds). Populated from "
            "CLI flags. Any is intentional because recipe-defined fields are dynamic."
        ),
    )


class SearchRecipeOutput(BaseConfig):
    """Compiled output of a recipe's ``expand()`` call.

    Exactly one of ``adaptive_search`` (BO) or ``sweep_variables`` (grid) MUST be set.
    The v1->v2 converter writes the populated branch into the corresponding
    ``LoadGeneratorConfig`` fields so the existing adaptive-search /
    magic-list paths consume it without needing recipe-aware code.
    """

    model_config = ConfigDict(extra="forbid")

    adaptive_search: AdaptiveSearchConfig | None = Field(
        default=None,
        description=(
            "Bayesian-optimized adaptive search config. Mutually exclusive with "
            "sweep_variables."
        ),
    )
    sweep_variables: dict[str, list[Any]] | None = Field(
        default=None,
        description=(
            "Grid-sweep variables as a path -> list-of-values map (matches the "
            "shape of sweep.variables). Mutually exclusive with adaptive_search. "
            "Any is intentional because grid values are dynamically typed per dimension."
        ),
    )
    sla_filters: list[SLAFilter] = Field(
        default_factory=list,
        description=(
            "SLA filters produced by the recipe. Phase 1 carries these through "
            "without enforcement; Phase 2 wires them into BO scoring."
        ),
    )
    post_process: PostProcessSpec | None = Field(
        default=None,
        description=(
            "Optional post-aggregation handler spec. Phase 3 wires it into "
            "aggregate_sweep_and_export."
        ),
    )

    @model_validator(mode="after")
    def _check_exactly_one_branch(self) -> SearchRecipeOutput:
        has_adaptive = self.adaptive_search is not None
        has_sweep = self.sweep_variables is not None
        if has_adaptive == has_sweep:
            raise ValueError(
                "SearchRecipeOutput requires exactly one of "
                "'adaptive_search' or 'sweep_variables' to be set "
                f"(got adaptive_search={has_adaptive}, sweep_variables={has_sweep})."
            )
        return self


@runtime_checkable
class SearchRecipe(Protocol):
    """Protocol for a Search Recipe plugin.

    Implementations must expose two ClassVars (``name``, ``description``) and an
    ``expand(ctx) -> SearchRecipeOutput`` method. Recipes are registered under the
    ``search_recipe`` plugin category.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the recipe under the given context.

        Args:
            ctx: User-config + SLA targets + sweep overrides snapshot.

        Returns:
            A populated ``SearchRecipeOutput`` with exactly one of
            ``adaptive_search`` / ``sweep_variables`` set.

        Raises:
            ValueError: If the recipe's required inputs are missing or if the
                user's UserConfig conflicts with the recipe's assumptions.
        """
        ...
