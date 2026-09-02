# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base types for the Search Recipe plugin category.

Search Recipes are named, plugin-registered presets that compile down to
canonical sweep fields (``AdaptiveSearchSweep`` for BO,
``sweep.parameters`` for grid) at the CLI converter boundary
(``aiperf.config.flags._converter_optionals.expand_search_recipe``,
which delegates to ``aiperf.config.flags.recipes.maybe_expand_search_recipe``).
The recipe NAME never reaches ``AIPerfConfig``: it is a CLI-only input
that expands into existing canonical fields.

See ``aiperf.search_recipes.builtins`` for the concrete implementations
(``MaxThroughputUnderTTFTSLA`` and friends).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from aiperf.config.base import BaseConfig
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep.adaptive import SLAFilter

if TYPE_CHECKING:
    from aiperf.config.sweep import AdaptiveSearchSweep
from aiperf.search_recipes._pareto_axes import ParetoAxesSpec
from aiperf.search_recipes._post_process import PostProcessSpec

__all__ = [
    "PostProcessSpec",
    "SLAFilter",
    "SearchRecipe",
    "SearchRecipeContext",
    "SearchRecipeOutput",
    "get_inter_token_sla_ms",
    "require_streaming",
    "resolve_concurrency_bounds",
]


def resolve_concurrency_bounds(
    overrides: dict[str, Any],
    *,
    recipe_name: str,
    default_lo: int,
    default_hi: int,
) -> tuple[int, int]:
    """Read concurrency_min / concurrency_max overrides with bounds validation.

    Returns (lo, hi) where each falls back to the recipe-provided default when
    the override is unset. Raises ``ValueError`` if the resolved bounds are
    inverted (lo >= hi) so users see a clear error instead of a downstream
    Sobol/grid generator failure.

    Recipes call this in lieu of reading the override keys directly so the
    inverted-bounds rejection lives in one place.
    """
    lo = int(overrides.get("concurrency_min", default_lo))
    hi = int(overrides.get("concurrency_max", default_hi))
    if lo >= hi:
        raise ValueError(
            f"recipe {recipe_name!r}: --concurrency-min ({lo}) must be < "
            f"--concurrency-max ({hi}); inverted bounds collapse the search space."
        )
    return lo, hi


def get_inter_token_sla_ms(sla_targets: dict[str, float]) -> float | None:
    """Read the inter-token-latency SLA threshold from CLI sla_targets.

    ``--tpot-sla-ms`` (canonical) and ``--itl-sla-ms`` are aliases for the
    same underlying ``inter_token_latency`` metric. Recipes call this helper
    instead of reading either key directly so users can pass either flag and
    so an explicit conflict (both set with different values) raises a clear
    error instead of silently preferring one.

    Args:
        sla_targets: ``ctx.sla_targets`` dict; typically contains zero or one
            of ``tpot_sla_ms`` / ``itl_sla_ms``.

    Returns:
        The threshold in milliseconds, or ``None`` if neither is set.

    Raises:
        ValueError: If both keys are set to different values.
    """
    tpot = sla_targets.get("tpot_sla_ms")
    itl = sla_targets.get("itl_sla_ms")
    if tpot is not None and itl is not None and float(tpot) != float(itl):
        raise ValueError(
            "--tpot-sla-ms and --itl-sla-ms are aliases for the same "
            f"inter-token-latency SLA but were set to different values "
            f"({tpot} vs {itl}); pass only one."
        )
    if tpot is not None:
        return float(tpot)
    if itl is not None:
        return float(itl)
    return None


def require_streaming(
    endpoint: Any,
    *,
    recipe_name: str,
    reason: str,
) -> None:
    """Raise ValueError if the user has explicitly disabled streaming.

    Use this when a recipe references streaming-only metrics (TTFT, ITL/TPOT).
    The ``is False`` check (not ``not endpoint.streaming``) lets an unset
    (None) streaming flag fall through — only an explicit ``--no-streaming``
    hard-rejects.

    Args:
        endpoint: ``ctx.benchmark_config.endpoint``. Typed ``Any`` so this
            helper only depends on the structural ``streaming`` attribute.
        recipe_name: ``self.name`` (for clear error messages).
        reason: Human-readable rationale, e.g.
            ``"TTFT is a streaming-only metric"`` or
            ``"TPOT is a streaming-only metric"``. Slotted into the error
            message in parentheses after ``--streaming``.

    Raises:
        ValueError: If ``endpoint.streaming is False``.
    """
    if endpoint is not None and endpoint.streaming is False:
        raise ValueError(
            f"recipe {recipe_name!r} requires --streaming ({reason}); "
            f"enable streaming on the endpoint or pick a different recipe."
        )


class SearchRecipeContext(BaseConfig):
    """Inputs available to a recipe at expand time.

    Recipes read a validated BenchmarkConfig plus recipe-specific CLI inputs
    from this context so expansion stays decoupled from CLI DTOs.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    benchmark_config: BenchmarkConfig = Field(
        description=(
            "Validated BenchmarkConfig snapshot available to recipe expansion. "
            "Recipes treat it as read-only and currently inspect endpoint.streaming "
            "before emitting streaming-only metric recipes."
        ),
    )
    sla_targets: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Recipe-specific SLA target values keyed by short name (e.g. 'ttft_sla_ms'). "
            "Populated from CLI flags like --ttft-sla-ms by the CLI converter "
            "(``aiperf.config.flags.recipes._invoke_recipe``)."
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

    Exactly one of ``adaptive_search`` (BO), ``sweep_parameters`` (grid), or
    ``scenarios`` (pre-flattened ScenarioSweep runs) MUST be set.
    The CLI converter (``expand_search_recipe`` +
    ``_apply_recipe_sweep_parameters``) writes the populated branch into the
    corresponding sweep envelope shape so the existing adaptive-search /
    magic-list paths consume it without needing recipe-aware code.
    """

    model_config = ConfigDict(extra="forbid")

    adaptive_search: AdaptiveSearchSweep | None = Field(
        default=None,
        description=(
            "Bayesian-optimized adaptive search config. Mutually exclusive with "
            "sweep_parameters and scenarios."
        ),
    )
    sweep_parameters: dict[str, list[Any]] | None = Field(
        default=None,
        description=(
            "Grid-sweep parameters as a path -> list-of-values map (matches the "
            "shape of sweep.parameters). Mutually exclusive with adaptive_search. "
            "Any is intentional because grid values are dynamically typed per "
            "dimension. Routed through the CLI converter by "
            "``aiperf.config.flags.converter._apply_recipe_sweep_parameters``, "
            "which prefixes each path with ``benchmark.`` and lifts the dict "
            "into the top-level ``sweep.parameters`` block (creating a "
            "``GridSweep`` envelope when none exists). Used by recipes like "
            "``max-concurrency-under-sla --search-style grid`` and "
            "``concurrency-ramp``. The field validator below enforces the "
            "exactly-one-of contract on the union."
        ),
    )
    scenarios: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Pre-flattened ScenarioSweep runs. Mutually exclusive with "
            "adaptive_search and sweep_parameters. Each entry is a dict with "
            "a ``name`` key and a deep-merged ``benchmark`` subtree applied "
            "on top of the base config -- exactly the shape ScenarioSweep.runs "
            "consumes. Used by recipes that need paired (non-Cartesian) sweep "
            "axes, e.g. pareto-sweep over (isl, osl) pairs."
        ),
    )
    sla_filters: list[SLAFilter] = Field(
        default_factory=list,
        description=(
            "SLA filters produced by the recipe. Carried through onto "
            "AdaptiveSearchSweep.sla_filters (BO path) or sweep.sla_filters "
            "(grid path) by the CLI converter."
        ),
    )
    post_process: PostProcessSpec | None = Field(
        default=None,
        description=(
            "Optional post-aggregation handler spec invoked by "
            "``aggregate_sweep_and_export`` after per-variation aggregation."
        ),
    )
    slos: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-request SLO thresholds keyed by metric tag (e.g. "
            "{'time_to_first_token': 500, 'inter_token_latency': 15, "
            "'request_latency': 2000}). The config assembly pipeline merges "
            "this into BenchmarkConfig.slos so GoodRequestCountMetric.set_slos() "
            "can apply the per-request 'good' criterion. Set by goodput-style recipes "
            "(MaxGoodputUnderSLO) that need to define what 'good' means before "
            "the goodput metric is computed. None when the recipe has no "
            "per-request SLO opinion (most non-goodput recipes)."
        ),
    )

    @model_validator(mode="after")
    def _check_exactly_one_branch(self) -> SearchRecipeOutput:
        branches = [
            self.adaptive_search is not None,
            self.sweep_parameters is not None,
            self.scenarios is not None,
        ]
        if sum(branches) != 1:
            raise ValueError(
                "SearchRecipeOutput requires exactly one of "
                "'adaptive_search', 'sweep_parameters', or 'scenarios' to be set "
                f"(got adaptive_search={branches[0]}, sweep_parameters={branches[1]}, "
                f"scenarios={branches[2]})."
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
    pareto_axes: ClassVar[ParetoAxesSpec | None] = None
    """Optional 2D axes spec for live and end-of-sweep console Pareto plots.

    Recipes that have a meaningful 2D dominance interpretation set this; all
    others leave it ``None`` and the Pareto-plot path is skipped.
    """
    consumed_magic_lists: ClassVar[frozenset[str]] = frozenset()
    """Magic-list field names this recipe consumes from the user's CLI input.

    A recipe + magic-list combo normally hard-fails (recipes own their sweep
    variables). When a recipe lists a field here, the rejection skips for that
    field and the recipe's expand() reads the user's list out of
    ``ctx.sweep_overrides[<field>]``. Default: empty.
    """

    auto_plot_default: ClassVar[bool]
    """Whether ``aiperf profile`` should auto-invoke ``aiperf plot`` after a
    successful run when the user hasn't passed ``--auto-plot`` / ``--no-auto-plot``.

    Optional: implementers MAY set ``auto_plot_default: ClassVar[bool] = True``
    on recipes whose output is a curve worth visualizing immediately
    (e.g. ``concurrency-ramp``, ``prefill-ttft-curve``, ``decode-itl-curve``).
    Recipes that search for an optimum rather than a curve should leave this
    unset.

    Protocol attribute defaults are NOT inherited by implementers, so this
    declaration intentionally has no runtime default (unlike ``pareto_axes``
    and ``consumed_magic_lists`` above, which DO carry sentinel defaults).
    The read site uses ``getattr(recipe, "auto_plot_default", False)`` so
    external plugin recipes that don't opt in keep working with an implicit
    ``False``.
    """

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the recipe under the given context.

        Args:
            ctx: Benchmark config + SLA targets + sweep overrides snapshot.

        Returns:
            A populated ``SearchRecipeOutput`` with exactly one of
            ``adaptive_search`` / ``sweep_parameters`` / ``scenarios`` set.

        Raises:
            ValueError: If the recipe's required inputs are missing or if the
                benchmark config conflicts with the recipe's assumptions.
        """
        ...


# Resolve the forward ref on `SearchRecipeOutput.adaptive_search`
# (`AdaptiveSearchSweep` is defined in `aiperf.config.sweep`). This is a
# best-effort rebuild: when `_base.py` is loaded as part of the
# `aiperf.config.sweep` import chain, sweep.py is still mid-init and
# AdaptiveSearchSweep is not yet defined -- in that case skip silently;
# the caller (sweep.py finishing its module body) re-runs the rebuild
# from its own bottom-of-file hook below.
try:  # pragma: no cover -- import-order glue
    from aiperf.config.sweep import AdaptiveSearchSweep as _AdaptiveSearchSweep

    SearchRecipeOutput.model_rebuild(
        _types_namespace={"AdaptiveSearchSweep": _AdaptiveSearchSweep}
    )
except Exception:  # noqa: S110 - import-order glue: catch ANY error from a partially-loaded sweep module so we degrade silently. The bottom-of-file hook in `aiperf.config.sweep` re-runs the rebuild once that module finishes initializing; nothing here is recoverable on the failing call, so logging would just spam the import path.
    # Sweep module not yet fully loaded; the rebuild will be re-attempted
    # from the bottom of `aiperf.config.sweep` once it finishes initializing.
    pass
