# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search-recipe expansion helpers for the CLIConfig -> AIPerfConfig converter.

Imported by :mod:`aiperf.config.flags._converter_optionals`; not part of the
flags package public surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.flags import CLIConfig

__all__ = [
    "_RECIPE_DEFINING_FIELDS",
    "_RECIPE_OVERRIDABLE_FIELDS",
    "_RECIPE_TUNABLE_FIELDS",
    "_RECIPE_TUNABLE_FIELD_TO_SWEEP_FIELD",
    "maybe_expand_search_recipe",
]


# --search-* flags that DEFINE the search problem (objective, search space,
# planner choice). Mutually exclusive with --search-recipe: a recipe owns
# these decisions and accepting overrides for them would silently produce a
# recipe-shaped sweep that doesn't match any recipe.
_RECIPE_DEFINING_FIELDS: tuple[str, ...] = (
    "search_space",
    "search_metric",
    "search_stat",
    "search_direction",
    "search_planner",
    "optuna_sampler",
    "optuna_acquisition",
    "optuna_terminator",
    "search_percentile_pooling",
    "bo_constraint_mode",
)

# --search-* flags that TUNE the search budget / determinism without changing
# what's being searched. Allowed alongside --search-recipe: they layer on top
# of the recipe's defaults so users can extend an iteration budget or pin a
# seed without forking the recipe class.
_RECIPE_TUNABLE_FIELDS: tuple[str, ...] = (
    "search_max_iterations",
    "search_initial_points",
    "search_random_seed",
)

# Map sweeping-config field names to ``AdaptiveSearchSweep`` field names for
# the tunable subset above. Used by :func:`_apply_recipe_tunable_overrides`.
_RECIPE_TUNABLE_FIELD_TO_SWEEP_FIELD: tuple[tuple[str, str], ...] = (
    ("search_max_iterations", "max_iterations"),
    ("search_initial_points", "n_initial_points"),
    ("search_random_seed", "random_seed"),
)

# Combined view: every --search-* flag the explicit-flag adaptive-search
# builder knows how to read. Defining + tunable. ``_build_adaptive_search``
# uses this to detect "any --search-* flag set" without --search-recipe.
_RECIPE_OVERRIDABLE_FIELDS: tuple[str, ...] = (
    *_RECIPE_DEFINING_FIELDS,
    *_RECIPE_TUNABLE_FIELDS,
)


def maybe_expand_search_recipe(
    cli: CLIConfig,
    sw: Any,
    cli_set: set[str],
    *,
    benchmark_config: BenchmarkConfig | None,
) -> dict[str, Any] | None:
    """Expand --search-recipe into a converter-shaped dict.

    Returns a dict with one or more of these keys:

    - ``adaptive_search`` (BO recipes): the recipe's expanded
      ``AdaptiveSearchConfig.model_dump()`` with ``sla_filters`` /
      ``recipe_name`` baked in. Lives at the top-level ``sweep`` envelope key.
    - ``sweep_parameters`` (grid recipes): a path -> list-of-values map ready
      to be merged into the top-level ``sweep.parameters`` block by the
      caller (``convert_cli_to_aiperf``).
    - ``scenarios`` (scenario recipes, e.g. pareto-sweep): a list of
      per-scenario dicts merged into the top-level ``sweep.scenarios`` block.
    - ``post_process`` (grid recipes with derived artifacts): a
      ``PostProcessSpec.model_dump()`` for the top-level ``sweep.post_process``.
    - ``sla_filters`` (any recipe): list of ``SLAFilter.model_dump()`` for
      ``sweep.sla_filters`` (grid path) or already baked into
      ``adaptive_search.sla_filters`` (BO path).
    - ``recipe_name`` (grid / scenario paths): the recipe identifier, set so
      the planner and ``search_history.json`` see the recipe's contract.
    - ``slos`` (per-request SLO recipes, e.g. max-goodput-under-slo): metric
      tag -> threshold map consumed by ``GoodRequestCountMetric.set_slos()``.

    Returns ``None`` when no recipe is set. Rejects explicit
    recipe-DEFINING --search-* flags (e.g. ``--search-space``,
    ``--search-metric``); tunable budget/seed flags
    (``--search-max-iterations``, ``--search-initial-points``,
    ``--search-random-seed``) are accepted and override the recipe's
    defaults on the BO branch. The snapshot in ``cli_set`` is the user-set
    field list captured BEFORE this function runs.
    """
    if "search_recipe" not in cli_set or sw.search_recipe is None:
        return None

    if benchmark_config is None:
        # Defensive: convert_cli_to_aiperf is responsible for building the
        # speculative BenchmarkConfig whenever search_recipe is set. If we got
        # here with None, the caller's gating diverged from this branch.
        raise TypeError(
            "expand_search_recipe requires a built BenchmarkConfig when "
            f"search_recipe={sw.search_recipe!r} is set"
        )

    user_search_flags = sorted(cli_set & set(_RECIPE_DEFINING_FIELDS))
    if user_search_flags:
        raise TypeError(
            f"--search-recipe {sw.search_recipe!r} is mutually exclusive with "
            f"recipe-defining --search-* flags {user_search_flags}. "
            "These flags choose the objective / search space / planner -- "
            "the recipe owns those. Tunable knobs "
            "(--search-max-iterations, --search-initial-points, "
            "--search-random-seed) are accepted alongside a recipe."
        )

    output = _invoke_recipe(cli, sw, cli_set, benchmark_config=benchmark_config)
    out_dict = _recipe_output_to_dict(output, sw.search_recipe)
    _apply_recipe_tunable_overrides(sw, cli_set, out_dict, sw.search_recipe)
    return out_dict


def _apply_recipe_tunable_overrides(
    sw: Any,
    cli_set: set[str],
    out_dict: dict[str, Any],
    recipe_name: str,
) -> None:
    """Layer user-set tunable --search-* flags onto a recipe's expanded sweep.

    Called immediately after ``_recipe_output_to_dict``: when the user passes
    e.g. ``--search-max-iterations 100 --search-recipe max-throughput-ttft-sla``,
    we keep the recipe's objective/space/planner but bump the iteration
    budget to 100. Grid recipes (``adaptive_search`` is absent) reject any
    tunable flag because these budget knobs are BO-specific.
    """
    tunable_set = cli_set & set(_RECIPE_TUNABLE_FIELDS)
    if not tunable_set:
        return
    adaptive = out_dict.get("adaptive_search")
    if adaptive is None:
        flag_names = sorted(f"--{name.replace('_', '-')}" for name in tunable_set)
        raise TypeError(
            f"--search-recipe {recipe_name!r} expanded to a grid sweep, but "
            f"tunable --search-* flags {flag_names} are only meaningful for "
            "BO recipes. Drop the flags or pick a BO recipe."
        )
    for sw_field, sweep_field in _RECIPE_TUNABLE_FIELD_TO_SWEEP_FIELD:
        if sw_field in tunable_set and getattr(sw, sw_field) is not None:
            adaptive[sweep_field] = getattr(sw, sw_field)


_SLA_TARGET_FIELDS: tuple[str, ...] = (
    "ttft_sla_ms",
    "itl_sla_ms",
    "tpot_sla_ms",
    "e2e_sla_ms",
    "error_rate_sla",
    "slo_attainment_fraction",
)

_SWEEP_OVERRIDE_FIELDS: tuple[str, ...] = (
    "concurrency_max",
    "concurrency_min",
    "concurrency_steps",
    "degradation_metric_tag",
    "degradation_stat",
    "degradation_threshold",
    "isl_max",
    "isl_min",
    "isl_steps",
    "osl_max",
    "osl_min",
    "osl_steps",
    "search_style",
)


def _build_recipe_sla_targets(sw: Any, cli_set: set[str]) -> dict[str, float]:
    """Pull explicitly-set SLA-target floats off the sweeping config."""
    targets: dict[str, float] = {}
    for key in _SLA_TARGET_FIELDS:
        if key in cli_set and getattr(sw, key) is not None:
            targets[key] = float(getattr(sw, key))
    return targets


def _build_recipe_sweep_overrides(
    cli: CLIConfig, sw: Any, cli_set: set[str]
) -> dict[str, Any]:
    """Pull explicitly-set sweep-override fields plus loadgen magic-lists."""
    overrides: dict[str, Any] = {}
    for key in _SWEEP_OVERRIDE_FIELDS:
        if key in cli_set and getattr(sw, key) is not None:
            overrides[key] = getattr(sw, key)

    # Surface the user's --concurrency value (scalar or list) when the recipe
    # opted into consuming it via consumed_magic_lists. Read from CLIConfig
    # directly (post-flatten), not phases — the magic-list lives on the
    # top-level CLIConfig.concurrency field.
    if "concurrency" in cli.model_fields_set:
        overrides["concurrency"] = cli.concurrency

    # --isl-osl-pairs is recipe-only; surface unconditionally so recipes that
    # need it can read it, and recipes that don't can ignore it.
    if "isl_osl_pairs" in cli_set and sw.isl_osl_pairs is not None:
        overrides["isl_osl_pairs"] = sw.isl_osl_pairs
    return overrides


def _invoke_recipe(
    cli: CLIConfig,
    sw: Any,
    cli_set: set[str],
    *,
    benchmark_config: BenchmarkConfig,
) -> Any:
    """Look up the recipe by name and invoke ``expand()`` against a built ctx.

    Local imports keep the converter layer free of unconditional plugin-system
    imports at module load (matches the late-import pattern used by
    _build_adaptive_search for OptimizationDirection / parse_search_space).
    """
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.plugins import get_class
    from aiperf.search_recipes._base import SearchRecipeContext

    sla_targets = _build_recipe_sla_targets(sw, cli_set)
    sweep_overrides = _build_recipe_sweep_overrides(cli, sw, cli_set)

    recipe_cls = get_class(PluginType.SEARCH_RECIPE, sw.search_recipe)
    recipe = recipe_cls()
    ctx = SearchRecipeContext(
        benchmark_config=benchmark_config,
        sla_targets=sla_targets,
        sweep_overrides=sweep_overrides,
    )
    return recipe.expand(ctx)


def _recipe_output_to_dict(output: Any, recipe_name: str) -> dict[str, Any]:
    """Project a ``SearchRecipeOutput`` to the converter-shaped dict.

    Splits BO (adaptive_search-only) vs grid (sweep_parameters + post_process +
    sla_filters) cases. The recipe is allowed to omit ``sla_filters`` /
    ``recipe_name`` on the BO branch (they default empty); we always set them
    on the returned config so the planner and search_history.json see the
    recipe's contract regardless of recipe-author hygiene.
    """
    out: dict[str, Any] = {}
    if output.adaptive_search is not None:
        expanded = output.adaptive_search.model_copy(
            update={
                "sla_filters": list(output.sla_filters),
                "recipe_name": recipe_name,
            }
        )
        out["adaptive_search"] = expanded.model_dump(mode="json")
    elif output.sweep_parameters is not None:
        out["sweep_parameters"] = dict(output.sweep_parameters)
        if output.sla_filters:
            out["sla_filters"] = [f.model_dump(mode="json") for f in output.sla_filters]
        if output.post_process is not None:
            out["post_process"] = output.post_process.model_dump(mode="json")
        out["recipe_name"] = recipe_name
    elif output.scenarios is not None:
        out["scenarios"] = [dict(s) for s in output.scenarios]
        if output.sla_filters:
            out["sla_filters"] = [f.model_dump(mode="json") for f in output.sla_filters]
        if output.post_process is not None:
            out["post_process"] = output.post_process.model_dump(mode="json")
        out["recipe_name"] = recipe_name
    if output.slos:
        out["slos"] = dict(output.slos)
    return out
