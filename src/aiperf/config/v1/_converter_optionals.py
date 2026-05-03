# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional-section builders for the v1 -> v2 UserConfig converter.

Each builder inspects a nested section on the v1 ``UserConfig`` and, when at
least one field was explicitly set by the user, returns a dict shaped for
``AIPerfConfig`` consumption. When the section is absent or no fields were
set, the builder returns ``None`` so the top-level converter can omit the
section cleanly rather than emitting empty sub-objects.

Mirrors the section-builder logic in ``aiperf.config._cli_sections`` for the
flat CLIModel input, rerouted to read from nested ``UserConfig`` sub-models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.v1 import UserConfig


def build_tokenizer(user: UserConfig) -> dict[str, Any] | None:
    """Build the tokenizer section dict from explicitly-set v1 fields.

    Returns ``None`` when ``user.tokenizer`` is unset or has no explicitly
    populated fields (so the converter skips the section entirely).
    """
    tok = user.tokenizer
    if tok is None or not tok.model_fields_set:
        return None
    out: dict[str, Any] = {}
    if "name" in tok.model_fields_set:
        out["name"] = tok.name
    if "revision" in tok.model_fields_set:
        out["revision"] = tok.revision
    if "trust_remote_code" in tok.model_fields_set:
        out["trust_remote_code"] = tok.trust_remote_code
    return out or None


def build_accuracy(user: UserConfig) -> dict[str, Any] | None:
    """Build the accuracy section dict from explicitly-set v1 fields.

    Returns ``None`` when ``user.accuracy`` is unset or has no explicitly
    populated fields.
    """
    acc = user.accuracy
    if acc is None or not acc.model_fields_set:
        return None
    keys = (
        "benchmark",
        "tasks",
        "n_shots",
        "enable_cot",
        "grader",
        "system_prompt",
        "verbose",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key in acc.model_fields_set:
            out[key] = getattr(acc, key)
    return out or None


def expand_search_recipe(user: UserConfig) -> dict[str, Any] | None:
    """Expand --search-recipe (if set) into a converter-shaped dict.

    Public entry point used by ``convert_user_to_aiperf`` (which lifts
    ``sweep_variables`` to the top-level ``sweep`` block) and by
    :func:`build_multi_run` (which routes ``adaptive_search`` /
    ``post_process`` / ``sla_filters`` into ``MultiRunConfig``).

    Returns ``None`` when no recipe is set; otherwise see
    :func:`_maybe_expand_search_recipe` for the dict shape.
    """
    lg = user.loadgen
    if lg is None or not lg.model_fields_set:
        return None
    return _maybe_expand_search_recipe(user, lg, set(lg.model_fields_set))


def build_multi_run(
    user: UserConfig,
    *,
    recipe_output: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the multi-run section dict from explicitly-set v1 loadgen fields.

    When --search-* flags are present, builds a typed AdaptiveSearchConfig and
    emits its model_dump() as `out["adaptive_search"]`. MultiRunConfig has
    `extra="forbid"` so the typed field is the only legal carrier.

    When --search-recipe is set, the named recipe expands directly into
    either:

    - a populated ``AdaptiveSearchConfig`` (carrying ``sla_filters`` and
      ``recipe_name``) emitted at ``out["adaptive_search"]`` (BO recipes); or
    - a ``sweep_variables`` dict (grid recipes) -- handled by
      :func:`expand_search_recipe` and lifted by the top-level converter,
      with ``post_process`` and ``sla_filters`` threaded through
      ``out["post_process"]`` / ``out["sla_filters"]`` for
      ``aggregate_sweep_and_export`` to consume.

    Hard-fails if --search-space is set without the required companion flags
    (--search-metric, --search-direction, --search-max-iterations).

    ``recipe_output`` is the cached output of :func:`expand_search_recipe`;
    callers compute it once at the top of ``convert_user_to_aiperf`` so the
    recipe's ``expand()`` doesn't run twice. ``None`` means "no recipe";
    callers that don't pre-compute pass ``None`` and we recompute lazily.
    """
    lg = user.loadgen
    if lg is None or not lg.model_fields_set:
        return None
    if recipe_output is None:
        recipe_output = expand_search_recipe(user)
    mapping = {
        "num_profile_runs": "num_runs",
        "profile_run_cooldown_seconds": "cooldown_seconds",
        "confidence_level": "confidence_level",
        "profile_run_disable_warmup_after_first": "disable_warmup_after_first",
        "set_consistent_seed": "set_consistent_seed",
        "convergence_metric": "convergence_metric",
        "convergence_mode": "convergence_mode",
        "convergence_threshold": "convergence_threshold",
        "convergence_stat": "convergence_stat",
        "parameter_sweep_cooldown_seconds": "parameter_sweep_cooldown_seconds",
        "parameter_sweep_same_seed": "parameter_sweep_same_seed",
        "parameter_sweep_mode": "mode",
    }
    out: dict[str, Any] = {}
    for field, key in mapping.items():
        if field in lg.model_fields_set:
            out[key] = getattr(lg, field)
    adaptive_search = _resolve_adaptive_search(lg, recipe_output)
    if adaptive_search is not None:
        out["adaptive_search"] = adaptive_search
        _reject_search_plus_convergence(lg)
    if recipe_output is not None and recipe_output.get("post_process") is not None:
        out["post_process"] = recipe_output["post_process"]
    if recipe_output is not None and recipe_output.get("sla_filters"):
        out["sla_filters"] = recipe_output["sla_filters"]
    return out or None


def _resolve_adaptive_search(
    lg: Any, recipe_output: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Pick the adaptive_search source: recipe (BO) or explicit --search-* flags.

    Grid recipes have ``recipe_output["adaptive_search"] is None`` -- the
    function returns ``None`` so build_multi_run skips the adaptive_search
    branch entirely (sweep variables flow through a different field).
    """
    if recipe_output is not None and recipe_output.get("adaptive_search") is not None:
        return recipe_output["adaptive_search"]
    if recipe_output is not None:
        return None
    return _build_adaptive_search(lg)


def _reject_search_plus_convergence(lg: Any) -> None:
    """Hard-fail when both --search-* (BO) and --convergence-metric are set.

    --search-* and --convergence-metric (trial-level adaptive early-stop) are
    conceptually orthogonal but their interaction wasn't designed: the BO
    orchestrator path silently ignores convergence_metric. Reject explicitly
    so users don't think trial-level convergence is doing anything during a
    BO run. Documented in docs/sweeping/bayesian-optimization.md.
    """
    if (
        "convergence_metric" in lg.model_fields_set
        and lg.convergence_metric is not None
    ):
        raise TypeError(
            "--search-* (Bayesian Optimization) is mutually exclusive with "
            "--convergence-metric (trial-level adaptive early-stop). The two "
            "operate at different levels (outer-loop vs. inner-trial) and "
            "their composition is undefined. Drop one of them."
        )


_RECIPE_OVERRIDABLE_FIELDS: tuple[str, ...] = (
    "search_space",
    "search_metric",
    "search_stat",
    "search_direction",
    "search_max_iterations",
    "search_initial_points",
    "search_random_seed",
    "search_planner",
)


def _maybe_expand_search_recipe(
    user: UserConfig, lg: Any, user_set: set[str]
) -> dict[str, Any] | None:
    """Expand --search-recipe into a converter-shaped dict.

    Returns a dict with one or more of these keys:

    - ``adaptive_search`` (BO recipes): the recipe's expanded
      ``AdaptiveSearchConfig.model_dump()`` with ``sla_filters`` /
      ``recipe_name`` baked in. Lives at ``MultiRunConfig.adaptive_search``.
    - ``sweep_variables`` (grid recipes): a path -> list-of-values map ready
      to be merged into the top-level ``sweep.variables`` block by the
      caller (``convert_user_to_aiperf``).
    - ``post_process`` (grid recipes with derived artifacts): a
      ``PostProcessSpec.model_dump()`` for ``MultiRunConfig.post_process``.
    - ``sla_filters`` (any recipe): list of ``SLAFilter.model_dump()`` for
      ``MultiRunConfig.sla_filters`` (grid path) or already baked into
      ``adaptive_search.sla_filters`` (BO path).

    Returns ``None`` when no recipe is set. Rejects explicit --search-* +
    --search-recipe combinations; the snapshot in ``user_set`` is the
    user-set field list captured BEFORE this function runs.
    """
    if "search_recipe" not in user_set or lg.search_recipe is None:
        return None

    user_search_flags = sorted(user_set & set(_RECIPE_OVERRIDABLE_FIELDS))
    if user_search_flags:
        raise TypeError(
            f"--search-recipe {lg.search_recipe!r} is mutually exclusive with "
            f"explicit --search-* flags {user_search_flags}. "
            "Either drop the explicit flags and let the recipe expand them, "
            "or drop --search-recipe and configure --search-* by hand."
        )

    output = _invoke_recipe(user, lg, user_set)
    return _recipe_output_to_dict(output, lg.search_recipe)


def _invoke_recipe(user: UserConfig, lg: Any, user_set: set[str]) -> Any:
    """Look up the recipe by name and invoke ``expand()`` against a built ctx.

    Local imports keep the v1 layer free of unconditional plugin-system
    imports at module load (matches the late-import pattern used by
    _build_adaptive_search for OptimizationDirection / parse_search_space).
    """
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.plugins import get_class
    from aiperf.search_recipes._base import SearchRecipeContext

    sla_targets: dict[str, float] = {}
    if "ttft_sla_ms" in user_set and lg.ttft_sla_ms is not None:
        sla_targets["ttft_sla_ms"] = float(lg.ttft_sla_ms)
    if "itl_sla_ms" in user_set and lg.itl_sla_ms is not None:
        sla_targets["itl_sla_ms"] = float(lg.itl_sla_ms)

    sweep_overrides: dict[str, Any] = {}
    for key in ("degradation_threshold", "isl_min", "isl_max"):
        if key in user_set and getattr(lg, key) is not None:
            sweep_overrides[key] = getattr(lg, key)

    recipe_cls = get_class(PluginType.SEARCH_RECIPE, lg.search_recipe)
    recipe = recipe_cls()
    ctx = SearchRecipeContext(
        user_config=user,
        sla_targets=sla_targets,
        sweep_overrides=sweep_overrides,
    )
    return recipe.expand(ctx)


def _recipe_output_to_dict(output: Any, recipe_name: str) -> dict[str, Any]:
    """Project a ``SearchRecipeOutput`` to the converter-shaped dict.

    Splits BO (adaptive_search-only) vs grid (sweep_variables + post_process +
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
    elif output.sweep_variables is not None:
        out["sweep_variables"] = dict(output.sweep_variables)
        if output.sla_filters:
            out["sla_filters"] = [f.model_dump(mode="json") for f in output.sla_filters]
        if output.post_process is not None:
            out["post_process"] = output.post_process.model_dump(mode="json")
    return out


def _build_adaptive_search(lg: Any) -> dict[str, Any] | None:
    """Parse --search-* flags into a model-dumped AdaptiveSearchConfig dict.

    Returns ``None`` when no --search-* flags were set. Raises ``TypeError``
    when the flag combination is invalid (search-space without companions, or
    other --search-* flags without --search-space).
    """
    search_set = {f for f in _RECIPE_OVERRIDABLE_FIELDS if f in lg.model_fields_set}
    if "search_space" not in search_set:
        if search_set:
            raise TypeError(
                f"--search-* flags {sorted(search_set)} require --search-space."
            )
        return None
    for required, flag in (
        ("search_metric", "--search-metric"),
        ("search_direction", "--search-direction"),
        ("search_max_iterations", "--search-max-iterations"),
    ):
        if required not in search_set:
            raise TypeError(
                f"--search-space requires {flag} (companion flag missing). "
                "See docs/sweeping/bayesian-optimization.md for examples."
            )
    # Done here (not later in build_benchmark_plan) so MultiRunConfig
    # validation catches structural errors early at the v1->v2 boundary.
    from aiperf.config.adaptive_search import AdaptiveSearchConfig
    from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
    from aiperf.orchestrator.search_planner.parsing import parse_search_space

    ol_kwargs: dict[str, Any] = dict(
        algorithm="bayes",
        search_space=parse_search_space(lg.search_space),
        objective_metric=lg.search_metric,
        objective_stat=lg.search_stat or "avg",
        objective_direction=OptimizationDirection(lg.search_direction),
        max_iterations=lg.search_max_iterations,
    )
    if "search_initial_points" in search_set and lg.search_initial_points is not None:
        ol_kwargs["n_initial_points"] = lg.search_initial_points
    if "search_random_seed" in search_set and lg.search_random_seed is not None:
        ol_kwargs["random_seed"] = lg.search_random_seed
    if "search_planner" in search_set and lg.search_planner is not None:
        ol_kwargs["planner"] = lg.search_planner
    return AdaptiveSearchConfig(**ol_kwargs).model_dump(mode="json")
