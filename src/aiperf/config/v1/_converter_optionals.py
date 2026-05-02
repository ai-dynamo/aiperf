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


def build_multi_run(user: UserConfig) -> dict[str, Any] | None:
    """Build the multi-run section dict from explicitly-set v1 loadgen fields.

    When --search-* flags are present, builds a typed AdaptiveSearchConfig and
    emits its model_dump() as `out["adaptive_search"]`. MultiRunConfig has
    `extra="forbid"` so the typed field is the only legal carrier.

    When --search-recipe is set, the named recipe expands directly into a
    populated ``AdaptiveSearchConfig`` (carrying ``sla_filters`` and
    ``recipe_name`` set to the recipe's name) which is emitted at
    ``out["adaptive_search"]``. The recipe path bypasses
    ``_build_adaptive_search`` so SLA filters and recipe metadata flow through
    intact; explicit --search-* flags continue to use the by-hand path.

    Hard-fails if --search-space is set without the required companion flags
    (--search-metric, --search-direction, --search-max-iterations).
    """
    lg = user.loadgen
    if lg is None or not lg.model_fields_set:
        return None
    user_set = set(lg.model_fields_set)
    recipe_adaptive_search = _maybe_expand_search_recipe(user, lg, user_set)
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
    if recipe_adaptive_search is not None:
        adaptive_search: dict[str, Any] | None = recipe_adaptive_search
    else:
        adaptive_search = _build_adaptive_search(lg)
    if adaptive_search is not None:
        out["adaptive_search"] = adaptive_search
        # --search-* and --convergence-metric (trial-level adaptive early-stop)
        # are conceptually orthogonal but their interaction wasn't designed:
        # the BO orchestrator path silently ignores convergence_metric. Reject
        # explicitly so users don't think trial-level convergence is doing
        # anything during a BO run. Documented in docs/sweeping/bayesian-optimization.md.
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
    return out or None


_RECIPE_OVERRIDABLE_FIELDS: tuple[str, ...] = (
    "search_space",
    "search_metric",
    "search_stat",
    "search_direction",
    "search_max_iterations",
    "search_initial_points",
    "search_random_seed",
)


def _maybe_expand_search_recipe(
    user: UserConfig, lg: Any, user_set: set[str]
) -> dict[str, Any] | None:
    """Expand --search-recipe into a populated AdaptiveSearchConfig dict.

    Returns the recipe's expanded ``AdaptiveSearchConfig.model_dump()`` (with
    ``sla_filters`` and ``recipe_name`` populated), or ``None`` when no recipe
    is set. The caller writes the dict to ``out["adaptive_search"]`` directly,
    bypassing ``_build_adaptive_search`` so SLA filters and recipe metadata
    survive end-to-end.

    Rejects explicit --search-* + --search-recipe combinations; the snapshot in
    ``user_set`` is the user-set field list captured BEFORE this function runs.
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

    # Local imports keep the v1 layer free of unconditional plugin-system
    # imports at module load (matches the late-import pattern used by
    # _build_adaptive_search for OptimizationDirection / parse_search_space).
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.plugins import get_class
    from aiperf.search_recipes._base import SearchRecipeContext

    sla_targets: dict[str, float] = {}
    if "ttft_sla_ms" in user_set and lg.ttft_sla_ms is not None:
        sla_targets["ttft_sla_ms"] = float(lg.ttft_sla_ms)

    recipe_cls = get_class(PluginType.SEARCH_RECIPE, lg.search_recipe)
    recipe = recipe_cls()
    ctx = SearchRecipeContext(user_config=user, sla_targets=sla_targets)
    output = recipe.expand(ctx)

    if output.adaptive_search is None:
        # Phase 1 only ships a BO recipe; grid recipes land in Phase 3 and will
        # write into a different downstream path (sweep.variables) here.
        raise TypeError(
            f"--search-recipe {lg.search_recipe!r} produced a sweep_variables "
            "output, which the v1->v2 converter does not yet support. Phase 3 "
            "wires grid recipes through to the sweep.variables block."
        )

    # Re-emit the AdaptiveSearchConfig with sla_filters + recipe_name baked in.
    # The recipe is allowed to omit these (they default empty); we always set
    # them on the returned config so the planner and search_history.json see
    # the recipe's contract regardless of recipe-author hygiene.
    expanded = output.adaptive_search.model_copy(
        update={
            "sla_filters": list(output.sla_filters),
            "recipe_name": lg.search_recipe,
        }
    )
    return expanded.model_dump(mode="json")


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
    return AdaptiveSearchConfig(**ol_kwargs).model_dump(mode="json")
