# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional-section builders for the CLIConfig -> AIPerfConfig converter.

Each builder inspects a nested section on the ``CLIConfig`` and, when at
least one field was explicitly set by the cli, returns a dict shaped for
``AIPerfConfig`` consumption. When the section is absent or no fields were
set, the builder returns ``None`` so the top-level converter can omit the
section cleanly rather than emitting empty sub-objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.config.flags._section_fields import (
    ACCURACY_FIELDS,
    SWEEPING_FIELDS,
    TOKENIZER_FIELDS,
)
from aiperf.config.flags._section_fields import SWEEPING_FIELDS as _SWEEPING_FIELD_NAMES
from aiperf.config.flags.recipes import (
    _RECIPE_OVERRIDABLE_FIELDS,
)
from aiperf.config.flags.recipes import (
    maybe_expand_search_recipe as _maybe_expand_search_recipe,
)

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.flags import CLIConfig


def build_tokenizer(cli: CLIConfig) -> dict[str, Any] | None:
    """Build the tokenizer section dict from explicitly-set CLIConfig fields.

    Returns ``None`` when no tokenizer fields were explicitly populated
    (so the converter skips the section entirely).
    """
    tok_set = cli.model_fields_set & TOKENIZER_FIELDS
    if not tok_set:
        return None
    out: dict[str, Any] = {}
    if "tokenizer_name" in tok_set:
        out["name"] = cli.tokenizer_name
    if "tokenizer_revision" in tok_set:
        out["revision"] = cli.tokenizer_revision
    if "trust_remote_code" in tok_set:
        out["trust_remote_code"] = cli.trust_remote_code
    return out or None


def build_accuracy(cli: CLIConfig) -> dict[str, Any] | None:
    """Build the accuracy section dict from explicitly-set CLIConfig fields.

    Returns ``None`` when no accuracy fields were explicitly populated (so the
    converter skips the section entirely).
    """
    acc_set = cli.model_fields_set & ACCURACY_FIELDS
    if not acc_set:
        return None
    # Map flattened CLIConfig attribute -> AIPerfConfig accuracy key. The CLI
    # attrs carry an ``accuracy_`` prefix to keep the flat namespace
    # collision-free; the AIPerfConfig schema strips the prefix.
    field_map = (
        ("accuracy_benchmark", "benchmark"),
        ("accuracy_tasks", "tasks"),
        ("accuracy_n_shots", "n_shots"),
        ("accuracy_enable_cot", "enable_cot"),
        ("accuracy_grader", "grader"),
        ("accuracy_system_prompt", "system_prompt"),
        ("accuracy_verbose", "verbose"),
    )
    out: dict[str, Any] = {}
    for cli_attr, aiperf_key in field_map:
        if cli_attr in acc_set:
            out[aiperf_key] = getattr(cli, cli_attr)
    return out or None


def expand_search_recipe(
    cli: CLIConfig, *, benchmark_config: BenchmarkConfig | None
) -> dict[str, Any] | None:
    """Expand --search-recipe (if set) into a converter-shaped dict.

    Public entry point used by ``convert_cli_to_aiperf`` (which lifts
    ``sweep_parameters`` to the top-level ``sweep`` block) and by
    :func:`build_multi_run` (which routes ``adaptive_search`` /
    ``post_process`` / ``sla_filters`` onto the top-level sweep block).

    ``benchmark_config`` may be ``None`` only when no recipe is set; the
    caller in ``convert_cli_to_aiperf`` skips the speculative BenchmarkConfig
    build for magic-list-only invocations (where list-shaped phase fields
    haven't been promoted to the sweep block yet). Recipes always require a
    built ``BenchmarkConfig``; when ``search_recipe`` is set this function
    raises ``TypeError`` if ``benchmark_config`` is ``None``.

    Returns ``None`` when no recipe is set; otherwise see
    :func:`_maybe_expand_search_recipe` for the dict shape.
    """
    sweeping_set = cli.model_fields_set & SWEEPING_FIELDS
    if not sweeping_set:
        return None
    return _maybe_expand_search_recipe(
        cli, cli, sweeping_set, benchmark_config=benchmark_config
    )


def resolve_auto_plot(cli: CLIConfig) -> tuple[bool, bool]:
    """Resolve the ``(auto_plot, plot_required)`` pair for the artifacts block.

    ``auto_plot`` is tri-state on the CLIConfig input (None = defer to recipe).
    The recipe's ``auto_plot_default`` (read via ``getattr`` so external
    plugin recipes without the attribute keep working) supplies the value
    when the user did not pass an explicit override; with no recipe and no
    explicit flag the answer is ``False``.

    ``plot_required`` passes through unchanged. The caller is expected to
    write both into the ``artifacts`` dict.
    """
    explicit_auto_plot: bool | None = cli.auto_plot
    plot_required: bool = bool(cli.plot_required)

    recipe = _resolve_recipe_instance(cli)
    recipe_default = (
        bool(getattr(recipe, "auto_plot_default", False)) if recipe else False
    )

    resolved = explicit_auto_plot if explicit_auto_plot is not None else recipe_default
    return bool(resolved), plot_required


def _resolve_recipe_instance(cli: CLIConfig) -> Any | None:
    """Look up the active search recipe by name and instantiate it.

    Returns ``None`` when no recipe is set or the lookup fails (treating an
    unknown recipe name the same as "no recipe" for auto-plot resolution;
    the actual recipe-name validation happens later in
    :func:`_invoke_recipe`, which is the single source of truth for recipe
    errors). Local imports mirror the late-import pattern in
    ``recipes._invoke_recipe`` so the converter layer stays free of
    plugin-system imports at module load.
    """
    sw = cli
    if "search_recipe" not in sw.model_fields_set or sw.search_recipe is None:
        return None
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.plugins import get_class
    from aiperf.plugin.types import PluginError

    try:
        recipe_cls = get_class(PluginType.SEARCH_RECIPE, sw.search_recipe)
    except (PluginError, KeyError):
        return None
    try:
        return recipe_cls()
    except (TypeError, ValueError):
        return None


def build_multi_run(
    cli: CLIConfig,
    *,
    recipe_output: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the multi-run section dict from explicitly-set CLIConfig loadgen fields.

    When --search-* flags are present, builds a typed ``AdaptiveSearchSweep``
    and lifts it onto the top-level ``sweep`` block (NOT onto multi_run —
    schema-2.0 moved adaptive_search out of MultiRunConfig).

    When --search-recipe is set, the named recipe expands directly into
    either:

    - a populated ``AdaptiveSearchSweep`` (carrying ``sla_filters`` and
      ``recipe_name``) lifted onto the top-level sweep block (BO recipes); or
    - a ``sweep_parameters`` dict (grid recipes) -- handled by
      :func:`expand_search_recipe` and lifted onto the top-level sweep block by
      the top-level converter, with ``post_process`` and ``sla_filters`` carried
      on that sweep block for ``aggregate_sweep_and_export`` to consume.

    Hard-fails if --search-space is set without the required companion flags
    (--search-metric, --search-direction, --search-max-iterations).

    ``recipe_output`` is the cached output of :func:`expand_search_recipe`;
    callers compute it once at the top of ``convert_cli_to_aiperf`` so the
    recipe's ``expand()`` doesn't run twice. ``None`` means "no recipe";
    callers that don't pre-compute pass ``None`` and we recompute lazily.
    """
    sw = cli
    if not (sw.model_fields_set & _SWEEPING_FIELD_NAMES):
        return None
    if recipe_output is None and "search_recipe" in sw.model_fields_set:
        raise ValueError("recipe_output must be precomputed before build_multi_run")
    mapping = {
        "num_profile_runs": "num_runs",
        "profile_run_cooldown_seconds": "cooldown_seconds",
        "confidence_level": "confidence_level",
        "profile_run_disable_warmup_after_first": "disable_warmup_after_first",
        "set_consistent_seed": "set_consistent_seed",
        "vary_seed_per_trial": "vary_seed_per_trial",
    }
    # Schema-2.0 nests convergence fields under ``multi_run.convergence``
    # (a ConvergenceConfig sub-object) instead of carrying them flat on
    # MultiRunConfig. Presence of ``convergence_metric`` is the trigger:
    # it's a required field on ConvergenceConfig, and CLIConfig defaults it to
    # ``None`` (== adaptive disabled).
    convergence_mapping = {
        "convergence_metric": "metric",
        "convergence_mode": "mode",
        "convergence_threshold": "threshold",
        "convergence_stat": "stat",
    }
    # NOTE: ``parameter_sweep_*`` flags (mode, same_seed, cooldown_seconds)
    # were historically routed onto ``multi_run`` here. Schema-2.0 moved
    # them to the per-sweep config (GridSweep / ScenarioSweep /
    # AdaptiveSearchSweep). converter.py stamps them via
    # ``_apply_parameter_sweep_meta_to_sweep`` after sweep building.
    out: dict[str, Any] = {}
    for field, key in mapping.items():
        if field in sw.model_fields_set:
            out[key] = getattr(sw, field)
    if (
        "convergence_metric" in sw.model_fields_set
        and sw.convergence_metric is not None
    ):
        convergence: dict[str, Any] = {}
        for field, key in convergence_mapping.items():
            if field in sw.model_fields_set:
                convergence[key] = getattr(sw, field)
        out["convergence"] = convergence
    adaptive_search = _resolve_adaptive_search(sw, recipe_output)
    if adaptive_search is not None:
        # adaptive_search now lives on the top-level ``sweep`` envelope key
        # (carried via build_sweep), not on MultiRunConfig. Keep the
        # convergence-rejection invariant here because this is still the
        # single place that knows BO is active.
        _reject_search_plus_convergence(sw)
    return out or None


def build_sweep(
    cli: CLIConfig,
    *,
    recipe_output: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the top-level ``sweep`` envelope dict from --search-* flags.

    Returns an ``AdaptiveSearchSweep``-shaped dict (``type='adaptive_search'``)
    when --search-* flags are set, with ``--search-sla`` filters folded in.
    For grid recipes, returns the sweep metadata (``sla_filters`` /
    ``post_process``) and leaves ``parameters`` for
    ``_apply_recipe_sweep_parameters`` in the top-level converter.
    """
    sw = cli
    if not (sw.model_fields_set & _SWEEPING_FIELD_NAMES):
        return None
    if recipe_output is None and "search_recipe" in sw.model_fields_set:
        raise ValueError("recipe_output must be precomputed before build_sweep")
    extra_sla_dumps = _parse_search_sla_flags(sw)
    adaptive_search = _resolve_adaptive_search(sw, recipe_output)
    if adaptive_search is None:
        if recipe_output is None or not recipe_output.get("sweep_parameters"):
            return None
        sweep: dict[str, Any] = {"type": "grid"}
        recipe_sla_dumps = list(recipe_output.get("sla_filters") or [])
        if recipe_sla_dumps or extra_sla_dumps:
            sweep["sla_filters"] = recipe_sla_dumps + extra_sla_dumps
        if recipe_output.get("post_process") is not None:
            sweep["post_process"] = recipe_output["post_process"]
        return sweep
    if extra_sla_dumps:
        existing = list(adaptive_search.get("sla_filters") or [])
        adaptive_search["sla_filters"] = existing + extra_sla_dumps
    return adaptive_search


def _parse_search_sla_flags(sw: Any) -> list[dict[str, Any]]:
    """Parse `--search-sla` flag values into model-dumped SLAFilter dicts.

    Returns an empty list when the flag is unset or empty. Each parser error
    propagates as ``TypeError`` from :func:`parse_sla_filter`, naming the
    offending flag value.
    """
    if "search_sla" not in sw.model_fields_set or not sw.search_sla:
        return []
    from aiperf.orchestrator.search_planner.parsing import parse_sla_filter

    return [parse_sla_filter(s).model_dump(mode="json") for s in sw.search_sla]


def _resolve_adaptive_search(
    sw: Any, recipe_output: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Pick the adaptive_search source: recipe (BO) or explicit --search-* flags.

    Grid recipes have ``recipe_output["adaptive_search"] is None`` -- the
    function returns ``None`` so build_multi_run skips the adaptive_search
    branch entirely (sweep parameters flow through a different field).
    """
    if recipe_output is not None and recipe_output.get("adaptive_search") is not None:
        return recipe_output["adaptive_search"]
    if recipe_output is not None:
        return None
    return _build_adaptive_search(sw)


def _reject_search_plus_convergence(sw: Any) -> None:
    """Hard-fail when both --search-* (BO) and --convergence-metric are set.

    --search-* and --convergence-metric (trial-level adaptive early-stop) are
    conceptually orthogonal but their interaction wasn't designed: the BO
    orchestrator path silently ignores convergence_metric. Reject explicitly
    so users don't think trial-level convergence is doing anything during a
    BO run. Documented in docs/sweeping/bayesian-optimization.md.
    """
    if (
        "convergence_metric" in sw.model_fields_set
        and sw.convergence_metric is not None
    ):
        raise TypeError(
            "--search-* (Bayesian Optimization) is mutually exclusive with "
            "--convergence-metric (trial-level adaptive early-stop). The two "
            "operate at different levels (outer-loop vs. inner-trial) and "
            "their composition is undefined. Drop one of them."
        )


# (search_set field, AdaptiveSearchSweep output key) for the optional knobs
# that pass through unchanged. Required fields (search_space / metric /
# direction / max_iterations) feed structural sub-dicts so they're handled
# inline by :func:`_build_adaptive_search`.
_SWEEP_OPTIONAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("search_initial_points", "n_initial_points"),
    ("search_random_seed", "random_seed"),
    ("search_planner", "planner"),
    ("optuna_sampler", "optuna_sampler"),
    ("optuna_acquisition", "optuna_acquisition"),
    ("optuna_terminator", "optuna_terminator"),
    ("bo_constraint_mode", "constraint_mode"),
)


# Acquisition modes that require the BoTorch sampler. Validated at the
# CLIConfig -> AIPerfConfig boundary; bottom-line cross-flag check Optuna
# can't enforce because the sampler-vs-acquisition pairing is AIPerf's
# wrapper choice.
_BOTORCH_ACQUISITIONS: frozenset[str] = frozenset(
    {"logei", "qlogei", "qnei", "qlognei", "qehvi", "qnehvi", "qlognehvi"}
)


def _check_search_required_companions(search_set: set[str]) -> None:
    """Raise ``TypeError`` when --search-space is used without its companions."""
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


def _build_adaptive_search(sw: Any) -> dict[str, Any] | None:
    """Parse --search-* flags into an ``AdaptiveSearchSweep``-shaped dict.

    Returns a dict with ``type='adaptive_search'`` and the search-space /
    objective / iteration knobs ready to feed into ``AIPerfConfig`` as the
    top-level ``sweep`` envelope key. Returns ``None`` when no --search-*
    flags were set. Raises ``TypeError`` when the flag combination is
    invalid (search-space without companions, or other --search-* flags
    without --search-space).
    """
    search_set = {f for f in _RECIPE_OVERRIDABLE_FIELDS if f in sw.model_fields_set}
    if "search_space" not in search_set:
        if search_set:
            raise TypeError(
                f"--search-* flags {sorted(search_set)} require --search-space."
            )
        return None
    _check_search_required_companions(search_set)
    _validate_acquisition_compat(sw)
    _validate_pooling_compat(sw)
    # Done here (not later in build_benchmark_plan) so AdaptiveSearchSweep
    # validation catches structural errors early at the CLIConfig -> AIPerfConfig boundary.
    from aiperf.orchestrator.search_planner.parsing import parse_search_space

    space = parse_search_space(list(sw.search_space))
    objective: dict[str, Any] = {
        "metric": sw.search_metric,
        "direction": sw.search_direction,
    }
    if "search_stat" in search_set and sw.search_stat is not None:
        objective["stat"] = sw.search_stat

    out: dict[str, Any] = {
        "type": "adaptive_search",
        "search_space": [dim.model_dump(mode="json") for dim in space],
        "objectives": [objective],
        "max_iterations": sw.search_max_iterations,
    }
    for src_field, out_key in _SWEEP_OPTIONAL_FIELDS:
        if src_field in search_set and getattr(sw, src_field) is not None:
            out[out_key] = getattr(sw, src_field)
    if (
        "search_percentile_pooling" in sw.model_fields_set
        and sw.search_percentile_pooling is not None
    ):
        out["objective_pooling"] = sw.search_percentile_pooling
    return out


def _validate_acquisition_compat(sw: Any) -> None:
    """Reject incompatible --optuna-acquisition / --optuna-sampler combinations.

    Acquisition overrides require the BoTorch sampler; rejecting at the
    CLIConfig -> AIPerfConfig boundary surfaces the misconfig before the
    planner builds. The acquisition family vs. ``len(objectives)`` pairing
    is enforced by a cross-field
    validator on ``AdaptiveSearchSweep``.
    """
    if "optuna_acquisition" not in sw.model_fields_set:
        return
    acq = sw.optuna_acquisition
    if acq is None:
        return
    if acq not in _BOTORCH_ACQUISITIONS:
        # Defensive: schema Literal narrows already; future additions land here.
        raise TypeError(f"--optuna-acquisition {acq!r} is not recognized")
    sampler = sw.optuna_sampler if "optuna_sampler" in sw.model_fields_set else None
    if sampler is not None and sampler != "botorch":
        raise TypeError(
            f"--optuna-acquisition {acq!r} requires --optuna-sampler botorch; "
            f"got --optuna-sampler {sampler!r}. Acquisition overrides are only "
            "consulted on the BoTorch path."
        )


def _validate_pooling_compat(sw: Any) -> None:
    """Reject --search-percentile-pooling pooled when --search-stat is avg.

    Pooled aggregation only differs from mean-aggregation when the underlying
    statistic is a percentile -- pooling means is a tautology. Reject so the
    user gets a clear error instead of a silent no-op.
    """
    if "search_percentile_pooling" not in sw.model_fields_set:
        return
    pooling = sw.search_percentile_pooling
    if pooling != "pooled":
        return
    stat = sw.search_stat if "search_stat" in sw.model_fields_set else None
    if stat is not None and stat == "avg":
        raise TypeError(
            "--search-percentile-pooling pooled requires --search-stat to be "
            "a percentile (p50/p90/p95/p99); pooling means is a no-op. Either "
            "switch --search-stat to a percentile or drop "
            "--search-percentile-pooling."
        )
