# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``CLIConfig`` -> ``AIPerfConfig`` entrypoint.

Composes the seven section-builders that live alongside this module
(``_converter_endpoint``, ``_converter_profiling``, ``_converter_warmup``,
``_converter_dataset``, ``_converter_runtime``, ``_converter_telemetry``,
``_converter_optionals``) into a single nested dict, then validates it
through ``AIPerfConfig``.

The converter is the only module outside ``cli_commands/`` allowed to read
``CLIConfig`` attributes; downstream code consumes the validated
``AIPerfConfig``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags._converter_endpoint import build_endpoint, build_models
from aiperf.config.flags._converter_optionals import (
    build_accuracy,
    build_multi_run,
    build_sweep,
    build_tokenizer,
    expand_search_recipe,
    resolve_auto_plot,
)
from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags._converter_runtime import (
    build_artifacts,
    build_logging_runtime,
)
from aiperf.config.flags._converter_telemetry import (
    build_gpu_telemetry,
    build_mlflow,
    build_otel,
    build_server_metrics,
)
from aiperf.config.flags._converter_warmup import build_warmup
from aiperf.config.sweep import MAGIC_LIST_FIELDS

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig
    from aiperf.config.flags import CLIConfig


# Envelope keys that stay at top level. Everything else moves under `benchmark:`.
_ENVELOPE_KEYS = {
    "sweep",
    "multi_run",
    "variables",
    "random_seed",
    "no_sweep_table",
    "benchmark",
}


def _wrap_under_envelope(v2_dict: dict[str, Any]) -> dict[str, Any]:
    """Partition a flat-shaped AIPerfConfig dict into envelope shape.

    Envelope keys ({sweep, multi_run, variables, random_seed, no_sweep_table,
    benchmark}) stay at top level. Everything else is moved under `benchmark:`.
    Idempotent: a dict that is already envelope-shaped passes through.
    """
    body = {k: v2_dict[k] for k in list(v2_dict.keys()) if k not in _ENVELOPE_KEYS}
    if not body:
        return v2_dict
    for k in body:
        del v2_dict[k]
    if "benchmark" in v2_dict and isinstance(v2_dict["benchmark"], dict):
        v2_dict["benchmark"].update(body)
    else:
        v2_dict["benchmark"] = body
    return v2_dict


def _init_random_seed(cli: CLIConfig) -> None:
    from aiperf.common import random_generator as rng
    from aiperf.common.exceptions import InvalidStateError

    seed = cli.random_seed
    if seed is None:
        return
    with contextlib.suppress(InvalidStateError):
        rng.init(seed)


def _assemble_optional(
    nested: dict[str, Any],
    cli: CLIConfig,
    *,
    recipe_output: dict[str, Any] | None,
) -> None:
    if tok := build_tokenizer(cli):
        nested["tokenizer"] = tok
    if acc := build_accuracy(cli):
        nested["accuracy"] = acc
    if mr := build_multi_run(cli, recipe_output=recipe_output):
        nested["multi_run"] = mr
    if sweep := build_sweep(cli, recipe_output=recipe_output):
        # adaptive_search lives on the top-level ``sweep`` envelope key;
        # MultiRunConfig only carries trial mechanics now.
        nested["sweep"] = sweep
    slos: dict[str, float] = {}
    if "random_seed" in cli.model_fields_set:
        nested["random_seed"] = cli.random_seed
    if "no_sweep_table" in cli.model_fields_set:
        nested["no_sweep_table"] = cli.no_sweep_table
    if cli.goodput:
        slos.update(cli.goodput)
    # Recipe-emitted slos win on key collision: a goodput-style recipe owns the
    # SLO contract for its run, so a stray --goodput on the same CLI invocation
    # shouldn't silently override the recipe's TTFT/TPOT/E2E thresholds.
    if recipe_output is not None and recipe_output.get("slos"):
        slos.update(recipe_output["slos"])
    if slos:
        nested["slos"] = slos


def _apply_recipe_sweep_parameters(
    nested: dict[str, Any],
    recipe_output: dict[str, Any] | None,
    cli: CLIConfig,
) -> None:
    """Lift a grid-recipe's ``sweep_parameters`` onto the top-level ``sweep`` block.

    Mutually exclusive with magic-list flags: a recipe owns sweep parameters on
    the grid path, so the user passing ``--concurrency 10,20,30`` alongside a
    grid ``--search-recipe`` is ambiguous (which list wins?). We defer the
    decision to the user by hard-failing here with a clear message. The
    detection runs against the ``CLIConfig`` (not the assembled phase
    dicts) so the rejection fires before ``_promote_magic_lists_to_sweep_block``
    silently merges them.
    """
    if recipe_output is None:
        return
    sweep_parameters = recipe_output.get("sweep_parameters")
    if not sweep_parameters:
        return

    _reject_recipe_plus_magic_lists(cli, recipe_cls=_lookup_recipe_class(cli))

    existing = nested.get("sweep")
    if isinstance(existing, dict):
        existing.setdefault("type", "grid")
        existing.setdefault("parameters", {})
        existing["parameters"].update(sweep_parameters)
        if recipe_output.get("recipe_name"):
            existing.setdefault("recipe_name", recipe_output["recipe_name"])
    else:
        block: dict[str, Any] = {
            "type": "grid",
            "parameters": dict(sweep_parameters),
        }
        if recipe_output.get("recipe_name"):
            block["recipe_name"] = recipe_output["recipe_name"]
        nested["sweep"] = block


def _apply_recipe_scenarios(
    nested: dict[str, Any],
    recipe_output: dict[str, Any] | None,
    cli: CLIConfig,
) -> None:
    """Lift a recipe's ``scenarios`` branch onto the top-level ``sweep`` block.

    Mirrors ``_apply_recipe_sweep_parameters`` for the ScenarioSweep shape.
    Each scenario is a ``{"name": ..., "benchmark": {...}}`` dict; the result
    is a ``{"type": "scenarios", "runs": [...]}`` sweep block. Mutually
    exclusive with a YAML-declared sweep block AND with the grid-recipe path
    (``_recipe_output_to_dict`` populates exactly one of ``sweep_parameters``
    or ``scenarios``).
    """
    if recipe_output is None:
        return
    scenarios = recipe_output.get("scenarios")
    if not scenarios:
        return
    if nested.get("sweep") is not None:
        raise TypeError(
            "--search-recipe (scenarios path) is mutually exclusive with a "
            "YAML-declared sweep block. Drop one."
        )
    nested["sweep"] = {"type": "scenarios", "runs": list(scenarios)}
    if recipe_output.get("recipe_name"):
        nested["sweep"]["recipe_name"] = recipe_output["recipe_name"]


def _reject_recipe_plus_magic_lists(
    cli: CLIConfig, *, recipe_cls: Any | None = None
) -> None:
    """Raise when a CLIConfig field carries a magic-list alongside a grid recipe.

    Walks ``cli.model_fields_set`` for any user-set field whose name is in
    ``MAGIC_LIST_FIELDS`` and whose value is a list (e.g. ``concurrency``,
    ``prompt_input_tokens_mean``).

    A recipe whose ``consumed_magic_lists`` ClassVar includes a field name is
    NOT considered an offender for that field -- it consumes the user's list
    via ``ctx.sweep_overrides[field]`` instead.
    """
    consumed: frozenset[str] = (
        getattr(recipe_cls, "consumed_magic_lists", frozenset())
        if recipe_cls is not None
        else frozenset()
    )
    offenders: list[str] = []
    for name in cli.model_fields_set:
        if (
            name in MAGIC_LIST_FIELDS
            and name not in consumed
            and isinstance(getattr(cli, name), list)
        ):
            offenders.append(name)
    if offenders:
        raise TypeError(
            f"--search-recipe (grid path) is mutually exclusive with "
            f"magic-list flags {sorted(offenders)} -- the recipe owns the "
            "sweep parameters. Drop the list-shaped flag, or drop --search-recipe "
            "and configure the sweep by hand."
        )


def _lookup_recipe_class(cli: CLIConfig) -> Any | None:
    """Resolve the recipe plugin class from the user's --search-recipe value, or None."""
    if cli.search_recipe is None:
        return None
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.plugins import get_class

    try:
        return get_class(PluginType.SEARCH_RECIPE, cli.search_recipe)
    except Exception:  # noqa: BLE001 - missing recipe will surface as a clearer error downstream
        return None


def _promote_magic_lists_to_sweep_block(
    nested: dict[str, Any], sweep_type: str = "grid"
) -> None:
    """Lift list-shaped magic-list fields under ``phases[*]`` and selected
    dataset paths to a ``sweep`` block.

    PhaseConfig's scalar fields (``concurrency: int | None``, etc.) reject
    list inputs at validation time — but ``--concurrency 10,20,30`` is a
    list at this point. We detect any phase field whose key is in
    ``MAGIC_LIST_FIELDS`` and whose value is a list, strip it from the
    phase dict, and add it as a ``sweep.parameters`` entry keyed by the
    dotted path ``phases.<phase_name>.<field>`` — the same convention
    ``expand_sweep`` consumes downstream in ``build_benchmark_plan``.

    Dataset-rooted magic-lists (``--isl``, ``--osl``, ``--num-conversations``)
    are hoisted separately by ``_promote_cli_dataset_magic_lists`` before
    this function runs (they don't flow through ``phases[*]``).

    ``sweep_type`` controls the resulting ``sweep.type`` (``grid`` or
    ``zip``) — only meaningful when no YAML ``sweep:`` block already
    pins the type. With ``zip``, all hoisted lists must have equal
    length; ``expand_sweep`` enforces this downstream.

    No-ops when no list-shaped magic-list fields are present.
    """
    phases = nested.get("phases")
    if phases is None:
        return
    if not isinstance(phases, list):
        raise TypeError(
            f"phases must be a list of phase dicts, got "
            f"{type(phases).__name__}: {phases!r}"
        )
    sweep_parameters: dict[str, list[Any]] = {}
    for idx, phase in enumerate(phases):
        _hoist_phase_magic_lists(phase, idx, sweep_parameters)
    if sweep_parameters:
        _merge_into_sweep_block(nested, sweep_parameters, sweep_type)


def _hoist_phase_magic_lists(
    phase: Any, idx: int, sweep_parameters: dict[str, list[Any]]
) -> None:
    """Lift list-shaped magic-list fields out of one phase dict.

    Mutates ``phase`` in-place (replacing each hoisted list with its first
    element as a placeholder so required no-default phase fields still
    validate) and records the lifted lists into ``sweep_parameters``
    keyed by ``phases.<name>.<field>``.
    """
    if not isinstance(phase, dict):
        raise TypeError(
            f"phases[{idx}] must be a dict with a 'name' key, got "
            f"{type(phase).__name__}: {phase!r}. Sweep magic-list "
            f"promotion cannot lift list-shaped fields out of a "
            f"non-dict phase entry."
        )
    phase_name = phase.get("name")
    if not isinstance(phase_name, str):
        raise ValueError(
            f"phases[{idx}] is missing a string 'name' field "
            f"(got {phase_name!r}); cannot key sweep parameters on "
            f"phases.<name>.<field>."
        )
    for key in list(phase.keys()):
        if key in MAGIC_LIST_FIELDS and isinstance(phase[key], list):
            values = phase[key]
            sweep_parameters[f"phases.{phase_name}.{key}"] = values
            # Leave the first element behind as a placeholder. Required
            # phase fields without defaults (e.g. ``PoissonPhase.rate``)
            # would otherwise fail base-config validation after the list
            # is hoisted to the sweep block. Each sweep variation
            # overrides this scalar per-cell at expand time.
            if values:
                phase[key] = values[0]
            else:
                phase.pop(key)


def _merge_into_sweep_block(
    nested: dict[str, Any],
    additions: dict[str, list[Any]],
    sweep_type: str,
) -> None:
    """Merge ``additions`` into ``nested['sweep'].parameters``, creating
    the sweep block with the given ``type`` if absent. Shared by the
    phase-promote and dataset-promote passes.
    """
    existing_sweep = nested.get("sweep")
    if isinstance(existing_sweep, dict):
        existing_sweep.setdefault("type", sweep_type)
        existing_sweep.setdefault("parameters", {})
        existing_sweep["parameters"].update(additions)
    else:
        nested["sweep"] = {"type": sweep_type, "parameters": additions}


# CLI-attribute -> canonical dataset sweep path. Used by
# `_promote_cli_dataset_magic_lists` to hoist list-shaped dataset flags
# (--isl, --osl) onto the sweep block. Keys are CLI attribute names so
# we can read directly off CLIConfig; values are the dotted paths
# `expand_sweep` resolves against the deep-copied envelope.
_CLI_DATASET_MAGIC_LIST_PATHS: tuple[tuple[str, str], ...] = (
    ("prompt_input_tokens_mean", "datasets.main.prompts.isl.mean"),
    ("prompt_input_tokens_stddev", "datasets.main.prompts.isl.stddev"),
    ("prompt_output_tokens_mean", "datasets.main.prompts.osl.mean"),
    ("prompt_output_tokens_stddev", "datasets.main.prompts.osl.stddev"),
    ("conversation_turn_mean", "datasets.main.turns.mean"),
)


def _promote_cli_dataset_magic_lists(
    nested: dict[str, Any], cli: CLIConfig, sweep_type: str = "grid"
) -> None:
    """Hoist list-shaped CLI dataset flags to the sweep block.

    Mirrors `_promote_magic_lists_to_sweep_block` for fields that don't
    live on phases. ``--isl 128,512,2048`` parses as a list on
    ``cli.prompt_input_tokens_mean`` and must end up as a
    ``datasets.main.prompts.isl.mean: [128, 512, 2048]`` sweep parameter.

    The base config's dataset block keeps the first element of each list
    as a scalar placeholder so AIPerfConfig validation passes; each sweep
    variation overrides per-cell at expand time. The dataset converter is
    responsible for unwrapping the list at base-build time (see
    ``_build_synthetic_prompts_block``).
    """
    additions: dict[str, list[Any]] = {}
    for attr, path in _CLI_DATASET_MAGIC_LIST_PATHS:
        value = getattr(cli, attr, None)
        if isinstance(value, list):
            additions[path] = value
    if additions:
        _merge_into_sweep_block(nested, additions, sweep_type)


def _strip_consumed_magic_lists_from_phases(
    nested: dict[str, Any], consumed_magic_lists: frozenset[str]
) -> None:
    """Drop list-shaped recipe-consumed magic-list keys off each phase dict.

    Called after ``expand_search_recipe`` has already read the user's
    list-shaped values via ``sweep_overrides``. Removing the list-shaped
    keys from ``phases[*]`` prevents ``_promote_magic_lists_to_sweep_block``
    from re-lifting them as grid parameters onto the recipe's own sweep
    block (e.g. a scenarios block, which has ``extra="forbid"``). The
    field defaults take effect for the speculative ``BenchmarkConfig``
    path; the real per-scenario values come from the recipe's emitted
    ``runs[*].benchmark`` overlays.
    """
    phases = nested.get("phases")
    if not isinstance(phases, list):
        return
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        for field in consumed_magic_lists:
            if field in phase and isinstance(phase[field], list):
                del phase[field]


def _build_speculative_benchmark_config(
    nested: dict[str, Any], consumed_magic_lists: frozenset[str]
) -> Any:
    """Build a recipe-side ``BenchmarkConfig`` view of ``nested``.

    ``_promote_magic_lists_to_sweep_block`` runs after the recipe expansion,
    so any user-supplied magic-list (e.g. ``--concurrency 1,2,4``) is still
    list-shaped under ``phases[*]`` at the point the recipe needs a built
    ``BenchmarkConfig`` to plan its sweep. PhaseConfig's scalar fields
    reject lists, so we deep-copy ``nested``, drop the recipe-consumed
    magic-list keys off each phase dict so the field defaults apply, and
    build BenchmarkConfig from the stripped copy. The recipe still reads
    the original list values from the user-side ``CLIConfig`` via its
    own context, so nothing is lost.

    Stripping is gated to ``consumed_magic_lists``: a non-consumed list
    alongside a recipe is either user error (caught by
    ``_reject_recipe_plus_magic_lists`` on the grid path) or routed
    through ``_promote_magic_lists_to_sweep_block`` later (scenarios path).
    """
    from copy import deepcopy

    from aiperf.config.config import BenchmarkConfig

    if not consumed_magic_lists:
        return BenchmarkConfig(**nested)

    nested_for_speculative = deepcopy(nested)
    _strip_consumed_magic_lists_from_phases(
        nested_for_speculative, consumed_magic_lists
    )
    return BenchmarkConfig(**nested_for_speculative)


def _expand_recipe_and_strip_consumed_lists(
    nested: dict[str, Any], cli: CLIConfig
) -> Any:
    """Build the speculative BenchmarkConfig, expand the recipe, then strip.

    Magic-list CLI inputs (e.g. ``--concurrency 1,2,4``) leave list-shaped
    values under ``phases[*]`` until ``_promote_magic_lists_to_sweep_block``
    lifts them to the sweep block. Constructing BenchmarkConfig before that
    would reject those lists against the scalar phase fields. Recipes don't
    combine with magic-list CLI inputs except for the fields the recipe opts
    into via ``consumed_magic_lists`` (e.g. pareto-sweep consumes
    ``--concurrency``); for those, strip the list-shaped values out of the
    speculative dict so BenchmarkConfig validates -- the recipe reads the
    original user-side list during ``expand_search_recipe``.

    After expansion, strip the consumed lists from the real ``phases`` so
    the post-recipe magic-list promoter doesn't double-lift them onto a
    sweep block the recipe just owned (e.g. a scenarios block, which
    forbids ``parameters``).

    Expanding the recipe up-front (rather than inside ``_assemble_optional``
    + ``_apply_recipe_sweep_parameters`` separately) lets both call sites
    share one ``recipe.expand()`` invocation.
    """
    from aiperf.config.config import BenchmarkConfig

    has_recipe = (
        "search_recipe" in cli.model_fields_set and cli.search_recipe is not None
    )
    benchmark_config: BenchmarkConfig | None = None
    consumed_for_recipe: frozenset[str] = frozenset()
    if has_recipe:
        recipe_cls = _lookup_recipe_class(cli)
        consumed_for_recipe = (
            getattr(recipe_cls, "consumed_magic_lists", frozenset())
            if recipe_cls is not None
            else frozenset()
        )
        benchmark_config = _build_speculative_benchmark_config(
            nested, consumed_for_recipe
        )

    recipe_output = expand_search_recipe(cli, benchmark_config=benchmark_config)

    if consumed_for_recipe:
        _strip_consumed_magic_lists_from_phases(nested, consumed_for_recipe)
    return recipe_output


def convert_cli_to_aiperf(cli: CLIConfig) -> AIPerfConfig:
    """Convert a parsed ``CLIConfig`` into ``AIPerfConfig``.

    Composes the seven section-builders, then runs the assembled nested dict
    through ``AIPerfConfig`` validation. Optional sections (warmup, tokenizer,
    accuracy, multi_run, slos) are included only when their builders return a
    non-empty result.
    """
    from aiperf.config.config import AIPerfConfig

    nested = _assemble_envelope_dict(cli)
    _apply_variants_scenario_sweep(nested, cli)
    return AIPerfConfig(**nested)


def _assemble_envelope_dict(cli: CLIConfig) -> dict[str, Any]:
    """Run the section-builders and return the envelope-shaped dict.

    Split out of ``convert_cli_to_aiperf`` so the variant emitter can
    re-assemble per-variant dicts without re-validating against
    ``AIPerfConfig``.
    """
    endpoint = build_endpoint(cli)
    models = build_models(cli)
    prof = build_profiling(cli)

    phases: list[dict[str, Any]] = []
    if (warmup := build_warmup(cli)) is not None:
        phases.append({"name": "warmup", **warmup})
    phases.append({"name": "profiling", **prof})

    ds = build_dataset(cli)

    _init_random_seed(cli)
    artifacts = build_artifacts(cli)
    gpu_telemetry = build_gpu_telemetry(cli)
    server_metrics = build_server_metrics(cli)
    otel = build_otel(cli)
    mlflow = build_mlflow(cli)
    logging_dict, runtime_dict = build_logging_runtime(cli)

    nested: dict[str, Any] = {
        "endpoint": endpoint,
        "models": models,
        "phases": phases,
        # Dataset name "main": kept in sync with _V1_DEFAULT_DATASET_NAME in
        # search_recipes.builtins (which can't import from this module without
        # creating a load-order cycle through aiperf.config/__init__.py).
        # If renaming, update both call sites and the regression test in
        # tests/unit/search_recipes/test_grid_recipe_converter.py.
        "datasets": [{"name": "main", **ds}],
        "artifacts": artifacts,
        "gpu_telemetry": gpu_telemetry,
        "server_metrics": server_metrics,
        "otel": otel,
        "mlflow": mlflow,
    }
    if logging_dict:
        nested["logging"] = logging_dict
    if runtime_dict:
        nested["runtime"] = runtime_dict

    recipe_output = _expand_recipe_and_strip_consumed_lists(nested, cli)

    # auto_plot resolves alongside the recipe: explicit CLI flag wins, else the
    # recipe's auto_plot_default, else False. plot_required is a pass-through.
    auto_plot, plot_required = resolve_auto_plot(cli)
    artifacts["auto_plot"] = auto_plot
    artifacts["plot_required"] = plot_required

    _assemble_optional(nested, cli, recipe_output=recipe_output)
    _apply_recipe_sweep_parameters(nested, recipe_output, cli)
    _apply_recipe_scenarios(nested, recipe_output, cli)
    sweep_type = getattr(cli, "sweep_type", "grid")
    _promote_cli_dataset_magic_lists(nested, cli, sweep_type=sweep_type)
    _promote_magic_lists_to_sweep_block(nested, sweep_type=sweep_type)
    _apply_parameter_sweep_meta_to_sweep(nested, cli)

    return _wrap_under_envelope(nested)


def _apply_parameter_sweep_meta_to_sweep(
    nested: dict[str, Any], cli: CLIConfig
) -> None:
    """Stamp ``--parameter-sweep-*`` knobs on the top-level ``sweep:`` block.

    Schema-2.0 moved these from ``MultiRunConfig`` to the per-sweep config
    (``GridSweep.iteration_order`` / ``GridSweep.same_seed`` /
    ``GridSweep.cooldown_seconds``). They are CLI-set on
    ``CLIConfig.sweeping``; lift them onto whatever sweep block
    ``_assemble_optional`` / ``_promote_magic_lists_to_sweep_block``
    already produced. No-op when no sweep is in flight.

    ``iteration_order`` and ``same_seed`` only exist on grid-shaped sweeps
    (``GridSweep`` / ``ScenarioSweep`` via ``_GridSweepBase``); the
    adaptive-search envelope inherits from ``_SweepBase`` directly and
    sets ``extra="forbid"``, so writing those keys onto an
    ``adaptive_search`` sweep would crash Pydantic validation. Gate the
    stamp on the sweep being grid-shaped. ``cooldown_seconds`` lives on
    ``_SweepBase`` and applies to all sweep types, so it stays
    unconditional.
    """
    sweep = nested.get("sweep")
    if not isinstance(sweep, dict):
        return
    set_fields = cli.model_fields_set
    is_grid_shaped = sweep.get("type") in ("grid", "scenarios")
    if is_grid_shaped and "parameter_sweep_mode" in set_fields:
        sweep["iteration_order"] = cli.parameter_sweep_mode
    if is_grid_shaped and "parameter_sweep_same_seed" in set_fields:
        sweep["same_seed"] = cli.parameter_sweep_same_seed
    if "parameter_sweep_cooldown_seconds" in set_fields:
        sweep["cooldown_seconds"] = cli.parameter_sweep_cooldown_seconds


def _apply_variants_scenario_sweep(
    nested: dict[str, Any],
    cli: CLIConfig,
) -> None:
    """Emit a ``ScenarioSweep`` block from `--variant` occurrences.

    Each variant string parses to (label, kvpairs); each kvpair maps a
    short-CLI alias to a value. Per variant, we deep-copy the cli, apply
    the overrides, re-assemble the envelope dict, and diff its
    ``benchmark`` subtree vs the base ``benchmark`` subtree. The resulting
    diff lands as ``runs[i].benchmark``. Mutually exclusive with magic-list
    flags, --search-recipe, and a YAML-declared sweep block.
    """
    if not cli.sweep_variants:
        return

    variants = list(cli.sweep_variants)
    if len(variants) == 1:
        raise TypeError(
            "--variant: single occurrence is rejected. Use the individual "
            "--isl/--osl/--concurrency flags for a one-off; --variant is for "
            "multi-variant sweeps. Pass --variant at least twice to declare a "
            "ScenarioSweep."
        )
    if "search_recipe" in cli.model_fields_set and cli.search_recipe is not None:
        raise TypeError(
            "--variant is mutually exclusive with --search-recipe. Drop one: "
            "recipes own the sweep parameters, --variant declares scenarios."
        )
    _reject_variants_plus_magic_lists(cli)
    if nested.get("sweep") is not None:
        raise TypeError(
            "--variant is mutually exclusive with a YAML-declared sweep block "
            "(or --search-* flags). Drop --variant or remove the sweep "
            "configuration."
        )

    from aiperf.config.flags.variant_parser import build_alias_table, parse_variant

    alias_table = build_alias_table()

    runs: list[dict[str, Any]] = []
    for auto_index, raw in enumerate(variants):
        name, kvpairs = parse_variant(raw)
        unknown = sorted(k for k in kvpairs if k not in alias_table)
        if unknown:
            sample = ", ".join(sorted(alias_table.keys())[:20])
            raise TypeError(
                f"--variant {raw!r}: unknown key(s) {unknown}. Supported "
                f"keys (sample of {len(alias_table)}): {sample}, ... -- any "
                f"CLI flag (without --) is accepted."
            )
        variant_cli = cli.model_copy(deep=True)
        variant_cli.sweep_variants = []
        for alias, value in kvpairs.items():
            cli_path = alias_table[alias]
            _set_cli_path(variant_cli, cli_path, value)
        variant_envelope = _assemble_envelope_dict(variant_cli)
        run_benchmark = _diff_envelope_benchmark(
            base=nested.get("benchmark", {}),
            override=variant_envelope.get("benchmark", {}),
        )
        run: dict[str, Any] = {"name": name if name is not None else f"v{auto_index}"}
        if run_benchmark:
            run["benchmark"] = run_benchmark
        runs.append(run)

    sweep_block: dict[str, Any] = {"type": "scenarios", "runs": runs}
    nested["sweep"] = sweep_block
    _apply_parameter_sweep_meta_to_sweep(nested, cli)


def _reject_variants_plus_magic_lists(cli: CLIConfig) -> None:
    """Hard-fail when --variant is set alongside any magic-list flag."""
    offenders: list[str] = []
    for name in cli.model_fields_set:
        if name in MAGIC_LIST_FIELDS and isinstance(getattr(cli, name), list):
            offenders.append(name)
    if offenders:
        raise TypeError(
            f"--variant is mutually exclusive with magic-list flags "
            f"{sorted(offenders)} (e.g. --concurrency 1,2,4). Drop one: "
            "magic-lists declare a Cartesian sweep, --variant declares "
            "hand-picked scenarios."
        )


def _set_cli_path(cli: CLIConfig, dotted_path: str, value: Any) -> None:
    """Walk a dotted field path on CLIConfig and assign ``value``.

    Auto-constructs intermediate ``BaseConfig`` sub-models when they are
    None on ``cli``. Pydantic assignment updates ``model_fields_set`` so
    converters detect the override.
    """
    parts = dotted_path.split(".")
    current: Any = cli
    for part in parts[:-1]:
        nxt = getattr(current, part, None)
        if nxt is None:
            field_info = type(current).model_fields.get(part)
            sub_cls = (
                _resolve_baseconfig_cls(field_info.annotation)
                if field_info is not None
                else None
            )
            if sub_cls is None:
                raise TypeError(
                    f"--variant: cannot resolve path {dotted_path!r} -- "
                    f"sub-model {part!r} is None on {type(current).__name__} "
                    f"and no BaseConfig type could be inferred."
                )
            nxt = sub_cls()
            setattr(current, part, nxt)
        current = nxt
    setattr(current, parts[-1], value)


def _resolve_baseconfig_cls(annotation: Any) -> type | None:
    from aiperf.config.flags.variant_parser import _extract_baseconfig

    return _extract_baseconfig(annotation)


def _diff_envelope_benchmark(
    *, base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Return only the keys/subtrees of `override` that differ from `base`.

    Mirrors the deep-merge semantics in ``aiperf.config.sweep.expand``:
    nested dicts recurse; lists of name-bearing dicts diff by ``name`` and
    emit only changed entries (with the matching name for downstream
    re-merge); other lists / scalars compare by equality.
    """
    out: dict[str, Any] = {}
    for key, ov in override.items():
        if key not in base:
            out[key] = ov
            continue
        bv = base[key]
        if isinstance(bv, dict) and isinstance(ov, dict):
            sub = _diff_envelope_benchmark(base=bv, override=ov)
            if sub:
                out[key] = sub
            continue
        if (
            isinstance(bv, list)
            and isinstance(ov, list)
            and _all_named_dicts(bv)
            and _all_named_dicts(ov)
        ):
            sub_list = _diff_named_dict_list(bv, ov)
            if sub_list:
                out[key] = sub_list
            continue
        if bv != ov:
            out[key] = ov
    return out


def _all_named_dicts(items: list[Any]) -> bool:
    return bool(items) and all(
        isinstance(it, dict) and isinstance(it.get("name"), str) for it in items
    )


def _diff_named_dict_list(
    base: list[dict[str, Any]], override: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_name = {b["name"]: b for b in base}
    out: list[dict[str, Any]] = []
    for ov in override:
        name = ov["name"]
        if name not in by_name:
            out.append(ov)
            continue
        sub = _diff_envelope_benchmark(base=by_name[name], override=ov)
        if sub:
            out.append({"name": name, **{k: v for k, v in sub.items() if k != "name"}})
    return out
