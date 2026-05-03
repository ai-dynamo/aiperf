# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 ``UserConfig`` + ``ServiceConfig`` -> ``AIPerfConfig`` entrypoint.

Composes the seven section-builders that live alongside this module
(``_converter_endpoint``, ``_converter_profiling``, ``_converter_warmup``,
``_converter_dataset``, ``_converter_runtime``, ``_converter_telemetry``,
``_converter_optionals``) into a single nested dict, then validates it
through ``AIPerfConfig``. Mirrors the flat-CLI ``build_aiperf_config`` in
``aiperf.config.cli_converter`` but reads from the nested v1 DTOs.

The converter is the only module outside ``cli_commands/`` allowed to read
v1 attributes; downstream code consumes the validated ``AIPerfConfig``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from aiperf.config.sweep import MAGIC_LIST_FIELDS
from aiperf.config.v1._converter_dataset import build_dataset
from aiperf.config.v1._converter_endpoint import build_endpoint, build_models
from aiperf.config.v1._converter_optionals import (
    build_accuracy,
    build_multi_run,
    build_tokenizer,
    expand_search_recipe,
)
from aiperf.config.v1._converter_profiling import build_profiling
from aiperf.config.v1._converter_runtime import (
    build_artifacts,
    build_logging_runtime,
    validate_steady_state,
)
from aiperf.config.v1._converter_telemetry import (
    build_gpu_telemetry,
    build_server_metrics,
)
from aiperf.config.v1._converter_warmup import build_warmup

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig
    from aiperf.config.v1 import ServiceConfig, UserConfig


def _init_random_seed(user: UserConfig) -> None:
    from aiperf.common import random_generator as rng
    from aiperf.common.exceptions import InvalidStateError

    seed = user.input.random_seed if user.input is not None else None
    if seed is None:
        return
    with contextlib.suppress(InvalidStateError):
        rng.init(seed)


def _assemble_optional(
    nested: dict[str, Any],
    user: UserConfig,
    *,
    recipe_output: dict[str, Any] | None,
) -> None:
    if tok := build_tokenizer(user):
        nested["tokenizer"] = tok
    if acc := build_accuracy(user):
        nested["accuracy"] = acc
    if mr := build_multi_run(user, recipe_output=recipe_output):
        nested["multi_run"] = mr
    inp = user.input
    if inp is not None:
        if "random_seed" in inp.model_fields_set:
            nested["random_seed"] = inp.random_seed
        if inp.goodput:
            nested["slos"] = dict(inp.goodput)


def _apply_recipe_sweep_variables(
    nested: dict[str, Any],
    recipe_output: dict[str, Any] | None,
    user: UserConfig,
) -> None:
    """Lift a grid-recipe's ``sweep_variables`` onto the top-level ``sweep`` block.

    Mutually exclusive with magic-list flags: a recipe owns sweep variables on
    the grid path, so the user passing ``--concurrency 10,20,30`` alongside a
    grid ``--search-recipe`` is ambiguous (which list wins?). We defer the
    decision to the user by hard-failing here with a clear message. The
    detection runs against the v1 ``UserConfig`` (not the assembled phase
    dicts) so the rejection fires before ``_promote_magic_lists_to_sweep_block``
    silently merges them.
    """
    if recipe_output is None:
        return
    sweep_variables = recipe_output.get("sweep_variables")
    if not sweep_variables:
        return

    _reject_recipe_plus_magic_lists(user)

    existing = nested.get("sweep")
    if isinstance(existing, dict):
        existing.setdefault("type", "grid")
        existing.setdefault("variables", {})
        existing["variables"].update(sweep_variables)
    else:
        nested["sweep"] = {"type": "grid", "variables": dict(sweep_variables)}


def _reject_recipe_plus_magic_lists(user: UserConfig) -> None:
    """Raise when a v1 phase field carries a magic-list alongside a grid recipe.

    Walks the v1 sub-models (loadgen / input) for any user-set field whose name
    is in ``MAGIC_LIST_FIELDS`` and whose value is a list. Magic-list fields can
    live on multiple v1 nests (e.g. ``loadgen.concurrency``,
    ``input.synthetic_input_tokens.mean``) so a generic walk is the simplest
    way to catch all of them.
    """
    offenders: list[str] = []
    for sub_name in ("loadgen", "input"):
        sub = getattr(user, sub_name, None)
        if sub is None:
            continue
        for name in sub.model_fields_set:
            if name in MAGIC_LIST_FIELDS and isinstance(getattr(sub, name), list):
                offenders.append(f"{sub_name}.{name}")
    if offenders:
        raise TypeError(
            f"--search-recipe (grid path) is mutually exclusive with "
            f"magic-list flags {sorted(offenders)} -- the recipe owns the "
            "sweep variables. Drop the list-shaped flag, or drop --search-recipe "
            "and configure the sweep by hand."
        )


def _promote_magic_lists_to_sweep_block(nested: dict[str, Any]) -> None:
    """Lift list-shaped magic-list fields under ``phases[*]`` to a ``sweep`` block.

    PhaseConfig's scalar fields (``concurrency: int | None``, etc.) reject
    list inputs at validation time — but ``--concurrency 10,20,30`` is a
    list at this point. We detect any phase field whose key is in
    ``MAGIC_LIST_FIELDS`` and whose value is a list, strip it from the
    phase dict, and add it as a ``sweep.variables`` entry keyed by the
    dotted path ``phases.<phase_name>.<field>`` — the same convention
    ``expand_sweep`` consumes downstream in ``build_benchmark_plan``.

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
    sweep_variables: dict[str, list[Any]] = {}
    for idx, phase in enumerate(phases):
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
                f"(got {phase_name!r}); cannot key sweep variables on "
                f"phases.<name>.<field>."
            )
        for key in list(phase.keys()):
            if key in MAGIC_LIST_FIELDS and isinstance(phase[key], list):
                sweep_variables[f"phases.{phase_name}.{key}"] = phase.pop(key)
    if sweep_variables:
        existing_sweep = nested.get("sweep")
        if isinstance(existing_sweep, dict):
            existing_sweep.setdefault("type", "grid")
            existing_sweep.setdefault("variables", {})
            existing_sweep["variables"].update(sweep_variables)
        else:
            nested["sweep"] = {"type": "grid", "variables": sweep_variables}


def convert_user_to_aiperf(user: UserConfig, service: ServiceConfig) -> AIPerfConfig:
    """Convert a parsed v1 ``UserConfig`` + ``ServiceConfig`` into ``AIPerfConfig``.

    Composes the seven section-builders, then runs the assembled nested dict
    through ``AIPerfConfig`` validation. Optional sections (warmup, tokenizer,
    accuracy, multi_run, slos) are included only when their builders return a
    non-empty result.
    """
    from aiperf.config.config import AIPerfConfig

    validate_steady_state(user)

    # Expanding the recipe up-front lets _assemble_optional and
    # _apply_recipe_sweep_variables share one ``recipe.expand()`` call instead
    # of running it twice.
    recipe_output = expand_search_recipe(user)

    endpoint = build_endpoint(user)
    models = build_models(user)
    prof = build_profiling(user)

    phases: list[dict[str, Any]] = []
    if (warmup := build_warmup(user)) is not None:
        phases.append({"name": "warmup", **warmup})
    phases.append({"name": "profiling", **prof})

    ds = build_dataset(user)

    _init_random_seed(user)
    artifacts = build_artifacts(user)
    gpu_telemetry = build_gpu_telemetry(user)
    server_metrics = build_server_metrics(user)
    logging_dict, runtime_dict = build_logging_runtime(user, service)

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
    }
    if logging_dict:
        nested["logging"] = logging_dict
    if runtime_dict:
        nested["runtime"] = runtime_dict

    _assemble_optional(nested, user, recipe_output=recipe_output)
    _apply_recipe_sweep_variables(nested, recipe_output, user)
    _promote_magic_lists_to_sweep_block(nested)

    return AIPerfConfig(**nested)
