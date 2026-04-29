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
)
from aiperf.config.v1._converter_profiling import build_profiling
from aiperf.config.v1._converter_runtime import build_artifacts, build_logging_runtime
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


def _assemble_optional(nested: dict[str, Any], user: UserConfig) -> None:
    if tok := build_tokenizer(user):
        nested["tokenizer"] = tok
    if acc := build_accuracy(user):
        nested["accuracy"] = acc
    if mr := build_multi_run(user):
        nested["multi_run"] = mr
    inp = user.input
    if inp is not None:
        if "random_seed" in inp.model_fields_set:
            nested["random_seed"] = inp.random_seed
        if inp.goodput:
            nested["slos"] = dict(inp.goodput)


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
    if not isinstance(phases, list):
        return
    sweep_variables: dict[str, list[Any]] = {}
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_name = phase.get("name")
        if not isinstance(phase_name, str):
            continue
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


def _reject_unsupported_sweep_mode(user: UserConfig) -> None:
    """Reject ``--parameter-sweep-mode=repeated`` per Path-A.

    The k8s sweep port keeps INDEPENDENT semantics only — variations form the
    outer loop, trials form the inner loop, matching ``expand_sweep`` +
    ``FixedTrialsStrategy``. REPEATED would require an outer trial loop the
    orchestrator does not support; defer to a follow-up phase if needed.
    """
    loadgen = getattr(user, "loadgen", None)
    if loadgen is None:
        return
    mode = getattr(loadgen, "parameter_sweep_mode", None)
    if mode is None:
        return
    mode_str = str(mode).lower()
    if "repeated" in mode_str:
        raise ValueError(
            "--parameter-sweep-mode=repeated is not supported in this release; "
            "use 'independent' (the default)."
        )


def convert_user_to_aiperf(user: UserConfig, service: ServiceConfig) -> AIPerfConfig:
    """Convert a parsed v1 ``UserConfig`` + ``ServiceConfig`` into ``AIPerfConfig``.

    Composes the seven section-builders, then runs the assembled nested dict
    through ``AIPerfConfig`` validation. Optional sections (warmup, tokenizer,
    accuracy, multi_run, slos) are included only when their builders return a
    non-empty result.
    """
    from aiperf.config.config import AIPerfConfig

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
        "datasets": [{"name": "main", **ds}],
        "artifacts": artifacts,
        "gpu_telemetry": gpu_telemetry,
        "server_metrics": server_metrics,
    }
    if logging_dict:
        nested["logging"] = logging_dict
    if runtime_dict:
        nested["runtime"] = runtime_dict

    _assemble_optional(nested, user)
    _reject_unsupported_sweep_mode(user)
    _promote_magic_lists_to_sweep_block(nested)

    return AIPerfConfig(**nested)
