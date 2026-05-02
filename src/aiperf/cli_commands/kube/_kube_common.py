# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers used by `aiperf kube profile` and `aiperf kube sweep`.

These helpers do not depend on AIPerfJob CR shape; they are concerned with
turning a `UserConfig` + `ServiceConfig` / config-file pair into an
`AIPerfConfig`, generating a DNS-safe benchmark name, and printing the memory
estimate panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config import AIPerfConfig
    from aiperf.config.kube import KubeOptions
    from aiperf.config.v1 import ServiceConfig, UserConfig


def resolve_config(
    user_config: UserConfig,
    service_config: ServiceConfig,
    config_file: Path | None,
) -> AIPerfConfig:
    """Return an `AIPerfConfig` from a plain YAML config file or CLI flags.

    Args:
        user_config: Parsed v1 ``UserConfig`` carrying flag-form benchmark options.
        service_config: Parsed v1 ``ServiceConfig`` carrying service-level
            options (UI, log level, ZMQ, etc.).
        config_file: Optional path to a YAML config file. When provided, the
            YAML supplies the base configuration and any explicitly-set CLI
            flags on ``user_config`` are deep-merged on top before validation.
            Without ``config_file``, the v1 -> v2 converter handles the full
            CLI-only path.

    Returns:
        Fully resolved `AIPerfConfig` ready for downstream use.
    """
    from aiperf.config.v1.converter import convert_user_to_aiperf

    if config_file is None:
        return convert_user_to_aiperf(user_config, service_config)

    from aiperf.config import AIPerfConfig
    from aiperf.config.loader import load_config_dict

    yaml_dict = load_config_dict(config_file)
    overrides = _build_v1_overrides(user_config)
    merged = _deep_merge(yaml_dict, overrides) if overrides else yaml_dict
    return AIPerfConfig.model_validate(merged)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; non-dict values replace.

    Lists are replaced wholesale (not concatenated) so that a CLI override
    list cleanly clobbers a YAML list rather than appending.
    """
    import copy

    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build_v1_overrides(user: UserConfig) -> dict[str, Any]:
    """Translate explicitly-set v1 CLI flags into a v2-shape override dict.

    Only fields the user explicitly set (per nested model's
    ``model_fields_set``) flow through; everything else is left for the YAML
    base to supply. Reuses the v1 converter's section-builders for endpoint /
    multi-run / tokenizer / accuracy so the YAML+CLI path produces identical
    v2 shape to the CLI-only path for the same inputs.

    Returns an empty dict when the user passed no CLI overrides; callers
    short-circuit the deep-merge in that case.
    """
    from aiperf.config.v1._converter_optionals import (
        build_accuracy,
        build_multi_run,
        build_tokenizer,
        expand_search_recipe,
    )

    out: dict[str, Any] = {}
    _apply_endpoint_overrides(out, user)
    _apply_input_overrides(out, user)

    # Recipes drive multi_run / sweep / sla_filters; reuse the converter path
    # so YAML+CLI emits the same shape as CLI-only.
    recipe_output = expand_search_recipe(user)
    if recipe_output is not None:
        sweep_vars = recipe_output.get("sweep_variables")
        if sweep_vars:
            out["sweep"] = {"type": "grid", "variables": dict(sweep_vars)}
    multi_run = build_multi_run(user, recipe_output=recipe_output)
    if multi_run:
        out["multi_run"] = multi_run

    tokenizer = build_tokenizer(user)
    if tokenizer:
        out["tokenizer"] = tokenizer
    accuracy = build_accuracy(user)
    if accuracy:
        out["accuracy"] = accuracy

    return out


def _apply_endpoint_overrides(out: dict[str, Any], user: UserConfig) -> None:
    """Translate explicitly-set endpoint flags into ``out['endpoint']`` and
    ``out['models']``.

    ``--model-names`` lives on v1 ``EndpointConfig`` but maps to the v2
    ``models.items`` block; everything else stays on ``endpoint``.
    """
    from aiperf.config.v1._converter_endpoint import _ENDPOINT_FIELD_MAP

    ep = user.endpoint
    if ep is None or not ep.model_fields_set:
        return
    ep_set = ep.model_fields_set
    endpoint: dict[str, Any] = {}
    if "urls" in ep_set:
        endpoint["urls"] = list(ep.urls)
    for v1_field, v2_key in _ENDPOINT_FIELD_MAP.items():
        if v1_field in ep_set:
            endpoint[v2_key] = getattr(ep, v1_field)
    if endpoint:
        out["endpoint"] = endpoint
    if "model_names" in ep_set and ep.model_names:
        models: dict[str, Any] = {"items": [{"name": name} for name in ep.model_names]}
        if "model_selection_strategy" in ep_set:
            models["selection_strategy"] = ep.model_selection_strategy
        out["models"] = models


def _apply_input_overrides(out: dict[str, Any], user: UserConfig) -> None:
    """Mirror ``build_endpoint``'s rule that ``--headers`` / ``--extra`` (which
    live on v1 ``InputConfig``) flow into the v2 ``endpoint`` block.
    """
    inp = user.input
    if inp is None or not inp.model_fields_set:
        return
    inp_set = inp.model_fields_set
    endpoint = out.setdefault("endpoint", {})
    if "headers" in inp_set and inp.headers:
        endpoint["headers"] = dict(inp.headers)
    if "extra" in inp_set and inp.extra:
        endpoint["extra"] = dict(inp.extra)
    if not endpoint:
        out.pop("endpoint", None)


def generate_benchmark_name(config: AIPerfConfig, *, suffix: str = "") -> str:
    """Generate a short DNS-safe benchmark name from `config`.

    Used by both `aiperf kube profile` and `aiperf kube sweep`.

    Args:
        config: AIPerfConfig instance.
        suffix: Optional suffix appended after a hyphen (e.g. ``"sweep"``).

    Returns:
        A short hyphenated name like ``"qwen3-openai-throughput"`` or
        ``"qwen3-openai-throughput-sweep"`` when a suffix is provided.
    """
    import re

    model_name = config.get_model_names()[0].split("/")[-1].lower()
    endpoint_type = str(config.endpoint.type)
    first_phase = config.phases[0]
    phase_type = str(first_phase.type)
    parts = [model_name, endpoint_type, phase_type]
    if suffix:
        parts.append(suffix)
    raw = "-".join(parts)
    return re.sub(r"[^a-z0-9-]", "-", raw).strip("-")[:40]


def print_memory_estimate(
    config: Any,
    kube_options: KubeOptions,
    spec: dict,
    *,
    label_prefix: str = "",
) -> None:
    """Compute and display the memory estimate panel for the planned benchmark.

    Args:
        config: Resolved `AIPerfConfig`.
        kube_options: Composite kube CLI options (workers count, etc.).
        spec: Submitted CRD spec dict; used to read ``connectionsPerWorker``.
        label_prefix: Optional prefix printed before the estimate (e.g.
            ``"Sweep template: "``); empty by default.
    """
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.memory_estimator import estimate_memory, format_estimate

    mem_est = estimate_memory(
        config,
        total_workers=kube_options.workers,
        workers_per_pod=config.runtime.workers_per_pod,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
    )
    rendered = format_estimate(mem_est)
    if label_prefix:
        kube_console.console.print(f"{label_prefix}", highlight=False)
    kube_console.console.print(rendered, highlight=False)
