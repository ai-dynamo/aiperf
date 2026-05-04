# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a v1 ``UserConfig`` + optional YAML ``--config`` file into a v2
``AIPerfConfig``.

Used by every CLI command that supports both flag-form and file-form input
(``aiperf profile``, ``aiperf kube profile``, ``aiperf kube sweep``,
``aiperf kube generate``). When both are supplied, the YAML supplies the
base configuration and any explicitly-set CLI flags on ``user_config`` are
deep-merged on top before AIPerfConfig validation -- so
``aiperf profile --config foo.yaml --search-recipe X --ttft-sla-ms 200``
works the way users intuit instead of throwing
``UserConfig.endpoint.modelNames: Field required``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config import AIPerfConfig
    from aiperf.config.v1 import ServiceConfig, UserConfig


def resolve_config(
    user_config: UserConfig,
    service_config: ServiceConfig,
    config_file: Path | None,
) -> AIPerfConfig:
    """Return an `AIPerfConfig` from a YAML config file and/or CLI flags.

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
    from aiperf.config.v1.converter import (
        _apply_consistent_seed_default,
        _wrap_under_envelope,
        convert_user_to_aiperf,
    )

    if config_file is None:
        return convert_user_to_aiperf(user_config, service_config)

    from aiperf.config import AIPerfConfig
    from aiperf.config.loader import load_config_dict

    yaml_dict = load_config_dict(config_file)
    overrides = build_v1_overrides(user_config, service_config)
    if overrides:
        overrides = _wrap_under_envelope(overrides)
    merged = deep_merge(yaml_dict, overrides) if overrides else yaml_dict
    # Apply --set-consistent-seed=True default-seed-42 to the merged shape:
    # neither the YAML nor explicit CLI flags supplied a seed, but
    # set_consistent_seed (the default) still promises one. See
    # converter._apply_consistent_seed_default.
    _apply_consistent_seed_default(merged)
    return AIPerfConfig.model_validate(merged)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; non-dict values replace.

    Lists are replaced wholesale (not concatenated) so that a CLI override
    list cleanly clobbers a YAML list rather than appending.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_v1_overrides(
    user: UserConfig, service: ServiceConfig | None = None
) -> dict[str, Any]:
    """Translate explicitly-set v1 CLI flags into a v2-shape override dict.

    Only fields the user explicitly set (per nested model's
    ``model_fields_set``) flow through; everything else is left for the YAML
    base to supply. Reuses the v1 converter's section-builders for endpoint /
    multi-run / tokenizer / accuracy / runtime / logging so the YAML+CLI path
    produces identical v2 shape to the CLI-only path for the same inputs.

    Returns an empty dict when the user passed no CLI overrides; callers
    short-circuit the deep-merge in that case.
    """
    from aiperf.config.v1._converter_optionals import (
        build_accuracy,
        build_tokenizer,
    )
    from aiperf.config.v1._converter_runtime import build_logging_runtime

    out: dict[str, Any] = {}
    _apply_endpoint_overrides(out, user)
    _apply_input_overrides(out, user)
    _apply_recipe_and_multirun(out, user)
    _apply_artifacts_overrides(out, user)
    _apply_optional_section(out, "tokenizer", build_tokenizer(user))
    _apply_optional_section(out, "accuracy", build_accuracy(user))

    if service is not None:
        # Service-level CLI flags (--ui, --log-level, --verbose, ZMQ knobs)
        # land on RuntimeConfig / LoggingConfig in v2. build_logging_runtime
        # already gates on service.model_fields_set, so YAML defaults stay
        # intact when the user didn't pass these flags.
        logging_dict, runtime_dict = build_logging_runtime(user, service)
        _apply_optional_section(out, "logging", logging_dict)
        _apply_optional_section(out, "runtime", runtime_dict)

    return out


def _apply_optional_section(
    out: dict[str, Any], key: str, value: dict[str, Any] | None
) -> None:
    """Set ``out[key] = value`` only when value is non-empty, mirroring the
    converter's policy of omitting empty subsections."""
    if value:
        out[key] = value


def _apply_recipe_and_multirun(out: dict[str, Any], user: UserConfig) -> None:
    """Recipes drive multi_run / sweep / sla_filters; reuse the converter
    path so YAML+CLI emits the same shape as CLI-only."""
    from aiperf.config.v1._converter_optionals import (
        build_multi_run,
        expand_search_recipe,
    )

    recipe_output = expand_search_recipe(user)
    if recipe_output is not None:
        sweep_vars = recipe_output.get("sweep_variables")
        if sweep_vars:
            # Recipe paths are body-rooted (``phases.<name>.<field>``);
            # the envelope-shaped sweep block needs them prefixed with
            # ``benchmark.`` to resolve. Idempotent for keys already so prefixed.
            prefixed = {
                (k if k.startswith("benchmark.") else f"benchmark.{k}"): v
                for k, v in sweep_vars.items()
            }
            out["sweep"] = {"type": "grid", "variables": prefixed}
    multi_run = build_multi_run(user, recipe_output=recipe_output)
    if multi_run:
        out["multi_run"] = multi_run


def _apply_artifacts_overrides(out: dict[str, Any], user: UserConfig) -> None:
    """Map ``--artifact-dir`` and friends to the v2 ``artifacts`` block.

    ``build_artifacts`` always synthesizes ``cli_command`` from sys.argv even
    when no --output flag was passed; we only emit the block when the user
    actually set an OutputConfig field, so a YAML ``artifacts.dir`` stays
    untouched on a plain ``aiperf profile -f base.yaml`` invocation.
    """
    from aiperf.config.v1._converter_runtime import build_artifacts

    if user.output is None or not user.output.model_fields_set:
        return
    artifacts = build_artifacts(user)
    # Drop the auto-synthesised cli_command when only it would land --
    # leaves the YAML's value alone for users who pin it.
    if set(artifacts.keys()) - {"cli_command"}:
        out["artifacts"] = artifacts


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
