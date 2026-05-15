# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a ``CLIConfig`` + optional YAML ``--config`` file into an
``AIPerfConfig``.

Used by every CLI command that supports both flag-form and file-form input
(``aiperf profile`` and ``aiperf service``). When both are supplied, the YAML
supplies the base configuration and any explicitly-set CLI flags on
``cli_config`` are deep-merged on top before AIPerfConfig validation -- so
``aiperf profile --config foo.yaml --search-recipe X --ttft-sla-ms 200``
works the way users intuit instead of throwing
``CLIConfig.endpoint.modelNames: Field required``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from aiperf.config.flags._section_fields import (
    ENDPOINT_FIELDS,
    INPUT_FIELDS,
    LOADGEN_FIELDS,
    OUTPUT_FIELDS,
    SWEEPING_FIELDS,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config import AIPerfConfig
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.flags import CLIConfig


def resolve_config(
    cli_config: CLIConfig,
    config_file: Path | None = None,
) -> AIPerfConfig:
    """Return an `AIPerfConfig` from a YAML config file and/or CLI flags.

    Args:
        cli_config: Parsed ``CLIConfig`` carrying flag-form benchmark and
            service-runtime options.
        config_file: Optional path to a YAML config file. Defaults to
            ``cli_config.config_file`` when not explicitly provided. When
            provided, the YAML supplies the base configuration and any
            explicitly-set CLI flags on ``cli_config`` are deep-merged on
            top before validation. Without ``config_file``, the
            CLIConfig -> AIPerfConfig converter handles the full CLI-only path.

    Returns:
        Fully resolved `AIPerfConfig` ready for downstream use.
    """
    from aiperf.config.flags.converter import (
        _promote_cli_dataset_magic_lists,
        _promote_magic_lists_to_sweep_block,
        _wrap_under_envelope,
        convert_cli_to_aiperf,
    )

    if config_file is None:
        config_file = cli_config.config_file

    if config_file is None:
        return convert_cli_to_aiperf(cli_config)

    from aiperf.config import AIPerfConfig
    from aiperf.config.loader import load_config_dict

    yaml_dict = load_config_dict(config_file)
    # Build the recipe's view of BenchmarkConfig from YAML + the
    # endpoint/input CLI overrides ONLY: the recipe inspects fields like
    # ``endpoint.streaming`` (via ``require_streaming``) before emitting
    # streaming-only metric recipes, so feeding it an unmerged YAML config
    # rejects ``-f base.yaml --search-recipe prefill-ttft-curve --streaming``
    # whenever ``base.yaml`` has ``streaming: false``. Building only the
    # endpoint/input overlay (no recipe / no sweep) keeps this preliminary
    # validation cheap and avoids a chicken-and-egg dependency on the
    # recipe's own outputs.
    pre_overrides: dict[str, Any] = {}
    _apply_endpoint_overrides(pre_overrides, cli_config)
    _apply_input_overrides(pre_overrides, cli_config)
    pre_merged = (
        deep_merge(yaml_dict, _wrap_under_envelope(copy.deepcopy(pre_overrides)))
        if pre_overrides
        else yaml_dict
    )
    base_config = AIPerfConfig.model_validate(pre_merged)

    overrides = build_cli_overrides(cli_config, benchmark_config=base_config.benchmark)
    if overrides:
        overrides = _wrap_under_envelope(overrides)
    merged = deep_merge(yaml_dict, overrides) if overrides else yaml_dict
    _apply_phase_loadgen_overrides(merged, cli_config)
    benchmark = merged.get("benchmark")
    if isinstance(benchmark, dict):
        sweep_type = getattr(cli_config, "sweep_type", "grid")
        _promote_cli_dataset_magic_lists(benchmark, cli_config, sweep_type=sweep_type)
        _retarget_dataset_magic_lists(benchmark)
        _promote_magic_lists_to_sweep_block(benchmark, sweep_type=sweep_type)
        promoted_sweep = benchmark.pop("sweep", None)
        if isinstance(promoted_sweep, dict):
            existing_sweep = merged.get("sweep")
            if isinstance(existing_sweep, dict):
                existing_sweep.setdefault(
                    "type", promoted_sweep.get("type", sweep_type)
                )
                existing_sweep.setdefault("parameters", {})
                existing_sweep["parameters"].update(
                    promoted_sweep.get("parameters", {})
                )
            else:
                merged["sweep"] = promoted_sweep
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


def build_cli_overrides(
    cli: CLIConfig,
    *,
    benchmark_config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Translate explicitly-set CLI flags into an AIPerfConfig-shape override dict.

    Only fields the user explicitly set (per nested model's
    ``model_fields_set``) flow through; everything else is left for the YAML
    base to supply. Reuses the converter's section-builders for endpoint /
    multi-run / tokenizer / accuracy / runtime / logging so the YAML+CLI path
    produces identical AIPerfConfig shape to the CLI-only path for the same
    inputs.

    Returns an empty dict when the user passed no CLI overrides; callers
    short-circuit the deep-merge in that case.
    """
    from aiperf.config.flags._converter_optionals import (
        build_accuracy,
        build_tokenizer,
    )
    from aiperf.config.flags._converter_runtime import build_logging_runtime

    out: dict[str, Any] = {}
    _apply_endpoint_overrides(out, cli)
    _apply_input_overrides(out, cli)
    _apply_recipe_and_multirun(out, cli, benchmark_config=benchmark_config)
    _apply_artifacts_overrides(out, cli)
    _apply_optional_section(out, "tokenizer", build_tokenizer(cli))
    _apply_optional_section(out, "accuracy", build_accuracy(cli))

    if "no_sweep_table" in cli.model_fields_set:
        out["no_sweep_table"] = cli.no_sweep_table

    # Service-runtime CLI flags (--ui, --log-level, --verbose, ZMQ knobs)
    # land on RuntimeConfig / LoggingConfig in AIPerfConfig. build_logging_runtime
    # already gates on cli.model_fields_set, so YAML defaults stay
    # intact when the user didn't pass these flags.
    logging_dict, runtime_dict = build_logging_runtime(cli)
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


def _apply_recipe_and_multirun(
    out: dict[str, Any],
    cli: CLIConfig,
    *,
    benchmark_config: BenchmarkConfig | None,
) -> None:
    """Recipes drive multi_run / sweep / sla_filters; reuse the converter
    path so YAML+CLI emits the same shape as CLI-only."""
    from aiperf.config.flags._converter_optionals import (
        build_multi_run,
        build_sweep,
        expand_search_recipe,
    )

    if benchmark_config is None:
        recipe_output = None
    else:
        recipe_output = expand_search_recipe(cli, benchmark_config=benchmark_config)
    if recipe_output is not None:
        sweep_params = recipe_output.get("sweep_parameters")
        if sweep_params:
            out["sweep"] = {"type": "grid", "parameters": dict(sweep_params)}
        # Recipe-emitted per-request SLOs (e.g. MaxGoodputUnderSLO) land on the
        # body's `slos` block. The envelope wrapper (`_wrap_under_envelope`) is
        # applied in `resolve_config` after this builder, so we write the body
        # path here -- ``benchmark.slos`` after wrapping.
        recipe_slos = recipe_output.get("slos")
        if recipe_slos:
            out["slos"] = dict(recipe_slos)
    sweep = build_sweep(cli, recipe_output=recipe_output)
    if sweep:
        # ``build_sweep`` returns a sweep envelope without ``parameters`` for
        # grid recipes (only ``sla_filters`` / ``post_process`` metadata) --
        # merge those keys onto whatever ``recipe_output["sweep_parameters"]``
        # already wrote into ``out["sweep"]`` instead of replacing it
        # wholesale, so the recipe's parameters don't get clobbered by the
        # metadata-only build_sweep result.
        existing = out.get("sweep")
        if isinstance(existing, dict) and isinstance(sweep, dict):
            for key, value in sweep.items():
                existing.setdefault(key, value)
        else:
            out["sweep"] = sweep
    multi_run = build_multi_run(cli, recipe_output=recipe_output)
    if multi_run:
        out["multi_run"] = multi_run


def _apply_artifacts_overrides(out: dict[str, Any], cli: CLIConfig) -> None:
    """Map ``--artifact-dir`` and friends to the ``artifacts`` block.

    Only emits the block when the user actually set one of the flattened output
    fields, so a YAML ``artifacts.dir`` stays untouched on a plain
    ``aiperf profile -f base.yaml`` invocation.

    Auto-plot resolution layers on top: when the user passed an explicit
    ``--auto-plot``/``--no-auto-plot`` flag OR a CLI ``--search-recipe``
    that defines an ``auto_plot_default``, the resolved bool is written
    into the artifacts override so it overlays the YAML.
    """
    from aiperf.config.flags._converter_optionals import resolve_auto_plot
    from aiperf.config.flags._converter_runtime import build_artifacts

    output_set = cli.model_fields_set & OUTPUT_FIELDS
    sweeping_set = cli.model_fields_set & SWEEPING_FIELDS

    artifacts: dict[str, Any] = {}
    if output_set:
        built = build_artifacts(cli)
        if built:
            artifacts.update(built)

    explicit_auto_plot = "auto_plot" in output_set
    explicit_plot_required = "plot_required" in output_set
    has_cli_recipe = "search_recipe" in sweeping_set and cli.search_recipe is not None
    if explicit_auto_plot or explicit_plot_required or has_cli_recipe:
        auto_plot, plot_required = resolve_auto_plot(cli)
        if explicit_auto_plot or has_cli_recipe:
            artifacts["auto_plot"] = auto_plot
        if explicit_plot_required:
            artifacts["plot_required"] = plot_required

    if artifacts:
        out["artifacts"] = artifacts


def _retarget_dataset_magic_lists(benchmark: dict[str, Any]) -> None:
    sweep = benchmark.get("sweep")
    if not isinstance(sweep, dict):
        return
    parameters = sweep.get("parameters")
    if not isinstance(parameters, dict):
        return
    dataset_name = _single_dataset_name(benchmark)
    if dataset_name is None or dataset_name == "main":
        return
    for path in list(parameters):
        if path.startswith("datasets.main."):
            parameters[
                f"datasets.{dataset_name}.{path.removeprefix('datasets.main.')}"
            ] = parameters.pop(path)


def _single_dataset_name(benchmark: dict[str, Any]) -> str | None:
    datasets = benchmark.get("datasets")
    if isinstance(datasets, list) and len(datasets) == 1:
        entry = datasets[0]
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            return entry["name"]
    dataset = benchmark.get("dataset")
    if isinstance(dataset, dict):
        return "default"
    return None


def _apply_endpoint_overrides(out: dict[str, Any], cli: CLIConfig) -> None:
    """Translate explicitly-set endpoint flags into ``out['endpoint']`` and
    ``out['models']``.

    ``--model-names`` lives on the CLIConfig endpoint section but maps to the
    ``models.items`` block on AIPerfConfig; everything else stays on ``endpoint``.
    """
    from aiperf.config.flags._converter_endpoint import _ENDPOINT_FIELD_MAP

    ep_set = cli.model_fields_set & ENDPOINT_FIELDS
    if not ep_set:
        return
    endpoint: dict[str, Any] = {}
    if "urls" in ep_set:
        endpoint["urls"] = list(cli.urls)
    for cli_field, aiperf_key in _ENDPOINT_FIELD_MAP.items():
        if cli_field in ep_set:
            endpoint[aiperf_key] = getattr(cli, cli_field)
    if endpoint:
        out["endpoint"] = endpoint
    if "model_names" in ep_set and cli.model_names:
        models: dict[str, Any] = {"items": [{"name": name} for name in cli.model_names]}
        if "model_selection_strategy" in ep_set:
            models["strategy"] = cli.model_selection_strategy
        out["models"] = models


def _apply_input_overrides(out: dict[str, Any], cli: CLIConfig) -> None:
    """Mirror ``build_endpoint``'s rule that ``--headers`` / ``--extra`` (which
    live on the input section of CLIConfig) flow into the AIPerfConfig
    ``endpoint`` block.
    """
    inp_set = cli.model_fields_set & INPUT_FIELDS
    if not inp_set:
        return
    endpoint = out.setdefault("endpoint", {})
    if "headers" in inp_set and cli.headers:
        endpoint["headers"] = dict(cli.headers)
    if "extra_inputs" in inp_set and cli.extra_inputs:
        endpoint["extra"] = dict(cli.extra_inputs)
    if not endpoint:
        out.pop("endpoint", None)


# CLI loadgen flag -> phase field. Each entry is (loadgen_attr, phase_key).
# The CLI help promises "CLI flags override values from the config file";
# this table makes that real for YAML-supplied phase shapes by overlaying
# the explicit CLI value onto the resolved profiling phase.
_LOADGEN_PHASE_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("request_count", "requests"),
    ("benchmark_duration", "duration"),
    ("benchmark_grace_period", "grace_period"),
    ("concurrency", "concurrency"),
    ("prefill_concurrency", "prefill_concurrency"),
    ("request_rate", "rate"),
    ("user_centric_rate", "rate"),
    ("num_users", "users"),
)


def _apply_phase_loadgen_overrides(merged: dict[str, Any], cli: CLIConfig) -> None:
    """Overlay explicit ``--request-count`` / ``--request-rate`` / etc. onto
    the YAML-supplied profiling phase.

    YAML configs land ``phases`` as a list under ``benchmark.phases``;
    ``deep_merge`` replaces lists wholesale, so the CLI flags otherwise
    silently no-op when the YAML already sets ``phases[*].requests``. This
    walks the merged envelope, finds the phase named ``profiling`` (or the
    sole phase entry if there's only one), and writes each user-set
    loadgen field onto it. Other phases (warmup) are left untouched so a
    user passing ``--request-count 10`` with ``warmup_profiling.yaml``
    doesn't clobber the warmup ramp.
    """
    fields_set = cli.model_fields_set & LOADGEN_FIELDS
    if not fields_set:
        return

    benchmark = merged.get("benchmark")
    if not isinstance(benchmark, dict):
        return
    phases = benchmark.get("phases")
    if not isinstance(phases, list) or not phases:
        return

    target = _find_profiling_phase(phases)
    if target is None:
        return

    _reject_loadgen_target_collisions(fields_set)

    for attr, key in _LOADGEN_PHASE_FIELD_MAP:
        if attr not in fields_set:
            continue
        value = getattr(cli, attr)
        if value is None:
            continue
        target[key] = value


def _reject_loadgen_target_collisions(fields_set: set[str]) -> None:
    """Raise when two distinct CLI source-attrs map to the same phase key.

    Without this guard, the second tuple in :data:`_LOADGEN_PHASE_FIELD_MAP`
    silently wins via dict assignment when both source-attrs are set (e.g.
    ``--request-rate`` and ``--user-centric-rate`` both write ``"rate"``).
    Two flags landing on the same key is always a user error.
    """
    collisions: dict[str, list[str]] = {}
    for attr, key in _LOADGEN_PHASE_FIELD_MAP:
        if attr in fields_set:
            collisions.setdefault(key, []).append(attr)
    duplicates = {k: v for k, v in collisions.items() if len(v) > 1}
    if not duplicates:
        return
    from aiperf.config.loader.errors import ConfigurationError

    details = "; ".join(
        f"{k!r} <- {sorted(attrs)}" for k, attrs in sorted(duplicates.items())
    )
    raise ConfigurationError(
        f"Mutually exclusive CLI loadgen flags target the same phase "
        f"key(s): {details}. Pass only one."
    )


def _find_profiling_phase(phases: list[Any]) -> dict[str, Any] | None:
    """Return the phase entry whose CLI loadgen flags should overlay onto.

    Prefers the phase named ``profiling``. If no phase has that name (e.g.
    a single-entry list authored without ``name``), returns the sole entry
    when there's exactly one. Otherwise returns ``None`` and the override
    silently no-ops -- matching the convention that ambiguous YAML wins.
    """
    profiling: dict[str, Any] | None = None
    non_warmup: list[dict[str, Any]] = []
    for entry in phases:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name == "profiling":
            profiling = entry
            break
        if name != "warmup":
            non_warmup.append(entry)
    if profiling is not None:
        return profiling
    if len(non_warmup) == 1:
        return non_warmup[0]
    return None
