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
import logging
from typing import TYPE_CHECKING, Any

from aiperf.common.enums import DatasetType
from aiperf.common.phase import infer_legacy_phase_kind
from aiperf.config.flags._resolver_gpu_telemetry import (
    build_gpu_telemetry_override,
    normalize_gpu_telemetry_base_for_override,
)
from aiperf.config.flags._resolver_helpers import promote_benchmark_magic_lists
from aiperf.config.flags._resolver_server_metrics import (
    build_server_metrics_override,
    normalize_server_metrics_base_for_override,
)
from aiperf.config.flags._section_fields import (
    ENDPOINT_FIELDS,
    INPUT_FIELDS,
    OUTPUT_FIELDS,
    SWEEPING_FIELDS,
)
from aiperf.plugin.enums import ArrivalPattern, PhaseType

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config import AIPerfConfig
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.flags import CLIConfig

logger = logging.getLogger(__name__)


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
    from aiperf.config.flags.converter import convert_cli_to_aiperf

    if config_file is None:
        config_file = cli_config.config_file

    if config_file is None:
        return convert_cli_to_aiperf(cli_config)

    from aiperf.config.loader import load_config_dict_with_raw_envelope

    yaml_dict, raw_yaml_dict = load_config_dict_with_raw_envelope(config_file)
    return _resolve_config_envelopes(cli_config, yaml_dict, raw_yaml_dict)


def _resolve_config_envelopes(
    cli_config: CLIConfig,
    yaml_dict: dict[str, Any],
    raw_yaml_dict: dict[str, Any],
) -> AIPerfConfig:
    """Resolve rendered and pre-Jinja envelopes through one override pipeline."""
    from aiperf.config import AIPerfConfig
    from aiperf.config.flags.converter import _wrap_under_envelope

    _normalize_loaded_benchmark_shorthands(yaml_dict)
    _normalize_loaded_benchmark_shorthands(raw_yaml_dict)
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
        else copy.deepcopy(yaml_dict)
    )
    base_config = AIPerfConfig.model_validate(pre_merged)
    dataset_override = _build_dataset_override(
        cli_config, benchmark_config=base_config.benchmark
    )
    _apply_dataset_override(pre_merged, dataset_override)
    if dataset_override is not None:
        base_config = AIPerfConfig.model_validate(pre_merged)

    overrides = build_cli_overrides(cli_config, benchmark_config=base_config.benchmark)
    overrides = _wrap_under_envelope(overrides) if overrides else overrides
    merged = _merge_overrides_into_envelope(
        yaml_dict, overrides, cli_config, dataset_override=dataset_override
    )
    raw_merged = _merge_overrides_into_envelope(
        raw_yaml_dict,
        overrides,
        cli_config,
        dataset_override=dataset_override,
    )

    config = AIPerfConfig.model_validate(merged)
    config._raw_envelope = raw_merged
    return config


def _merge_overrides_into_envelope(
    envelope: dict[str, Any],
    overrides: dict[str, Any] | None,
    cli_config: CLIConfig,
    *,
    dataset_override: tuple[bool, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the config-file CLI override pipeline to one envelope.

    The resolver calls this once for the rendered envelope used for Pydantic
    validation and once for the retained pre-Jinja envelope used by sweep
    expansion. Keeping both transformations identical prevents CLI overrides
    and Jinja-backed ``sweep.parameters`` from disagreeing at execution time.
    """
    from aiperf.config.flags.converter import (
        _promote_cli_dataset_magic_lists,
        _promote_magic_lists_to_sweep_block,
    )

    overrides = copy.deepcopy(overrides) if overrides else overrides
    envelope = normalize_gpu_telemetry_base_for_override(envelope, overrides)
    envelope = normalize_server_metrics_base_for_override(envelope, overrides)
    merged = deep_merge(envelope, overrides) if overrides else envelope
    _apply_control_hook_enable_overrides(merged, cli_config)
    _apply_dataset_override(merged, dataset_override)
    _apply_dataset_synthesis_overrides(merged, cli_config)
    _apply_dataset_filter_overrides(merged, cli_config)
    _apply_phase_loadgen_overrides(merged, cli_config)
    promote_benchmark_magic_lists(
        merged,
        cli_config,
        promote_cli_dataset_magic_lists=_promote_cli_dataset_magic_lists,
        promote_magic_lists_to_sweep_block=_promote_magic_lists_to_sweep_block,
        retarget_dataset_magic_lists=_retarget_dataset_magic_lists,
    )
    _coalesce_phase_aliases(merged)
    return merged


def _normalize_loaded_benchmark_shorthands(yaml_dict: dict[str, Any]) -> None:
    """Normalize YAML benchmark shorthands before raw-dict CLI overlays.

    ``AIPerfConfig.model_validate`` already accepts conveniences such as
    ``model:``, ``dataset:``, and single-dict ``phases: {type: ...}``, but
    resolver overlay helpers inspect the loaded YAML dict before final
    validation. Normalizing once here gives those helpers the same canonical
    shape while preserving CLI-over-YAML precedence.
    """
    from aiperf.config.loader.normalizers import normalize_benchmark_input

    benchmark = yaml_dict.get("benchmark")
    if isinstance(benchmark, dict):
        yaml_dict["benchmark"] = normalize_benchmark_input(benchmark)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; non-dict values replace.

    Lists are replaced wholesale (not concatenated) so that a CLI override
    list cleanly clobbers a YAML list rather than appending.

    An empty-dict override also replaces rather than recursing: ``--header``
    with no value, ``--extra`` with no value and an empty ``--goodput`` all
    mean "this section is empty", not "leave the YAML alone". Producers that
    need "enable but inherit the YAML sub-fields" (the ``--reset-kv-cache`` /
    ``--server-profiler`` bare booleans) cannot be expressed through this
    function at all and are applied post-merge by
    :func:`_apply_control_hook_enable_overrides`.
    """
    from pydantic.alias_generators import to_camel

    out = copy.deepcopy(base)
    for key, value in override.items():
        target_key = key
        alias = to_camel(key)
        if key not in out and alias in out:
            target_key = alias
        if isinstance(value, dict) and isinstance(out.get(target_key), dict) and value:
            out[target_key] = deep_merge(out[target_key], value)
        else:
            out[target_key] = value
    return out


def _coalesce_phase_aliases(envelope: dict[str, Any]) -> None:
    """Make phase overlays win over equivalent camelCase YAML keys."""
    from pydantic.alias_generators import to_camel

    benchmark = envelope.get("benchmark")
    phases = benchmark.get("phases") if isinstance(benchmark, dict) else None
    if not isinstance(phases, list):
        return
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        for key in list(phase):
            if "_" not in key:
                continue
            alias = to_camel(key)
            if alias not in phase:
                continue
            snake_value = phase.pop(key)
            camel_value = phase[alias]
            phase[alias] = (
                deep_merge(camel_value, snake_value)
                if isinstance(camel_value, dict)
                and isinstance(snake_value, dict)
                and snake_value
                else snake_value
            )


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
    from aiperf.config.flags._converter_telemetry import (
        build_mlflow,
        build_network_latency,
        build_otel,
        build_wandb,
    )

    out: dict[str, Any] = {}
    _apply_endpoint_overrides(out, cli)
    _apply_input_overrides(out, cli)
    _apply_recipe_and_multirun(out, cli, benchmark_config=benchmark_config)
    _apply_artifacts_overrides(out, cli)
    _apply_optional_section(out, "gpu_telemetry", build_gpu_telemetry_override(cli))
    _apply_optional_section(out, "server_metrics", build_server_metrics_override(cli))
    _apply_optional_section(out, "tokenizer", build_tokenizer(cli))
    _apply_optional_section(out, "accuracy", build_accuracy(cli))
    telemetry_fields = cli.model_fields_set
    if telemetry_fields & {
        "network_latency_automatic",
        "network_latency_mean",
        "network_latency_ping_interval",
    }:
        out["network_latency"] = build_network_latency(cli)
    if telemetry_fields & {
        "otel_url",
        "stream",
        "otel_resource_attributes",
        "gen_ai_provider",
    }:
        otel_cli = _hydrate_primary_from_yaml(
            cli,
            primary_field="otel_url",
            base_value=(
                getattr(benchmark_config.otel, "metrics_url", None)
                if benchmark_config is not None
                else None
            ),
        )
        _apply_optional_section(out, "otel", build_otel(otel_cli))
    if telemetry_fields & {
        "mlflow_tracking_uri",
        "mlflow_experiment",
        "mlflow_run_name",
        "mlflow_tags",
        "mlflow_parent_run_id",
        "mlflow_artifact_globs",
    }:
        mlflow_cli = _hydrate_primary_from_yaml(
            cli,
            primary_field="mlflow_tracking_uri",
            base_value=(
                getattr(benchmark_config.mlflow, "tracking_uri", None)
                if benchmark_config is not None
                else None
            ),
        )
        _apply_optional_section(out, "mlflow", build_mlflow(mlflow_cli))
    wandb_base_enabled = benchmark_config is not None and benchmark_config.wandb.enabled
    _apply_optional_section(
        out, "wandb", build_wandb(cli, base_enabled=wandb_base_enabled)
    )

    if "no_sweep_table" in cli.model_fields_set:
        out["no_sweep_table"] = cli.no_sweep_table
    if "random_seed" in cli.model_fields_set:
        out["random_seed"] = cli.random_seed
    if "goodput" in cli.model_fields_set:
        out["slos"] = dict(cli.goodput or {})
    if "scenario" in cli.model_fields_set:
        out["scenario"] = cli.scenario
    if "unsafe_override" in cli.model_fields_set:
        out["unsafe_override"] = cli.unsafe_override

    # Service-runtime CLI flags (--ui, --log-level, --verbose, ZMQ knobs)
    # land on RuntimeConfig / LoggingConfig in AIPerfConfig. build_logging_runtime
    # already gates on cli.model_fields_set, so YAML defaults stay
    # intact when the user didn't pass these flags.
    logging_dict, runtime_dict = build_logging_runtime(cli)
    _apply_optional_section(out, "logging", logging_dict)
    _apply_optional_section(out, "runtime", runtime_dict)

    return out


def _hydrate_primary_from_yaml(
    cli: CLIConfig,
    *,
    primary_field: str,
    base_value: Any,
) -> CLIConfig:
    """Let secondary telemetry flags reuse a primary configured in YAML."""
    if primary_field in cli.model_fields_set or base_value is None:
        return cli
    hydrated = cli.model_copy(deep=True)
    setattr(hydrated, primary_field, str(base_value))
    return hydrated


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
    ep_set = cli.model_fields_set & ENDPOINT_FIELDS
    if not ep_set:
        return

    endpoint = _build_endpoint_override(cli, ep_set)
    if endpoint:
        out["endpoint"] = endpoint

    models = _build_model_override(cli, ep_set)
    if models:
        out["models"] = models


def _build_endpoint_override(cli: CLIConfig, fields_set: set[str]) -> dict[str, Any]:
    from aiperf.config.flags._converter_endpoint import _ENDPOINT_FIELD_MAP

    endpoint: dict[str, Any] = {}
    if "urls" in fields_set:
        endpoint["urls"] = list(cli.urls)
    for cli_field, aiperf_key in _ENDPOINT_FIELD_MAP.items():
        if cli_field in fields_set:
            endpoint[aiperf_key] = getattr(cli, cli_field)

    _apply_reset_kv_cache_override(endpoint, cli, fields_set)
    _apply_server_profiler_override(endpoint, cli, fields_set)
    return endpoint


def _apply_reset_kv_cache_override(
    endpoint: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    reset_fields = {
        "reset_kv_cache",
        "reset_kv_cache_path",
        "reset_kv_cache_timeout_seconds",
    }
    if not fields_set & reset_fields:
        return
    if "reset_kv_cache" in fields_set and not cli.reset_kv_cache:
        endpoint["reset_kv_cache"] = False
        return

    from aiperf.config.flags._converter_endpoint import _maybe_build_reset_kv_cache

    # A bare ``--reset-kv-cache`` builds no sub-fields, and an empty dict here
    # would wipe a YAML-supplied path/timeout on merge. That case is enabled
    # post-merge instead; see _apply_control_hook_enable_overrides.
    if sub_fields := _maybe_build_reset_kv_cache(cli):
        endpoint["reset_kv_cache"] = sub_fields


def _apply_server_profiler_override(
    endpoint: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    profiler_fields = {
        "server_profiler",
        "server_profiler_start_path",
        "server_profiler_stop_path",
        "server_profiler_timeout_seconds",
    }
    if not fields_set & profiler_fields:
        return
    if "server_profiler" in fields_set and not cli.server_profiler:
        endpoint["server_profiler"] = False
        return

    from aiperf.config.flags._converter_endpoint import _maybe_build_server_profiler

    if sub_fields := _maybe_build_server_profiler(cli):
        endpoint["server_profiler"] = sub_fields


# Control hooks whose bare boolean CLI flag means "enable, but inherit whatever
# the YAML already configured". Each entry is
# (cli_flag_attr, endpoint_key, cli_sub_field_attrs).
_CONTROL_HOOK_ENABLE_FLAGS: tuple[tuple[str, str, frozenset[str]], ...] = (
    (
        "reset_kv_cache",
        "reset_kv_cache",
        frozenset({"reset_kv_cache_path", "reset_kv_cache_timeout_seconds"}),
    ),
    (
        "server_profiler",
        "server_profiler",
        frozenset(
            {
                "server_profiler_start_path",
                "server_profiler_stop_path",
                "server_profiler_timeout_seconds",
            }
        ),
    ),
)


def _apply_control_hook_enable_overrides(
    merged: dict[str, Any], cli: CLIConfig
) -> None:
    """Enable ``--reset-kv-cache`` / ``--server-profiler`` without clobbering YAML.

    These flags accept ``false | true | {sub-fields}``. A bare boolean flag
    carries no sub-fields, so it cannot be expressed as a deep-merge override:
    an empty dict replaces the YAML section (see :func:`deep_merge`) and a
    literal ``True`` replaces it too, either way discarding a user-authored
    ``path`` / ``start_path`` / ``timeout_seconds``. Running post-merge lets the
    overlay see the YAML value and leave an already-configured mapping alone,
    since a mapping already means "enabled".
    """
    from pydantic.alias_generators import to_camel

    fields_set = cli.model_fields_set
    benchmark = merged.get("benchmark")
    if not isinstance(benchmark, dict):
        return
    for flag_attr, endpoint_key, sub_field_attrs in _CONTROL_HOOK_ENABLE_FLAGS:
        if flag_attr not in fields_set or not getattr(cli, flag_attr):
            continue
        if fields_set & sub_field_attrs:
            continue
        endpoint = benchmark.setdefault("endpoint", {})
        if not isinstance(endpoint, dict):
            return
        target_key = endpoint_key
        alias = to_camel(endpoint_key)
        if endpoint_key not in endpoint and alias in endpoint:
            target_key = alias
        current = endpoint.get(target_key)
        if isinstance(current, dict) and current:
            continue
        endpoint[target_key] = True


def _build_model_override(cli: CLIConfig, fields_set: set[str]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    if "model_names" in fields_set:
        models["items"] = [{"name": name} for name in cli.model_names]
    if "model_selection_strategy" in fields_set:
        models["strategy"] = cli.model_selection_strategy
    return models


def _apply_input_overrides(out: dict[str, Any], cli: CLIConfig) -> None:
    """Mirror ``build_endpoint``'s rule that ``--headers`` / ``--extra`` (which
    live on the input section of CLIConfig) flow into the AIPerfConfig
    ``endpoint`` block.
    """
    inp_set = cli.model_fields_set & INPUT_FIELDS
    if not inp_set:
        return
    endpoint = out.setdefault("endpoint", {})
    if "headers" in inp_set:
        endpoint["headers"] = dict(cli.headers)
    if "extra_inputs" in inp_set:
        endpoint["extra"] = dict(cli.extra_inputs)
    if not endpoint:
        out.pop("endpoint", None)


_DATASET_OVERRIDE_FIELDS: frozenset[str] = frozenset(
    {
        "allow_dataset_wrap",
        "audio_batch_size",
        "audio_depths",
        "audio_format",
        "audio_length_mean",
        "audio_length_stddev",
        "audio_num_channels",
        "audio_sample_rates",
        "cache_bust",
        "conversation_num",
        "conversation_num_dataset_entries",
        "conversation_turn_delay_mean",
        "conversation_turn_delay_ratio",
        "conversation_turn_delay_stddev",
        "conversation_turn_mean",
        "conversation_turn_stddev",
        "custom_dataset_type",
        "dataset_sampling_strategy",
        "force_min_tokens",
        "hf_dataset_subset",
        "hf_weka_dataset",
        "ignore_trace_delays",
        "image_batch_size",
        "image_format",
        "image_height_mean",
        "image_height_stddev",
        "image_source",
        "image_source_sampling",
        "image_width_mean",
        "image_width_stddev",
        "input_file",
        "inter_turn_delay_cap_seconds",
        "max_context_length",
        "max_idle_gap_cap_seconds",
        "omit_kv_hints",
        "open_loop_replay",
        "open_loop_strict",
        "prompt_batch_size",
        "prompt_corpus",
        "prompt_input_tokens_block_size",
        "prompt_input_tokens_mean",
        "prompt_input_tokens_stddev",
        "prompt_output_tokens_mean",
        "prompt_output_tokens_stddev",
        "prompt_prefix_length",
        "prompt_prefix_pool_size",
        "prompt_prefix_shared_system_length",
        "prompt_prefix_user_context_length",
        "prompt_sequence_distribution",
        "public_dataset",
        "random_seed",
        "rankings_passages_mean",
        "rankings_passages_prompt_token_mean",
        "rankings_passages_prompt_token_stddev",
        "rankings_passages_stddev",
        "rankings_query_prompt_token_mean",
        "rankings_query_prompt_token_stddev",
        "replay_speedup",
        "synthesis_max_isl",
        "synthesis_max_osl",
        "synthesis_output_len_multiplier",
        "synthesis_prefix_len_multiplier",
        "synthesis_prefix_root_multiplier",
        "synthesis_prompt_len_multiplier",
        "synthesis_speedup_ratio",
        "trace_idle_gap_cap_seconds",
        "trace_session_sample_ratio",
        "use_think_time_only",
        "video_audio_channels",
        "video_audio_codec",
        "video_audio_depth",
        "video_audio_sample_rate",
        "video_batch_size",
        "video_codec",
        "video_duration",
        "video_format",
        "video_fps",
        "video_height",
        "video_synth_type",
        "video_width",
    }
)

_DATASET_REPLACEMENT_FIELDS: frozenset[str] = frozenset(
    {"input_file", "public_dataset", "hf_weka_dataset"}
)


def _build_dataset_override(
    cli: CLIConfig,
    *,
    benchmark_config: BenchmarkConfig,
) -> tuple[bool, dict[str, Any]] | None:
    """Build a sole-dataset CLI overlay without importing CLI-only defaults.

    Config v2 currently requires exactly one dataset, so an explicit dataset
    flag has one unambiguous target. Source-selection flags replace that entry;
    modifier flags deep-merge into it. The converter is hydrated with the YAML
    dataset source without marking those values explicit, allowing its existing
    file/public validation and canonical field routing to be reused.
    """
    fields_set = cli.model_fields_set & _DATASET_OVERRIDE_FIELDS
    if not fields_set:
        return None

    from aiperf.common.enums import DatasetType
    from aiperf.config.flags._converter_dataset import build_dataset

    replacement = bool(fields_set & _DATASET_REPLACEMENT_FIELDS)
    dataset_cli = cli.model_copy(deep=True)
    dataset_cli.__pydantic_fields_set__.discard("dataset_filters")
    object.__setattr__(dataset_cli, "dataset_filters", [])
    base_dataset = benchmark_config.datasets[0]
    if "endpoint_type" not in cli.model_fields_set:
        object.__setattr__(dataset_cli, "endpoint_type", benchmark_config.endpoint.type)
    if not replacement:
        if base_dataset.type == DatasetType.FILE:
            object.__setattr__(dataset_cli, "input_file", base_dataset.path)
            if "custom_dataset_type" not in fields_set:
                object.__setattr__(
                    dataset_cli, "custom_dataset_type", base_dataset.format
                )
        elif base_dataset.type == DatasetType.PUBLIC:
            object.__setattr__(dataset_cli, "public_dataset", base_dataset.dataset)
            if "hf_dataset_subset" not in fields_set:
                object.__setattr__(
                    dataset_cli, "hf_dataset_subset", base_dataset.hf_subset
                )
            object.__setattr__(
                dataset_cli, "hf_weka_dataset", base_dataset.hf_weka_dataset
            )

    patch = build_dataset(dataset_cli)
    if replacement:
        return True, patch

    # These are CLI-only construction defaults, not values the user selected.
    patch.pop("type", None)
    if not {"conversation_num", "conversation_num_dataset_entries"} & fields_set:
        patch.pop("entries", None)
        patch.pop("_entries_explicit", None)
    _drop_implicit_dataset_defaults(patch, fields_set)
    return (False, patch) if patch else None


def _drop_implicit_dataset_defaults(
    patch: dict[str, Any], fields_set: set[str]
) -> None:
    """Remove converter defaults so omitted CLI values preserve YAML intent."""
    prompts = patch.get("prompts")
    if isinstance(prompts, dict):
        isl = prompts.get("isl")
        if isinstance(isl, dict) and "prompt_input_tokens_mean" not in fields_set:
            isl.pop("mean", None)
            if not isl:
                prompts.pop("isl", None)
        if not prompts:
            patch.pop("prompts", None)

    for section, explicit_batch_field in (
        ("audio", "audio_batch_size"),
        ("images", "image_batch_size"),
        ("video", "video_batch_size"),
    ):
        value = patch.get(section)
        if isinstance(value, dict) and explicit_batch_field not in fields_set:
            value.pop("batch_size", None)
            if not value:
                patch.pop(section, None)


def _apply_dataset_override(
    merged: dict[str, Any],
    dataset_override: tuple[bool, dict[str, Any]] | None,
) -> None:
    """Apply a prepared override to the config's sole named dataset."""
    if dataset_override is None:
        return
    benchmark = merged.get("benchmark")
    datasets = benchmark.get("datasets") if isinstance(benchmark, dict) else None
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("CLI dataset flags require exactly one YAML dataset")
    target = datasets[0]
    if not isinstance(target, dict):
        raise ValueError("CLI dataset flags require a mapping-shaped YAML dataset")
    replacement, patch = dataset_override
    if replacement:
        name = target.get("name", "main")
        target.clear()
        target.update({"name": name, **copy.deepcopy(patch)})
        return
    target.update(deep_merge(target, patch))


def _apply_dataset_filter_overrides(merged: dict[str, Any], cli: CLIConfig) -> None:
    if "dataset_filters" not in cli.model_fields_set:
        return

    from aiperf.config.flags._converter_dataset import _parse_dataset_filters

    benchmark = merged.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("--dataset-filter requires a public dataset")
    dataset = benchmark.get("dataset")
    if not isinstance(dataset, dict):
        datasets = benchmark.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("--dataset-filter requires a public dataset")
        if len(datasets) > 1:
            logger.warning(
                "--dataset-filter with multiple YAML datasets applies only to "
                "the first dataset"
            )
        dataset = datasets[0]
    if not isinstance(dataset, dict):
        raise ValueError("--dataset-filter requires a public dataset")
    if dataset.get("type") != DatasetType.PUBLIC:
        raise ValueError("--dataset-filter requires a public dataset")
    filters = dataset.setdefault("filters", {})
    filters.update(_parse_dataset_filters(cli.dataset_filters))


def _apply_dataset_synthesis_overrides(merged: dict[str, Any], cli: CLIConfig) -> None:
    """Overlay explicit synthesis flags onto a YAML-supplied dataset."""
    if not any(
        field == "allow_dataset_wrap" or field.startswith("synthesis_")
        for field in cli.model_fields_set
    ):
        return

    from aiperf.config.dataset.trace import SynthesisConfig
    from aiperf.config.flags._converter_dataset import (
        _apply_synthesis,
        _reject_baseten_trace_unsupported_synthesis,
    )

    benchmark = merged.get("benchmark")
    datasets = benchmark.get("datasets") if isinstance(benchmark, dict) else None
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("synthesis flags require a file or public dataset")
    if len(datasets) > 1:
        logger.warning(
            "Synthesis flags with multiple YAML datasets apply only to the first dataset"
        )
    dataset = datasets[0]
    if not isinstance(dataset, dict):
        raise ValueError("synthesis flags require a file or public dataset")
    if dataset.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
        logger.warning(
            "Synthesis flags require a file or public dataset; ignoring them "
            "for dataset type %r",
            dataset.get("type"),
        )
        return

    if dataset.get("type") == DatasetType.FILE:
        _reject_baseten_trace_unsupported_synthesis(
            cli,
            dataset.get("format"),
            dataset_format_source="YAML format: baseten_trace",
        )

    override = {"type": dataset.get("type")}
    _apply_synthesis(override, cli)
    if synthesis := override.get("synthesis"):
        base = SynthesisConfig.model_validate(
            dataset.get("synthesis") or {}
        ).model_dump(by_alias=True, exclude_unset=True)
        update = SynthesisConfig.model_validate(synthesis).model_dump(
            by_alias=True, exclude_unset=True
        )
        dataset["synthesis"] = deep_merge(base, update)


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
    ("conversation_num", "sessions"),
)

_PROFILING_PHASE_OVERRIDE_FIELDS: frozenset[str] = frozenset(
    {attr for attr, _ in _LOADGEN_PHASE_FIELD_MAP}
    | {
        "arrival_pattern",
        "arrival_smoothness",
        "concurrency_ramp_duration",
        "fixed_schedule",
        "fixed_schedule_auto_offset",
        "fixed_schedule_end_offset",
        "fixed_schedule_start_offset",
        "prefill_concurrency_ramp_duration",
        "request_cancellation_delay",
        "request_cancellation_rate",
        "request_rate_ramp_duration",
        "request_rate_series",
    }
)


def _apply_phase_loadgen_overrides(merged: dict[str, Any], cli: CLIConfig) -> None:
    """Overlay explicit ``--request-count`` / ``--request-rate`` / etc. onto
    the YAML-supplied profiling phase.

    YAML configs land ``phases`` as a list under ``benchmark.phases``;
    ``deep_merge`` replaces lists wholesale, so the CLI flags otherwise
    silently no-op when the YAML already sets ``phases[*].requests``. This
    walks the merged envelope, finds the unique profiling-kind phase, and
    writes each user-set loadgen field onto it. Other phases (warmup) are
    left untouched so a
    user passing ``--request-count 10`` with ``warmup_profiling.yaml``
    doesn't clobber the warmup ramp.

    The AGENTIC_REPLAY phase fields (``--agentic-cache-warmup-duration``,
    ``--burst-phase-starts``, ``--failed-request-threshold``,
    ``--trajectory-start-min/max-ratio``) live on ``BasePhaseConfig`` and
    are overlaid via the same converter helper the CLI-only path uses, so a
    ``-f scenario.yaml --agentic-cache-warmup-duration 30`` honors the
    documented "CLI flags override values from the config file" contract.
    """
    from aiperf.config.flags._converter_profiling import (
        _AGENTIC_REPLAY_ROUTES,
        _apply_agentic_replay_fields,
    )

    loadgen_set = cli.model_fields_set & _PROFILING_PHASE_OVERRIDE_FIELDS
    agentic_set = cli.model_fields_set.intersection(_AGENTIC_REPLAY_ROUTES)
    if not loadgen_set and not agentic_set:
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

    _reject_loadgen_target_collisions(loadgen_set)
    _apply_rate_series_override(target, cli, loadgen_set)
    _apply_loadgen_value_overrides(target, cli, loadgen_set)
    _apply_default_grace_period_override(target, cli, loadgen_set)
    _apply_agentic_replay_fields(target, cli)
    _apply_phase_shape_overrides(target, cli, loadgen_set)


def _apply_rate_series_override(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    if "request_rate_series" not in fields_set or cli.request_rate_series is None:
        return

    from aiperf.config.rate_series import RateSeriesConfig

    series = RateSeriesConfig(path=str(cli.request_rate_series))
    target["rate_series"] = series.model_dump(exclude_none=True, exclude={"path"})
    target.pop("rate", None)
    if "arrival_pattern" in fields_set:
        target["type"] = {
            ArrivalPattern.POISSON: PhaseType.POISSON,
            ArrivalPattern.GAMMA: PhaseType.GAMMA,
            ArrivalPattern.CONSTANT: PhaseType.CONSTANT,
        }.get(cli.arrival_pattern, PhaseType.POISSON)


def _apply_loadgen_value_overrides(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    for attr, key in _LOADGEN_PHASE_FIELD_MAP:
        if attr not in fields_set:
            continue
        value = getattr(cli, attr)
        if value is None:
            continue
        target[key] = value


def _apply_default_grace_period_override(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    if (
        "benchmark_duration" in fields_set
        and "benchmark_grace_period" not in fields_set
        and cli.benchmark_duration is not None
        and "grace_period" not in target
    ):
        target["grace_period"] = cli.benchmark_grace_period


def _apply_phase_shape_overrides(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    """Apply phase-discriminator, ramp, and cancellation CLI overrides."""
    _apply_phase_type_override(target, cli, fields_set)
    _apply_arrival_smoothness_override(target, cli, fields_set)
    _apply_phase_ramp_overrides(target, cli, fields_set)
    _apply_fixed_schedule_offset_overrides(target, cli, fields_set)
    _apply_cancellation_override(target, cli, fields_set)


def _apply_phase_type_override(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    if "fixed_schedule" in fields_set and cli.fixed_schedule:
        _apply_fixed_schedule_type_override(target, fields_set)
        return
    if "user_centric_rate" in fields_set:
        _transition_phase_type(target, PhaseType.USER_CENTRIC)
        return
    if "request_rate_series" in fields_set:
        phase_type = (
            _arrival_phase_type(cli)
            if "arrival_pattern" in fields_set
            else PhaseType.POISSON
        )
        _transition_phase_type(target, phase_type)
        return
    if "request_rate" in fields_set:
        _apply_request_rate_type_override(target, cli, fields_set)
        return
    if "arrival_pattern" in fields_set:
        _require_rate_controlled_phase(
            target, "--arrival-pattern requires a rate-controlled profiling phase"
        )
        if _preserve_user_centric_phase(target, fields_set):
            return
        _transition_phase_type(target, _arrival_phase_type(cli))


def _preserve_user_centric_phase(target: dict[str, Any], fields_set: set[str]) -> bool:
    """Return True when a YAML ``user_centric`` phase must keep its type.

    ``--request-rate`` / ``--arrival-pattern`` imply an open-loop phase only
    when the CLI alone defines the workload. Against a config-file
    ``user_centric`` phase they are edits to a phase that already owns a
    ``rate`` field, so switching the discriminator would silently drop
    ``users`` and swap the closed-loop user model the config asked for.
    """
    if target.get("type") != PhaseType.USER_CENTRIC:
        return False
    if "arrival_pattern" in fields_set:
        logger.warning(
            "--arrival-pattern is ignored: the profiling phase in the config "
            "file is 'user_centric', which has no arrival distribution. The "
            "phase keeps type 'user_centric' and its 'users' value. Change the "
            "phase type in YAML to poisson/gamma/constant for an open-loop "
            "arrival pattern."
        )
    return True


def _apply_fixed_schedule_type_override(
    target: dict[str, Any], fields_set: set[str]
) -> None:
    from aiperf.config.loader.errors import ConfigurationError

    conflicts = fields_set & {
        "request_rate",
        "request_rate_series",
        "user_centric_rate",
    }
    if conflicts:
        raise ConfigurationError(
            "--fixed-schedule cannot be combined with rate-control CLI flags: "
            f"{', '.join(sorted(conflicts))}"
        )
    _transition_phase_type(target, PhaseType.FIXED_SCHEDULE)


def _apply_request_rate_type_override(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    if _preserve_user_centric_phase(target, fields_set):
        return
    current_type = target.get("type")
    rate_types = {
        PhaseType.POISSON,
        PhaseType.GAMMA,
        PhaseType.CONSTANT,
    }
    if "arrival_pattern" in fields_set:
        phase_type = _arrival_phase_type(cli)
    elif current_type in rate_types:
        phase_type = current_type
    else:
        phase_type = PhaseType.POISSON
    _transition_phase_type(target, phase_type)


def _require_rate_controlled_phase(target: dict[str, Any], message: str) -> None:
    from aiperf.config.loader.errors import ConfigurationError

    if target.get("rate") is None and _get_config_value(target, "rate_series") is None:
        raise ConfigurationError(message)


def _apply_arrival_smoothness_override(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    if "arrival_smoothness" not in fields_set:
        return
    _require_rate_controlled_phase(
        target,
        "--arrival-smoothness requires a rate-controlled profiling phase",
    )
    if target.get("type") == PhaseType.USER_CENTRIC:
        # Unlike --request-rate, smoothness cannot be preserved in place:
        # UserCentricPhase has no `smoothness` field, so honoring the flag
        # would mean silently dropping `users` and rewriting the load model.
        from aiperf.config.loader.errors import ConfigurationError

        raise ConfigurationError(
            "--arrival-smoothness cannot be applied to the 'user_centric' "
            "profiling phase from the config file: user-centric phases have no "
            "arrival-distribution shape. Change the phase type in YAML to "
            "'gamma' to use --arrival-smoothness."
        )
    _transition_phase_type(target, PhaseType.GAMMA)
    target["smoothness"] = cli.arrival_smoothness


def _apply_phase_ramp_overrides(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    from aiperf.config.loader.errors import ConfigurationError

    for cli_field, phase_field in (
        ("concurrency_ramp_duration", "concurrency_ramp"),
        ("prefill_concurrency_ramp_duration", "prefill_ramp"),
        ("request_rate_ramp_duration", "rate_ramp"),
    ):
        if cli_field not in fields_set:
            continue
        if cli_field == "request_rate_ramp_duration" and target.get("type") not in {
            PhaseType.POISSON,
            PhaseType.GAMMA,
            PhaseType.CONSTANT,
            PhaseType.USER_CENTRIC,
        }:
            raise ConfigurationError(
                "--request-rate-ramp-duration requires a rate-controlled profiling phase"
            )
        target[phase_field] = {"duration": getattr(cli, cli_field)}


def _apply_fixed_schedule_offset_overrides(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    from aiperf.config.loader.errors import ConfigurationError

    fixed_offset_fields = {
        "fixed_schedule_auto_offset": "auto_offset",
        "fixed_schedule_start_offset": "start_offset",
        "fixed_schedule_end_offset": "end_offset",
    }
    if fields_set & fixed_offset_fields.keys():
        if target.get("type") != PhaseType.FIXED_SCHEDULE:
            raise ConfigurationError(
                "fixed-schedule offset CLI flags require a fixed_schedule profiling phase"
            )
        for cli_field, phase_field in fixed_offset_fields.items():
            if cli_field in fields_set:
                target[phase_field] = getattr(cli, cli_field)
        if "fixed_schedule_start_offset" in fields_set:
            target.setdefault("auto_offset", False)


def _arrival_phase_type(cli: CLIConfig) -> PhaseType:
    return {
        ArrivalPattern.GAMMA: PhaseType.GAMMA,
        ArrivalPattern.CONSTANT: PhaseType.CONSTANT,
    }.get(cli.arrival_pattern, PhaseType.POISSON)


def _transition_phase_type(target: dict[str, Any], phase_type: PhaseType) -> None:
    """Change a phase discriminator and discard only incompatible YAML fields."""
    rate_types = {
        PhaseType.POISSON,
        PhaseType.GAMMA,
        PhaseType.CONSTANT,
        PhaseType.USER_CENTRIC,
    }
    if phase_type not in rate_types:
        for key in ("rate", "rate_ramp", "rate_series", "smoothness", "users"):
            _pop_config_value(target, key)
    else:
        if phase_type != PhaseType.GAMMA:
            _pop_config_value(target, "smoothness")
        if phase_type != PhaseType.USER_CENTRIC:
            _pop_config_value(target, "users")
        for key in ("auto_offset", "start_offset", "end_offset"):
            _pop_config_value(target, key)
    target["type"] = phase_type


def _get_config_value(mapping: dict[str, Any], key: str) -> Any:
    from pydantic.alias_generators import to_camel

    return mapping.get(key, mapping.get(to_camel(key)))


def _pop_config_value(mapping: dict[str, Any], key: str) -> None:
    from pydantic.alias_generators import to_camel

    mapping.pop(key, None)
    mapping.pop(to_camel(key), None)


def _apply_cancellation_override(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    from aiperf.config.loader.errors import ConfigurationError

    if not fields_set & {"request_cancellation_rate", "request_cancellation_delay"}:
        return
    cancellation = target.get("cancellation")
    if not isinstance(cancellation, dict):
        cancellation = {}
    if "request_cancellation_rate" in fields_set:
        cancellation["rate"] = cli.request_cancellation_rate
    if "request_cancellation_delay" in fields_set:
        cancellation["delay"] = cli.request_cancellation_delay
    if "rate" not in cancellation:
        raise ConfigurationError(
            "--request-cancellation-delay requires a cancellation rate in YAML "
            "or --request-cancellation-rate"
        )
    target["cancellation"] = cancellation


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
    if "request_rate_series" in fields_set:
        collisions.setdefault("rate", []).append("request_rate_series")
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
    """Return the unique profiling-kind phase for CLI loadgen overlays.

    Legacy YAML may omit ``kind`` on canonical names, so infer it for this
    pre-validation merge pass. Ambiguous multi-profiling configs must express
    values directly in YAML for v1.
    """
    candidates: list[dict[str, Any]] = []
    for entry in phases:
        if not isinstance(entry, dict):
            continue
        kind = infer_legacy_phase_kind(entry.get("name"), entry.get("kind"))
        if kind is not None:
            entry["kind"] = kind
        if kind == "profiling":
            candidates.append(entry)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        from aiperf.config.loader.errors import ConfigurationError

        names = [str(entry.get("name")) for entry in candidates]
        raise ConfigurationError(
            "CLI loadgen flags target the profiling phase, but this config has "
            f"{len(candidates)} profiling phases: {', '.join(names)}. Set the "
            "value in YAML or use an explicit phase path."
        )
    return None
