# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a ``CLIConfig`` + optional YAML ``--config`` file into an
``AIPerfConfig``.

Used by every CLI command that supports both flag-form and file-form input
(``aiperf profile`` and ``aiperf service``). When both are supplied, the YAML
supplies the base configuration and any explicitly-set CLI flags on
``cli_config`` are deep-merged on top before AIPerfConfig validation -- so
``aiperf profile --config foo.yaml --streaming --search-recipe X`` works the
way users intuit instead of throwing
``CLIConfig.endpoint.modelNames: Field required``.

Not every flag can be applied this way. Anything this path cannot route is
rejected up front by ``reject_unrouted_cli_flags`` with an error naming the
flag, rather than being silently discarded -- see ``_config_flag_routing``
and ``docs/dev/global-invariants.md`` for the classification and the tests
that keep it honest. ``--ttft-sla-ms`` is one such flag today: it does not
take effect under ``--config`` even alongside a recipe, so it errors.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_args

from pydantic.alias_generators import to_camel

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
    LOADGEN_FIELDS,
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


def apply_cli_overrides(
    config: AIPerfConfig,
    cli_config: CLIConfig,
) -> AIPerfConfig:
    """Overlay explicitly-authored CLI values on an already-loaded config.

    Kubernetes workload YAML is first separated from its CR deployment fields,
    so it cannot use :func:`resolve_config`'s file-loading entry point directly.
    This adapter feeds the validated config and its retained pre-Jinja envelope
    through the exact same override pipeline used by ordinary Config-v2 files.

    Args:
        config: Loaded Config-v2 envelope that supplies the YAML baseline.
        cli_config: Parsed CLI values; only ``model_fields_set`` entries apply.

    Returns:
        A new config with CLI precedence and a matching raw sweep envelope.
    """
    rendered = config.model_dump(
        mode="python",
        by_alias=True,
        exclude_unset=True,
        exclude_none=True,
        context={"include_secrets": True},
    )
    raw = copy.deepcopy(config._raw_envelope or rendered)
    return _resolve_config_envelopes(cli_config, rendered, raw)


def _resolve_config_envelopes(
    cli_config: CLIConfig,
    yaml_dict: dict[str, Any],
    raw_yaml_dict: dict[str, Any],
) -> AIPerfConfig:
    """Resolve rendered and pre-Jinja envelopes through one override pipeline."""
    from aiperf.config import AIPerfConfig
    from aiperf.config.flags._config_flag_routing import reject_unrouted_cli_flags
    from aiperf.config.flags.converter import _wrap_under_envelope

    # Fail before any merging: a flag this path cannot route would otherwise
    # be dropped without a word, handing the user a benchmark that silently
    # ignored what they asked for.
    reject_unrouted_cli_flags(cli_config)
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

    overrides = build_cli_overrides(cli_config, benchmark_config=base_config.benchmark)
    overrides = _wrap_under_envelope(overrides) if overrides else overrides
    merged = _merge_overrides_into_envelope(yaml_dict, overrides, cli_config)
    raw_merged = _merge_overrides_into_envelope(
        raw_yaml_dict,
        overrides,
        cli_config,
        phase_shape_decision=merged.phase_shape_decision,
    )

    config = AIPerfConfig.model_validate(merged.envelope)
    config._raw_envelope = raw_merged.envelope
    _validate_search_space_phase_targets(config, merged.envelope)
    return config


def _validate_search_space_phase_targets(
    config: AIPerfConfig, envelope: dict[str, Any]
) -> None:
    """Reject searched dimensions the resolved phase cannot accept.

    ``--search-space`` shape inference only runs when the profiling phase is
    built from CLI flags (``_converter_profiling.build_profiling``). A config
    that authors its own ``phases:`` keeps its discriminator, so a dimension
    such as ``phases.profiling.users`` can target a phase type that has no
    such field. The path is syntactically valid, so nothing rejects it until
    the planner writes its first sampled value and Pydantic raises
    ``extra_forbidden`` -- mid-run, after the benchmark is already underway
    and the searched dimension has cost real time. Fail at resolution instead.
    """
    from aiperf.config.loader.errors import ConfigurationError
    from aiperf.config.sweep.expand import _find_phase_or_recipe_alias

    dimensions = getattr(config.sweep, "search_space", None)
    if not dimensions:
        return
    benchmark = envelope.get("benchmark")
    if not isinstance(benchmark, dict):
        return
    phases = benchmark.get("phases")
    if not isinstance(phases, list) or not phases:
        return

    for dimension in dimensions:
        path = getattr(dimension, "path", None)
        if not isinstance(path, str):
            continue
        segments = path.split(".")
        # Only direct ``phases.<selector>.<field>`` scalars are checked; a
        # deeper path (``phases.profiling.cancellation.rate``) targets a
        # sub-model whose own validation owns the field.
        if len(segments) != 3 or segments[0] != "phases":
            continue
        selector, field_name = segments[1], segments[2]
        target = _find_phase_or_recipe_alias(phases, selector, parent_key="phases")
        if target is None:
            continue
        try:
            resolved = config.benchmark.phases[phases.index(target)]
        except (ValueError, IndexError):
            continue
        if _phase_accepts_field(type(resolved), field_name):
            continue
        raise ConfigurationError(
            f"--search-space dimension {path!r} targets field {field_name!r}, "
            f"which phase {resolved.name!r} (type {resolved.type}) does not "
            f"have. {_phase_types_declaring(field_name)} Either search a field "
            f"the phase declares, or change the phase's 'type:'."
        )


def _phase_accepts_field(phase_cls: type, field_name: str) -> bool:
    """Whether ``phase_cls`` declares ``field_name`` under either spelling."""
    fields = phase_cls.model_fields
    if field_name in fields:
        return True
    return any(info.alias == field_name for info in fields.values())


def _phase_types_declaring(field_name: str) -> str:
    """Human-readable hint naming the phase types that do declare a field."""
    from aiperf.config import phases as phase_models

    declaring: list[str] = []
    for name in dir(phase_models):
        candidate = getattr(phase_models, name)
        if not isinstance(candidate, type) or not hasattr(candidate, "model_fields"):
            continue
        type_field = candidate.model_fields.get("type")
        if type_field is None:
            continue
        # Concrete phases pin the discriminator as ``Literal[PhaseType.X]``;
        # the abstract bases annotate the bare enum, so an empty get_args()
        # is what filters them out of the hint.
        members = get_args(type_field.annotation)
        if len(members) != 1:
            continue
        if _phase_accepts_field(candidate, field_name):
            declaring.append(str(members[0]))
    unique = sorted(set(declaring))
    if not unique:
        return "No phase type declares it."
    return f"Declared by phase type(s): {', '.join(unique)}."


@dataclass(slots=True)
class _PhaseShapeDecision:
    """Value-dependent phase-shape outcome of one override pass.

    The rendered envelope is the only one whose phase fields hold real values;
    in the retained pre-Jinja envelope the same fields may still be ``{{ ... }}``
    source strings. Re-deriving the phase discriminator there would compare a
    template string against :class:`PhaseType` members and reach a different
    answer, so the decision is recorded once on the rendered pass and replayed
    verbatim onto the raw one.

    Attributes:
        phase_type: Final discriminator written onto the profiling phase, or
            ``None`` when no CLI flag changed it.
        removed_keys: Canonical (snake_case) phase keys discarded as
            incompatible with ``phase_type``; replayed through
            :func:`_pop_config_value` so either spelling is removed.
    """

    phase_type: PhaseType | None = None
    removed_keys: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _MergedEnvelope:
    """One override pass' merged envelope plus the decision it reached.

    Attributes:
        envelope: The merged config envelope dict.
        phase_shape_decision: Phase-shape outcome to replay onto sibling
            envelopes so validation and execution cannot disagree.
    """

    envelope: dict[str, Any]
    phase_shape_decision: _PhaseShapeDecision


def _merge_overrides_into_envelope(
    envelope: dict[str, Any],
    overrides: dict[str, Any] | None,
    cli_config: CLIConfig,
    *,
    phase_shape_decision: _PhaseShapeDecision | None = None,
) -> _MergedEnvelope:
    """Apply the config-file CLI override pipeline to one envelope.

    The resolver calls this once for the rendered envelope used for Pydantic
    validation and once for the retained pre-Jinja envelope used by sweep
    expansion. Structural rewrites run on both, but every value-dependent
    phase-shape branch (and the ``ConfigurationError`` guards built on it) runs
    only on the rendered pass; passing that pass' ``phase_shape_decision`` back
    in switches this call into replay mode, which is what keeps CLI overrides
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
    _apply_dataset_overrides(merged, cli_config)
    decision = _apply_phase_loadgen_overrides(
        merged, cli_config, phase_shape_decision=phase_shape_decision
    )
    _apply_warmup_overrides(merged, cli_config)
    promote_benchmark_magic_lists(
        merged,
        cli_config,
        promote_cli_dataset_magic_lists=_promote_cli_dataset_magic_lists,
        promote_magic_lists_to_sweep_block=_promote_magic_lists_to_sweep_block,
        retarget_dataset_magic_lists=_retarget_dataset_magic_lists,
    )
    _coalesce_phase_aliases(merged)
    return _MergedEnvelope(envelope=merged, phase_shape_decision=decision)


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
    wandb_base_enabled = benchmark_config is not None and benchmark_config.wandb.enabled
    _apply_optional_section(
        out, "wandb", build_wandb(cli, base_enabled=wandb_base_enabled)
    )
    # These three builders already existed and are used by the CLI-only
    # converter; this path simply never called them, so --mlflow-*,
    # --otel-url, and the network-latency flags were dropped whenever a
    # config file was supplied. They gate on model_fields_set, so an unset
    # flag leaves the YAML block alone.
    _apply_optional_section(out, "network_latency", build_network_latency(cli))
    otel_base_url = benchmark_config is not None and bool(
        benchmark_config.otel.metrics_url
    )
    _apply_optional_section(
        out, "otel", build_otel(cli, base_metrics_url=otel_base_url)
    )
    mlflow_base_uri = benchmark_config is not None and bool(
        benchmark_config.mlflow.tracking_uri
    )
    _apply_optional_section(
        out, "mlflow", build_mlflow(cli, base_tracking_uri=mlflow_base_uri)
    )
    _apply_scenario_overrides(out, cli)

    if cli.goodput:
        # Recipe-emitted SLOs win on key collision, matching
        # convert_cli_to_aiperf: a goodput-style recipe owns the SLO contract
        # for its run, so a stray --goodput must not override its thresholds.
        slos = dict(cli.goodput)
        slos.update(out.get("slos") or {})
        out["slos"] = slos

    if "no_sweep_table" in cli.model_fields_set:
        out["no_sweep_table"] = cli.no_sweep_table
    # random_seed is an envelope key distinct from the dataset's own seed
    # field (written by build_dataset onto the dataset block, which only
    # feeds SessionIDGenerator). AIPerfConfig.random_seed is what
    # resolve_run_seed threads into rng.init(...) for every child service
    # process, so every other rng.derive(...) consumer -- synthetic prompt
    # content, media generation, per-conversation turn shaping -- needs it
    # too, matching _assemble_optional on the CLI-only path.
    if "random_seed" in cli.model_fields_set:
        out["random_seed"] = cli.random_seed

    # Service-runtime CLI flags (--ui, --log-level, --verbose, ZMQ knobs)
    # land on RuntimeConfig / LoggingConfig in AIPerfConfig. build_logging_runtime
    # already gates on cli.model_fields_set, so YAML defaults stay
    # intact when the user didn't pass these flags.
    runtime_base_port = benchmark_config is not None and (
        benchmark_config.runtime.api_port is not None
    )
    logging_dict, runtime_dict = build_logging_runtime(
        cli, base_api_port=runtime_base_port
    )
    _apply_optional_section(out, "logging", logging_dict)
    _apply_optional_section(out, "runtime", runtime_dict)

    return out


def _apply_scenario_overrides(out: dict[str, Any], cli: CLIConfig) -> None:
    """Mirror the converter's scenario-lock fields onto the override dict.

    ``scenario`` and ``unsafe_override`` are plain data on ``BenchmarkConfig``
    rather than envelope keys, so ``_wrap_under_envelope`` moves them under
    ``benchmark:`` exactly as ``_apply_scenario_fields`` relies on for the
    CLI-only path.
    """
    set_fields = cli.model_fields_set
    if "scenario" in set_fields:
        out["scenario"] = cli.scenario
    if "unsafe_override" in set_fields:
        out["unsafe_override"] = cli.unsafe_override


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

    ``--model-names`` and ``--model-selection-strategy`` live on the CLIConfig
    endpoint section but map to the ``models`` block on AIPerfConfig
    (``items`` / ``strategy``); everything else stays on ``endpoint``.
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
    if "model_names" in fields_set and cli.model_names:
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


def _locate_yaml_dataset(merged: dict[str, Any]) -> dict[str, Any] | None:
    """Return the YAML dataset dict that CLI overrides apply to.

    Accepts both the ``benchmark.dataset`` shorthand and the canonical
    ``benchmark.datasets`` list. The multi-dataset branch below anticipates a
    cap that has not been lifted yet, not current behavior: ``AIPerfConfig
    .datasets`` is ``max_length=1``, and ``resolve_config`` validates before
    overrides run, so a YAML with more than one dataset entry fails
    validation before this function is ever reached. Kept as future-proofing
    -- matching the long-standing convention of the dataset-filter and
    synthesis overlays this function replaces, from when the cap did not
    exist -- for whenever the limit lifts.
    """
    benchmark = merged.get("benchmark")
    if not isinstance(benchmark, dict):
        return None
    dataset = benchmark.get("dataset")
    if isinstance(dataset, dict):
        return dataset
    datasets = benchmark.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return None
    if len(datasets) > 1:
        logger.warning(
            "Dataset CLI flags with multiple YAML datasets apply only to the "
            "first dataset"
        )
    first = datasets[0]
    return first if isinstance(first, dict) else None


def _snake_to_camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word[:1].upper() + word[1:] for word in rest)


def _drop_alias_spellings(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Remove the camelCase spelling of every key the override sets.

    YAML configs address fields by their camelCase alias (``maxOsl``) while
    ``build_dataset`` emits the snake_case field name (``max_osl``). Merging
    the two as-is would leave both spellings of the same field in the dict and
    let validation pick a winner -- which is exactly the kind of quiet
    ambiguity this work exists to remove. Deleting the alias first makes the
    CLI value win outright, whichever spelling the config file used.
    """
    for key, value in override.items():
        alias = _snake_to_camel(key)
        if alias != key and alias in base:
            if isinstance(value, dict) and isinstance(base[alias], dict):
                # Same field, different spelling: keep the YAML's siblings by
                # folding its sub-dict onto the canonical key before merging.
                base[key] = deep_merge(base.pop(alias), base.get(key) or {})
            else:
                base.pop(alias, None)
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _drop_alias_spellings(base[key], value)


def _inert_dataset_flags(
    cli: CLIConfig,
    declared_type: Any,
    declared_format: Any,
    candidates: set[str],
) -> list[str]:
    """Return the flags in ``candidates`` that contribute nothing on their own.

    Attribution is by re-running the real builder with exactly one flag set,
    rather than by inspecting the combined result: a flag that emits nothing
    alone emits nothing in company either, and asking ``build_dataset``
    directly means the answer cannot drift from what it actually does.

    A flag whose solo build raises is treated as contributing -- it is loud,
    which is the property that matters, and the combined build already
    succeeded.
    """
    # CLIConfig is a TYPE_CHECKING-only import at module scope; it has to be
    # imported for real here, and the narrow `except` below is what would
    # have surfaced that rather than silently treating every flag as routed.
    from aiperf.config.flags import CLIConfig as _CLIConfig
    from aiperf.config.flags._converter_dataset import build_dataset
    from aiperf.config.loader.errors import ConfigurationError

    inert: list[str] = []
    for candidate in candidates:
        # Solo construction is deliberately outside the try: a CLIConfig
        # cross-field validator failing on this single field alone would be
        # a real bug in this function's premise (the combined CLIConfig
        # already validated), and must surface rather than be swallowed by
        # the except below, which exists for build_dataset only.
        solo = _CLIConfig(**{candidate: getattr(cli, candidate)})
        try:
            emitted = build_dataset(
                solo, declared_type=declared_type, declared_format=declared_format
            )
        except (ValueError, ConfigurationError):
            # Loud on its own, which is the property that matters; the
            # combined build already succeeded. Deliberately narrow: a broad
            # `except` here would turn a bug in this function into a silently
            # disabled guard.
            continue
        if not emitted:
            inert.append(candidate)
    return inert


def _reject_inert_dataset_flags(
    cli: CLIConfig, dataset: dict[str, Any], declared_type: Any
) -> None:
    """Raise for dataset flags that resolve cleanly while doing nothing.

    Checking only whether the whole override came back empty let any inert
    flag ride along with a routable one: the result was non-empty, the guard
    never fired, and the inert flag was dropped exactly as before this work.
    Every multi-flag command line touching the dataset escaped the guarantee,
    so the reconciliation is per flag.

    ``declared_type`` is the caller's already-defaulted value (``dataset.get(
    "type") or DatasetType.SYNTHETIC``), passed in rather than re-derived so
    the error message reports the same type the check actually ran against
    -- re-deriving it here previously meant the message rendered the raw,
    possibly-``None`` YAML value instead.
    """
    from aiperf.config.flags._config_flag_routing import (
        DATASET_FIELDS_OUTSIDE_INPUT,
        DATASET_OVERRIDE_FIELDS,
        flag_names_for,
    )
    from aiperf.config.loader.errors import ConfigurationError

    candidates = cli.model_fields_set & (
        DATASET_OVERRIDE_FIELDS | DATASET_FIELDS_OUTSIDE_INPUT
    )
    if not candidates:
        return

    inert = _inert_dataset_flags(
        cli,
        declared_type,
        dataset.get("format"),
        candidates,
    )
    if not inert:
        return

    names = sorted("/".join(flag_names_for(f) or (f,)) for f in inert)
    raise ConfigurationError(
        f"These CLI flags have no effect on a dataset of type "
        f"{str(declared_type)!r}: {', '.join(names)}. Remove them, or use a "
        f"dataset type that supports them."
    )


def _apply_dataset_overrides(merged: dict[str, Any], cli: CLIConfig) -> None:
    """Overlay explicitly-set dataset flags onto the YAML-supplied dataset.

    Delegates to ``build_dataset`` -- the same builder the CLI-only path uses
    -- in override mode, so the two paths share one implementation of how a
    flag maps onto the dataset shape. Previously this file carried a
    hand-written routing function per field class (synthesis, filters,
    random_pool batch sizes), each of which had to be kept in step with the
    converter by hand; anything nobody wrote a function for was silently
    dropped.

    ``build_dataset`` emits only keys backed by ``cli.model_fields_set``, so
    an unset flag cannot clobber a YAML value, and the config file keeps
    ownership of dataset type, format, and source.
    """
    from aiperf.config.flags._config_flag_routing import DATASET_OVERRIDE_FIELDS
    from aiperf.config.flags._converter_dataset import (
        apply_implicit_media_batch_override,
        build_dataset,
    )

    dataset = _locate_yaml_dataset(merged)
    if dataset is None:
        if cli.model_fields_set & DATASET_OVERRIDE_FIELDS:
            from aiperf.config.loader.errors import ConfigurationError

            raise ConfigurationError(
                "Dataset CLI flags require a dataset in the config file, but "
                "none was found under benchmark.dataset / benchmark.datasets."
            )
        return

    # A YAML dataset may omit `type`; validation resolves that to synthetic.
    # Passing None here would instead switch build_dataset back to inferring
    # the type from flags -- the CLI-only path, which materializes defaults
    # meant for building a dataset from nothing. Default it so both spellings
    # of "synthetic" take the same route.
    declared_type = dataset.get("type") or DatasetType.SYNTHETIC
    override = build_dataset(
        cli,
        declared_type=declared_type,
        declared_format=dataset.get("format"),
    )
    _reject_inert_dataset_flags(cli, dataset, declared_type)
    if not override:
        return

    apply_implicit_media_batch_override(override, dataset)
    _drop_alias_spellings(dataset, override)
    merged_dataset = deep_merge(dataset, override)
    dataset.clear()
    dataset.update(merged_dataset)


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


def _apply_phase_loadgen_overrides(
    merged: dict[str, Any],
    cli: CLIConfig,
    *,
    phase_shape_decision: _PhaseShapeDecision | None = None,
) -> _PhaseShapeDecision:
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

    Args:
        merged: Config envelope to mutate in place.
        cli: Parsed CLI values; only ``model_fields_set`` entries apply.
        phase_shape_decision: When supplied, the phase-shape outcome already
            decided against the rendered envelope, replayed here instead of
            being re-derived from this envelope's (possibly still-templated)
            values.

    Returns:
        The phase-shape decision this pass reached (or replayed).
    """
    from aiperf.config.flags._converter_profiling import (
        _AGENTIC_REPLAY_ROUTES,
        _apply_agentic_replay_fields,
    )

    decision = phase_shape_decision or _PhaseShapeDecision()
    loadgen_set = cli.model_fields_set & _PROFILING_PHASE_OVERRIDE_FIELDS
    agentic_set = cli.model_fields_set.intersection(_AGENTIC_REPLAY_ROUTES)
    if not loadgen_set and not agentic_set:
        return decision

    benchmark = merged.get("benchmark")
    if not isinstance(benchmark, dict):
        return decision
    phases = benchmark.get("phases")
    if not isinstance(phases, list) or not phases:
        return decision

    target = _find_profiling_phase(phases)
    if target is None:
        return decision

    _reject_loadgen_target_collisions(loadgen_set)
    _apply_loadgen_value_overrides(target, cli, loadgen_set)
    _apply_default_grace_period_override(target, cli, loadgen_set)
    _apply_agentic_replay_fields(target, cli)
    decision = _apply_phase_shape_overrides(
        target, cli, loadgen_set, decision=phase_shape_decision
    )
    _apply_rate_series_override(target, cli, loadgen_set)
    return decision


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
    # This helper runs before _coalesce_phase_aliases, so a YAML-authored
    # ``gracePeriod`` is still under its camelCase spelling here. Checking only
    # the snake_case key would miss it, write the CLI default under
    # ``grace_period``, and let the coalesce step overwrite the user's value.
    from pydantic.alias_generators import to_camel

    if (
        "benchmark_duration" in fields_set
        and "benchmark_grace_period" not in fields_set
        and cli.benchmark_duration is not None
        and "grace_period" not in target
        and to_camel("grace_period") not in target
    ):
        target["grace_period"] = cli.benchmark_grace_period


def _apply_phase_shape_overrides(
    target: dict[str, Any],
    cli: CLIConfig,
    fields_set: set[str],
    *,
    decision: _PhaseShapeDecision | None = None,
) -> _PhaseShapeDecision:
    """Apply phase-discriminator, ramp, and cancellation CLI overrides.

    With ``decision=None`` this is the deciding pass: the discriminator is
    derived from ``target``'s own values and every ``ConfigurationError`` guard
    is evaluated. With a ``decision`` supplied this is the replay pass over a
    sibling envelope whose values may still be Jinja source; the recorded
    discriminator is written verbatim and the guards are skipped, since they
    already passed against real values.
    """
    replay = decision is not None
    if decision is None:
        decision = _PhaseShapeDecision()
        _apply_phase_type_override(target, cli, fields_set, decision)
    else:
        _replay_phase_shape_decision(target, decision)
    _apply_arrival_smoothness_override(
        target, cli, fields_set, decision, validate=not replay
    )
    _apply_phase_ramp_overrides(target, cli, fields_set, validate=not replay)
    _apply_fixed_schedule_offset_overrides(target, cli, fields_set, validate=not replay)
    _apply_cancellation_override(target, cli, fields_set)
    return decision


def _replay_phase_shape_decision(
    target: dict[str, Any], decision: _PhaseShapeDecision
) -> None:
    """Write a previously-decided discriminator onto a sibling envelope."""
    for key in decision.removed_keys:
        _pop_config_value(target, key)
    if decision.phase_type is not None:
        target["type"] = decision.phase_type


def _apply_phase_type_override(
    target: dict[str, Any],
    cli: CLIConfig,
    fields_set: set[str],
    decision: _PhaseShapeDecision,
) -> None:
    if "fixed_schedule" in fields_set and cli.fixed_schedule:
        _apply_fixed_schedule_type_override(target, fields_set, decision)
        return
    if "user_centric_rate" in fields_set:
        _transition_phase_type(target, PhaseType.USER_CENTRIC, decision)
        return
    if "request_rate_series" in fields_set:
        phase_type = (
            _arrival_phase_type(cli)
            if "arrival_pattern" in fields_set
            else PhaseType.POISSON
        )
        _transition_phase_type(target, phase_type, decision)
        return
    if "request_rate" in fields_set:
        _apply_request_rate_type_override(target, cli, fields_set, decision)
        return
    if "arrival_pattern" in fields_set:
        _require_rate_controlled_phase(
            target, "--arrival-pattern requires a rate-controlled profiling phase"
        )
        if _preserve_user_centric_phase(target, fields_set):
            return
        _transition_phase_type(target, _arrival_phase_type(cli), decision)


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
    target: dict[str, Any], fields_set: set[str], decision: _PhaseShapeDecision
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
    _transition_phase_type(target, PhaseType.FIXED_SCHEDULE, decision)


def _apply_request_rate_type_override(
    target: dict[str, Any],
    cli: CLIConfig,
    fields_set: set[str],
    decision: _PhaseShapeDecision,
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
    _transition_phase_type(target, phase_type, decision)


def _require_rate_controlled_phase(target: dict[str, Any], message: str) -> None:
    from aiperf.config.loader.errors import ConfigurationError

    if target.get("rate") is None and _get_config_value(target, "rate_series") is None:
        raise ConfigurationError(message)


def _apply_arrival_smoothness_override(
    target: dict[str, Any],
    cli: CLIConfig,
    fields_set: set[str],
    decision: _PhaseShapeDecision,
    *,
    validate: bool,
) -> None:
    if "arrival_smoothness" not in fields_set:
        return
    if validate:
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
        _transition_phase_type(target, PhaseType.GAMMA, decision)
    target["smoothness"] = cli.arrival_smoothness


def _apply_phase_ramp_overrides(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str], *, validate: bool
) -> None:
    """Overlay ramp ``duration`` CLI flags onto the phase's ramp mappings.

    ``RampConfig`` carries ``strategy`` alongside ``duration``, so the flag
    merges into whichever spelling the YAML already used instead of replacing
    the mapping -- otherwise ``--concurrency-ramp-duration`` would silently
    reset a user-authored ``strategy: exponential`` back to the model default,
    and would do so only for the snake_case spelling (``_coalesce_phase_aliases``
    already deep-merges the camelCase twin).
    """
    from aiperf.config.loader.errors import ConfigurationError

    for cli_field, phase_field in (
        ("concurrency_ramp_duration", "concurrency_ramp"),
        ("prefill_concurrency_ramp_duration", "prefill_ramp"),
        ("request_rate_ramp_duration", "rate_ramp"),
    ):
        if cli_field not in fields_set:
            continue
        if (
            validate
            and cli_field == "request_rate_ramp_duration"
            and target.get("type")
            not in {
                PhaseType.POISSON,
                PhaseType.GAMMA,
                PhaseType.CONSTANT,
                PhaseType.USER_CENTRIC,
            }
        ):
            raise ConfigurationError(
                "--request-rate-ramp-duration requires a rate-controlled profiling phase"
            )
        target_key = phase_field
        alias = to_camel(phase_field)
        if phase_field not in target and alias in target:
            target_key = alias
        existing = target.get(target_key)
        duration = {"duration": getattr(cli, cli_field)}
        target[target_key] = (
            deep_merge(existing, duration) if isinstance(existing, dict) else duration
        )


def _apply_fixed_schedule_offset_overrides(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str], *, validate: bool
) -> None:
    from aiperf.config.loader.errors import ConfigurationError

    fixed_offset_fields = {
        "fixed_schedule_auto_offset": "auto_offset",
        "fixed_schedule_start_offset": "start_offset",
        "fixed_schedule_end_offset": "end_offset",
    }
    if fields_set & fixed_offset_fields.keys():
        if validate and target.get("type") != PhaseType.FIXED_SCHEDULE:
            raise ConfigurationError(
                "fixed-schedule offset CLI flags require a fixed_schedule profiling phase"
            )
        for cli_field, phase_field in fixed_offset_fields.items():
            if cli_field in fields_set:
                target[phase_field] = getattr(cli, cli_field)
        # This runs before _coalesce_phase_aliases, so a YAML-authored
        # ``autoOffset`` is still under its camelCase spelling here. Checking
        # only the snake_case key would miss it, write the default under
        # ``auto_offset``, and let the coalesce step overwrite the user's
        # camelCase value with this False default (same hazard as
        # _apply_default_grace_period_override above).
        if (
            "fixed_schedule_start_offset" in fields_set
            and "auto_offset" not in target
            and to_camel("auto_offset") not in target
        ):
            target["auto_offset"] = False


def _arrival_phase_type(cli: CLIConfig) -> PhaseType:
    return {
        ArrivalPattern.GAMMA: PhaseType.GAMMA,
        ArrivalPattern.CONSTANT: PhaseType.CONSTANT,
    }.get(cli.arrival_pattern, PhaseType.POISSON)


def _transition_phase_type(
    target: dict[str, Any], phase_type: PhaseType, decision: _PhaseShapeDecision
) -> None:
    """Change a phase discriminator and discard only incompatible YAML fields.

    Records the outcome on ``decision`` so the same discriminator and the same
    discarded keys can be replayed onto the pre-Jinja envelope, whose values
    would otherwise steer this function to a different answer.
    """
    rate_types = {
        PhaseType.POISSON,
        PhaseType.GAMMA,
        PhaseType.CONSTANT,
        PhaseType.USER_CENTRIC,
    }
    if phase_type not in rate_types:
        removed = ("rate", "rate_ramp", "rate_series", "smoothness", "users")
    else:
        removed = ("auto_offset", "start_offset", "end_offset", "rate_series")
        if phase_type != PhaseType.GAMMA:
            removed += ("smoothness",)
        if phase_type != PhaseType.USER_CENTRIC:
            removed += ("users",)
    for key in removed:
        _pop_config_value(target, key)
    target["type"] = phase_type
    decision.phase_type = phase_type
    decision.removed_keys.update(removed)


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


# CLI flags that shape the warmup phase. Derived from the section frozenset
# rather than restated so a new warmup_* flag is covered automatically.
_WARMUP_FIELDS: frozenset[str] = frozenset(
    field for field in LOADGEN_FIELDS if field.startswith("warmup_")
)


def _find_warmup_phase(phases: list[Any]) -> dict[str, Any] | None:
    """Return the YAML-declared warmup phase, if there is one."""
    for entry in phases:
        if not isinstance(entry, dict):
            continue
        kind = infer_legacy_phase_kind(entry.get("name"), entry.get("kind"))
        if kind == "warmup":
            return entry
    return None


def _apply_warmup_overrides(merged: dict[str, Any], cli: CLIConfig) -> None:
    """Overlay explicitly-set ``--warmup-*`` flags onto the warmup phase.

    ``_apply_phase_loadgen_overrides`` deliberately targets only the profiling
    phase, so that ``--request-count`` cannot clobber a warmup ramp. That left
    the warmup flags with nowhere to go: they were dropped before the
    classification gate and rejected after it.

    Three cases, matching what the CLI-only path does where it can:

    - the config file declares a warmup phase -> merge the set flags onto it,
      leaving everything the user did not mention alone;
    - it does not, but a trigger flag (--warmup-request-count /
      --warmup-num-sessions / --warmup-duration) is set -> build the phase, as
      ``convert_cli_to_aiperf`` does;
    - neither -> raise, because a secondary flag such as
      ``--warmup-concurrency`` has nothing to attach to and would otherwise be
      silently ignored.
    """
    from aiperf.config.flags._converter_warmup import build_warmup
    from aiperf.config.loader.errors import ConfigurationError

    warmup_set = cli.model_fields_set & _WARMUP_FIELDS
    if not warmup_set:
        return

    benchmark = merged.get("benchmark")
    if not isinstance(benchmark, dict):
        return
    phases = benchmark.get("phases")
    if not isinstance(phases, list) or not phases:
        return

    existing = _find_warmup_phase(phases)
    built = build_warmup(cli, base_warmup=existing is not None)
    if built is None:
        from aiperf.config.flags._config_flag_routing import flag_names_for

        names = sorted("/".join(flag_names_for(f) or (f,)) for f in warmup_set)
        raise ConfigurationError(
            f"{', '.join(names)} needs a warmup phase to apply to. Declare one "
            f"in the config file, or pass a warmup trigger "
            f"(--warmup-request-count / --warmup-num-sessions / "
            f"--warmup-duration)."
        )

    if existing is None:
        phases.insert(0, {"name": "warmup", "kind": "warmup", **built})
        return

    _drop_alias_spellings(existing, built)
    merged_warmup = deep_merge(existing, built)
    _reject_incompatible_warmup_transition(merged_warmup, warmup_set)
    existing.clear()
    existing.update(merged_warmup)


def _reject_incompatible_warmup_transition(
    merged_warmup: dict[str, Any], warmup_set: frozenset[str] | set[str]
) -> None:
    """Raise before writing back a warmup override that would merge into an
    invalid phase.

    ``_warmup_override_pattern`` emits ``rate``/``concurrency``/``type``
    independently, so a flag that only touches one side of a type transition
    can produce a structurally invalid phase: ``--warmup-request-rate`` onto
    an existing concurrency phase adds ``rate``, which ``ConcurrencyPhase``
    forbids as an extra field; ``--warmup-arrival-pattern`` onto one switches
    ``type`` to a rate phase without a ``rate``, which that phase requires.
    Passing both together is a complete, valid transition and is not
    rejected here.
    """
    from aiperf.config.loader.errors import ConfigurationError

    final_type = merged_warmup.get("type")
    if (
        "warmup_request_rate" in warmup_set
        and final_type == PhaseType.CONCURRENCY
        and merged_warmup.get("rate") is not None
    ):
        raise ConfigurationError(
            "--warmup-request-rate has no effect: the warmup phase is "
            "type: concurrency, which has no rate field. Pass "
            "--warmup-arrival-pattern too to switch it to a rate-controlled "
            "phase, or drop --warmup-request-rate."
        )
    if (
        "warmup_arrival_pattern" in warmup_set
        and final_type != PhaseType.CONCURRENCY
        and merged_warmup.get("rate") is None
        and merged_warmup.get("rate_series") is None
    ):
        raise ConfigurationError(
            f"--warmup-arrival-pattern switches the warmup phase to "
            f"type: {final_type}, which requires a rate. Pass "
            "--warmup-request-rate too, or drop --warmup-arrival-pattern."
        )


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
