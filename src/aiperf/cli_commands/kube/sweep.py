# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf kube sweep` subcommand: submit an AIPerfSweep CR to the cluster.

The sweep command reads a YAML config file that contains both the base
benchmark config (the same shape as `aiperf kube profile -f ...`) and one or
both of the optional top-level keys ``sweep:`` and ``multiRun:``. Those keys
are hoisted out of the benchmark and placed under ``AIPerfSweep.spec``; the
rest of the YAML becomes ``AIPerfSweep.spec.benchmark``.
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

from cyclopts import App, Parameter

from aiperf.common.path_safety import safe_read_template_path
from aiperf.config.flags.kube_cli_config import KubeCLIConfig
from aiperf.config.kube import KubeOptions
from aiperf.kubernetes.cr_refs import AIPERF_SWEEP_KIND

if TYPE_CHECKING:
    from pathlib import Path

app = App(name="sweep", help="Submit an AIPerfSweep CR to the cluster")


_DETACH_PARAM = Parameter(
    name=["-d", "--detach"],
    help="Exit after submitting (don't tail). v1 always behaves as detach=True.",
)
_DRY_RUN_PARAM = Parameter(
    name="--dry-run",
    negative=(),
    help="Print the AIPerfSweep CR as JSON without submitting it.",
)
_TRIALS_PARAM = Parameter(
    name="--trials",
    help="Multi-run runs per sweep cell; overrides multiRun.numRuns / multi_run.num_runs in the YAML.",
)
_COOLDOWN_PARAM = Parameter(
    name="--cooldown",
    help="Cooldown seconds between multi-run trials (overrides YAML).",
)
_CONV_METRIC_PARAM = Parameter(
    name="--convergence-metric",
    help="Stop multi-run early when this metric converges (e.g. ttft_p99).",
)
_CONV_MIN_PARAM = Parameter(
    name="--min-runs",
    help="Minimum runs before convergence is checked (default 3).",
)
_CONV_MAX_PARAM = Parameter(
    name="--max-runs",
    help="Hard cap on runs even if not converged (default 10).",
)
_CONV_THRESH_PARAM = Parameter(
    name="--convergence-threshold",
    help="Relative convergence threshold (default 0.05 = 5%).",
)


@app.default
async def sweep(
    *,
    cli_config: KubeCLIConfig,
    kube_options: KubeOptions | None = None,
    multi_run_trials: Annotated[int | None, _TRIALS_PARAM] = None,
    cooldown_seconds: Annotated[float, _COOLDOWN_PARAM] = 0.0,
    convergence_metric: Annotated[str | None, _CONV_METRIC_PARAM] = None,
    convergence_min_runs: Annotated[int, _CONV_MIN_PARAM] = 3,
    convergence_max_runs: Annotated[int, _CONV_MAX_PARAM] = 10,
    convergence_threshold: Annotated[float, _CONV_THRESH_PARAM] = 0.05,
    detach: Annotated[bool, _DETACH_PARAM] = False,  # noqa: ARG001 - reserved for future tailing
    dry_run: Annotated[bool, _DRY_RUN_PARAM] = False,
) -> None:
    """Submit an AIPerfSweep CR for parameter or multi-run benchmarks.

    The config file (``--config <file>``) must contain a base AIPerfConfig plus
    an optional top-level ``sweep:`` and/or ``multiRun:`` section. Those are
    hoisted out of the benchmark config and placed at the AIPerfSweep.spec
    level; the rest becomes ``spec.benchmark``.

    Examples:
        # Parameter sweep declared in YAML
        aiperf kube sweep -f sweep.yaml --image aiperf:latest

        # Multi-run repeats with cooldown, no parameter axis
        aiperf kube sweep -f bench.yaml --image aiperf:latest \\
            --trials 5 --cooldown 30
    """
    from aiperf import cli_utils
    from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

    kube_options = kube_options or KubeOptions()
    config_file = cli_config.config_file
    with cli_utils.exit_on_error(title="Error Running Kubernetes Sweep"):
        cr_dict = _build_sweep_cr_dict(
            config_file=config_file,
            cli_config=cli_config,
            kube_options=kube_options,
            multi_run_trials=multi_run_trials,
            cooldown_seconds=cooldown_seconds,
            convergence_metric=convergence_metric,
            convergence_min_runs=convergence_min_runs,
            convergence_max_runs=convergence_max_runs,
            convergence_threshold=convergence_threshold,
        )
        namespace = kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE
        cr_dict.setdefault("metadata", {})["namespace"] = namespace
        if dry_run:
            import orjson

            from aiperf.kubernetes import console as kube_console

            kube_console.console.print(
                orjson.dumps(cr_dict, option=orjson.OPT_INDENT_2).decode(),
                highlight=False,
            )
            return
        await _submit_sweep(
            cr_dict=cr_dict,
            kube_options=kube_options,
            namespace=namespace,
        )


# Envelope-only fields that don't belong on the benchmark body.
_ENVELOPE_ONLY_KEYS = (
    "plot",
    "variables",
    "random_seed",
    "randomSeed",
    "no_sweep_table",
    "noSweepTable",
    "schemaVersion",
    "schema_version",
)


class _SweepYamlParts(NamedTuple):
    """Sections hoisted out of a `kube sweep` YAML document."""

    bench_dict: dict[str, Any]
    """Benchmark body (AIPerfConfig shape, pre-normalization)."""

    sweep_cfg: Any
    """Raw ``sweep:`` block, or None when absent."""

    multirun_cfg: Any
    """Raw ``multiRun:`` block (or legacy snake_case), or None when absent."""

    child_metadata: Any
    """AIPerfJob-CR-only ``childMetadata`` passthrough, or None."""

    envelope_extras: dict[str, Any]
    """Envelope-only keys (see ``_ENVELOPE_ONLY_KEYS``)."""

    workload_extras: dict[str, Any]
    """CR deployment/orchestration fields retained outside Config-v2."""


def _split_job_cr(raw: dict[str, Any]) -> _SweepYamlParts:
    """Hoist sweep/multi-run/envelope sections out of an AIPerfJob CR dict."""
    cr_spec = copy.deepcopy(dict(raw.get("spec") or {}))
    benchmark_raw = cr_spec.pop("benchmark", {}) or {}
    sweep_cfg = cr_spec.pop("sweep", None) or benchmark_raw.pop("sweep", None)
    multirun_cfg = (
        cr_spec.pop("multiRun", None)
        or cr_spec.pop("multi_run", None)
        or benchmark_raw.pop("multi_run", None)
        or benchmark_raw.pop("multiRun", None)
    )
    child_metadata = cr_spec.pop("childMetadata", None) or cr_spec.pop(
        "child_metadata", None
    )
    envelope_extras: dict[str, Any] = {}
    for env_key in _ENVELOPE_ONLY_KEYS:
        if env_key in cr_spec:
            envelope_extras[env_key] = cr_spec.pop(env_key)
    return _SweepYamlParts(
        benchmark_raw,
        sweep_cfg,
        multirun_cfg,
        child_metadata,
        envelope_extras,
        cr_spec,
    )


def _split_bare_yaml(raw: dict[str, Any]) -> _SweepYamlParts:
    """Hoist sweep/multi-run/envelope sections out of a bare AIPerfConfig YAML."""
    sweep_cfg = raw.pop("sweep", None)
    multirun_cfg = raw.pop("multi_run", None) or raw.pop("multiRun", None)
    envelope_extras: dict[str, Any] = {}
    for env_key in _ENVELOPE_ONLY_KEYS:
        if isinstance(raw, dict) and env_key in raw:
            envelope_extras[env_key] = raw.pop(env_key)
    # Envelope YAMLs nest the body under `benchmark:`. If present, unwrap
    # so bench_dict is the body-only shape downstream code expects.
    if (
        isinstance(raw, dict)
        and "benchmark" in raw
        and isinstance(raw["benchmark"], dict)
    ):
        bench_dict = raw["benchmark"]
    else:
        bench_dict = raw
    return _SweepYamlParts(
        bench_dict,
        sweep_cfg,
        multirun_cfg,
        None,
        envelope_extras,
        {},
    )


def _normalized_config_parts(
    bench_dict: dict[str, Any],
    envelope_extras: dict[str, Any],
    *,
    sweep_cfg: Any,
    multirun_cfg: Any,
    cli_config: Any | None,
    file_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Normalize origin/main's envelope while preserving Jinja leaves.

    The rendered baseline proves the input is valid and supplies canonical
    camelCase keys. It also resolves source-relative envelope values such as a
    bare ``plot:`` path. Post-environment, pre-Jinja leaf values are overlaid
    so a ``variables.*`` sweep can render each variation independently
    in-cluster.
    """
    from aiperf.config import AIPerfConfig, load_config_from_mapping
    from aiperf.config.flags.resolver import apply_cli_overrides
    from aiperf.kubernetes.spec_converter import restore_jinja_templates

    envelope = {
        "benchmark": copy.deepcopy(bench_dict),
        **copy.deepcopy(envelope_extras),
    }
    config = load_config_from_mapping(envelope, file_path=file_path)
    if cli_config is not None:
        config = apply_cli_overrides(config, cli_config)
    rendered = config.model_dump(
        mode="json",
        by_alias=True,
        exclude_unset=True,
        exclude_none=True,
    )
    raw_envelope = config._raw_envelope
    assert raw_envelope is not None
    templated = restore_jinja_templates(rendered, raw_envelope)
    benchmark = templated.pop("benchmark")
    cli_sweep = templated.pop("sweep", None)
    cli_multirun = templated.pop("multiRun", None)
    from aiperf.config.flags.resolver import deep_merge

    normalized_sweep = copy.deepcopy(sweep_cfg)
    if isinstance(normalized_sweep, dict) and isinstance(cli_sweep, dict):
        normalized_sweep = deep_merge(normalized_sweep, cli_sweep)
    elif cli_sweep is not None:
        normalized_sweep = cli_sweep
    normalized_multirun = copy.deepcopy(multirun_cfg)
    if isinstance(normalized_multirun, dict) and isinstance(cli_multirun, dict):
        normalized_multirun = deep_merge(normalized_multirun, cli_multirun)
    elif cli_multirun is not None:
        normalized_multirun = cli_multirun
    normalized_extras: dict[str, Any] = {}
    for name, model_field in AIPerfConfig.model_fields.items():
        if name in {"benchmark", "sweep", "multi_run"}:
            continue
        wire_name = model_field.alias or name
        if wire_name in templated:
            normalized_extras[wire_name] = templated[wire_name]
    return benchmark, normalized_extras, normalized_sweep, normalized_multirun


def _assemble_spec(
    *,
    kube_options: KubeOptions,
    bench_dict: dict[str, Any],
    sweep_cfg: Any,
    child_metadata: Any,
    envelope_extras: dict[str, Any],
    workload_extras: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the flat AIPerfSweep spec envelope (deployment + benchmark).

    AIPerfWorkloadSpec is a flat envelope (AIPerfConfig +
    DeploymentConfig); there is no `template.spec` wrapping.
    """
    from aiperf.config.flags.resolver import deep_merge

    deployment = kube_options.to_deployment_config()
    deployment_dict = deployment.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )
    spec: dict[str, Any] = copy.deepcopy(workload_extras)
    spec["benchmark"] = bench_dict
    spec = deep_merge(spec, deployment_dict)
    kube_options.apply_total_workers_override(spec)
    if sweep_cfg is not None:
        spec["sweep"] = sweep_cfg
    if child_metadata is not None:
        spec["childMetadata"] = child_metadata
    # Envelope-level fields (variables, random_seed) flow onto the spec
    # directly, mirroring AIPerfConfig's shape.
    for key, value in envelope_extras.items():
        spec[key] = value
    return spec


def _merged_multirun_config(
    *,
    multirun_cfg_from_yaml: Any,
    multi_run_trials: int | None,
    cooldown_seconds: float,
    convergence_metric: str | None,
    convergence_min_runs: int,
    convergence_max_runs: int,
    convergence_threshold: float,
) -> dict[str, Any] | None:
    """Merge the YAML ``multiRun`` config with CLI flag overrides.

    Convergence is a nested object on ``multiRun``. CLI
    maps --convergence-metric/--convergence-threshold/--min-runs to the
    canonical ConvergenceConfig (mode defaults to ci_width); --max-runs maps
    to ``multiRun.numRuns`` (the hard cap on trials).
    """
    multirun_cfg: dict[str, Any] | None = None
    if multirun_cfg_from_yaml is not None:
        multirun_cfg = dict(multirun_cfg_from_yaml)
    if multi_run_trials is not None:
        multirun_cfg = multirun_cfg or {}
        # CLI flag overrides YAML.
        multirun_cfg["numRuns"] = multi_run_trials
    if cooldown_seconds:
        multirun_cfg = multirun_cfg or {}
        multirun_cfg["cooldownSeconds"] = cooldown_seconds
    if convergence_metric is not None:
        multirun_cfg = multirun_cfg or {}
        existing = multirun_cfg.get("numRuns") or multirun_cfg.get("num_runs") or 1
        multirun_cfg["numRuns"] = max(int(existing), convergence_max_runs)
        multirun_cfg["convergence"] = {
            "metric": convergence_metric,
            "minRuns": convergence_min_runs,
            "threshold": convergence_threshold,
        }
    return multirun_cfg


def _finalized_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the rendered workload and emit its templated wire form.

    Validates before submission so users see the curated AIPerfSweepSpec
    error messages, not a raw apiserver CRD validation 422. Canonical Pydantic
    serialization supplies strict-decoding camelCase keys; only Jinja leaf
    values are restored afterward for per-variation rendering.
    """
    from aiperf.config import AIPerfConfig
    from aiperf.kubernetes.crd_models import AIPerfSweepSpec
    from aiperf.kubernetes.spec_converter import (
        prepare_workload_spec,
        restore_jinja_templates,
    )

    prepared_spec, raw_envelope = prepare_workload_spec(spec)
    validated = AIPerfSweepSpec.model_validate(prepared_spec)
    from aiperf.common.endpoint_credentials import (
        validate_kubernetes_credential_transport,
    )

    validate_kubernetes_credential_transport(
        validated.benchmark.endpoint, validated.pod_template.env
    )
    dumped = validated.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )
    for name, model_field in AIPerfConfig.model_fields.items():
        if name not in raw_envelope:
            continue
        wire_name = model_field.alias or name
        if wire_name in dumped:
            dumped[wire_name] = restore_jinja_templates(
                dumped[wire_name], raw_envelope[name]
            )
    # Direct/programmatic configs can omit a defaulted discriminator from
    # their authored field set. The CRD union still requires the tag.
    sweep_dict = dumped.get("sweep")
    if (
        isinstance(sweep_dict, dict)
        and "type" not in sweep_dict
        and validated.sweep is not None
    ):
        sweep_dict["type"] = type(validated.sweep).model_fields["type"].default
    return dumped


def _build_sweep_cr_dict(
    *,
    config_file: Path | None,
    cli_config: Any | None = None,
    kube_options: KubeOptions,
    multi_run_trials: int | None,
    cooldown_seconds: float,
    convergence_metric: str | None,
    convergence_min_runs: int,
    convergence_max_runs: int,
    convergence_threshold: float,
) -> dict[str, Any]:
    """Build an AIPerfSweep CR dict from a YAML config file with sweep config.

    The config file must contain (at minimum) a base AIPerfConfig. Optional
    top-level ``sweep:`` and ``multiRun:`` keys are extracted and placed under
    ``AIPerfSweep.spec``; the remainder becomes ``spec.benchmark``.

    Raises:
        ValueError: ``config_file`` is None — `aiperf kube sweep` requires a
            YAML config (no flag-only invocation supported).
    """
    if config_file is None:
        raise ValueError(
            "`aiperf kube sweep` requires --config <file> with a base AIPerfConfig "
            "and a top-level `sweep:` or `multiRun:` block."
        )

    import yaml

    text = safe_read_template_path(str(config_file))
    if text is None:
        raise ValueError(f"Cannot safely read sweep config: {config_file}")
    raw = yaml.safe_load(text) or {}

    # `kube sweep` accepts three YAML shapes; `kube init` produces #2 today, so
    # users who follow the "init -> sweep" path land here without rewriting.
    #
    # 1. Bare AIPerfConfig YAML with optional top-level `sweep:`/`multiRun:`.
    # 2. AIPerfJob CR (apiVersion + kind=AIPerfJob): benchmark lives under
    #    `spec.benchmark`; sweep/multiRun may be there if the user added them.
    # 3. AIPerfSweep CR: rejected -- if it's already a sweep CR, the user
    #    should `kubectl apply -f` directly rather than re-build it.
    is_aiperf_cr = (
        isinstance(raw, dict)
        and isinstance(raw.get("apiVersion"), str)
        and raw["apiVersion"].startswith("aiperf.nvidia.com")
    )
    if is_aiperf_cr and raw.get("kind") == "AIPerfSweep":
        raise ValueError(
            f"'{config_file}' is already an AIPerfSweep CR. Use "
            f"`kubectl apply -f {config_file}` to submit it, or pass a plain "
            f"AIPerfConfig YAML / AIPerfJob CR to have `aiperf kube sweep` "
            f"build the sweep CR."
        )
    if is_aiperf_cr and raw.get("kind") == "AIPerfJob":
        parts = _split_job_cr(raw)
    else:
        parts = _split_bare_yaml(raw)

    (
        normalized_benchmark,
        normalized_extras,
        normalized_sweep,
        normalized_multirun,
    ) = _normalized_config_parts(
        parts.bench_dict,
        parts.envelope_extras,
        sweep_cfg=parts.sweep_cfg,
        multirun_cfg=parts.multirun_cfg,
        cli_config=cli_config,
        file_path=config_file,
    )
    spec = _assemble_spec(
        kube_options=kube_options,
        bench_dict=normalized_benchmark,
        sweep_cfg=normalized_sweep,
        child_metadata=parts.child_metadata,
        envelope_extras=normalized_extras,
        workload_extras=parts.workload_extras,
    )

    multirun_cfg = _merged_multirun_config(
        multirun_cfg_from_yaml=normalized_multirun,
        multi_run_trials=multi_run_trials,
        cooldown_seconds=cooldown_seconds,
        convergence_metric=convergence_metric,
        convergence_min_runs=convergence_min_runs,
        convergence_max_runs=convergence_max_runs,
        convergence_threshold=convergence_threshold,
    )
    if multirun_cfg is not None:
        sweep_cfg = normalized_sweep
        if sweep_cfg is None:
            from aiperf.kubernetes.sweep_routing import one_cell_sweep

            sweep_cfg = one_cell_sweep(convergence="convergence" in multirun_cfg)
            spec["sweep"] = sweep_cfg
        # Mutating sweep_cfg also updates spec["sweep"] (same dict object).
        if (
            isinstance(sweep_cfg, dict)
            and "convergence" in multirun_cfg
            and "iteration_order" not in sweep_cfg
            and "iterationOrder" not in sweep_cfg
        ):
            sweep_cfg["iterationOrder"] = "independent"
        spec["multiRun"] = multirun_cfg

    name = kube_options.name or _name_from_config_file(config_file)
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfSweep",
        "metadata": {"name": name},
        "spec": _finalized_spec(spec),
    }


def _name_from_config_file(config_file: Path) -> str:
    """Derive a DNS-1123 label name from ``config_file.stem``.

    Args:
        config_file: Path to the user's sweep YAML.

    Returns:
        ``"<stem>-sweep"``: lowercased, every disallowed char collapsed into a
        single ``-``, leading/trailing ``-`` stripped (both before AND after
        truncation, so a cut that lands inside a hyphen run does
        not produce ``...---sweep``), with ``"aiperf"`` substituted when the
        stem sanitizes to empty (e.g. ``___.yaml``).

    The returned string always matches DNS-1123 label rules
    (``[a-z0-9]([-a-z0-9]*[a-z0-9])?``) and leaves enough headroom for the
    largest child suffix accepted by the operator.
    """
    from aiperf.sweep_controller._naming import MAX_SWEEP_NAME_LENGTH

    generated_suffix = "-sweep"
    stem_budget = MAX_SWEEP_NAME_LENGTH - len(generated_suffix)
    stem = config_file.stem.lower()
    sanitized = re.sub(r"[^a-z0-9-]", "-", stem)
    # Collapse runs of '-' into a single '-' so '__a__b__' doesn't become
    # '--a--b--' which then survives the strip with embedded '--'.
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    safe_stem = sanitized[:stem_budget].rstrip("-")
    # Fall back when sanitization eats everything (all-underscore stems,
    # leading-only-special stems, empty stems). DNS-1123 labels must start
    # with [a-z0-9].
    if not safe_stem:
        safe_stem = "aiperf"
    return f"{safe_stem}{generated_suffix}"


async def _submit_sweep(
    *,
    cr_dict: dict[str, Any],
    kube_options: KubeOptions,
    namespace: str,
) -> None:
    """Apply the AIPerfSweep CR to the cluster via CustomObjectsApi."""
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import k8s_client

    cr_dict["metadata"]["namespace"] = namespace
    async with k8s_client(
        kubeconfig=getattr(kube_options, "kubeconfig", None),
        context=getattr(kube_options, "kube_context", None),
    ) as api:
        custom = k8s.CustomObjectsApi(api)
        try:
            await custom.create_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="aiperfsweeps",
                body=cr_dict,
            )
        except k8s.ApiException as e:
            if getattr(e, "status", None) == 409:
                raise RuntimeError(
                    f"AIPerfSweep {namespace}/{cr_dict['metadata']['name']} "
                    "already exists. Pass --name to choose a different name, "
                    "or delete the existing CR first."
                ) from e
            raise
        kube_console.console.print(
            f"AIPerfSweep {namespace}/{cr_dict['metadata']['name']} created."
        )
        # Persist for `aiperf kube last-benchmark` parity with profile.
        kube_console.save_last_benchmark(
            cr_dict["metadata"]["name"],
            namespace,
            name=getattr(kube_options, "name", None),
            kind=AIPERF_SWEEP_KIND,
        )
