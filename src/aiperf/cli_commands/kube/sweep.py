# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf kube sweep` subcommand: submit an AIPerfSweep CR to the cluster.

The sweep command reads a YAML config file that contains both the base
benchmark config (the same shape as `aiperf kube profile -f ...`) and one or
both of the optional top-level keys ``sweep:`` and ``multi_run:``. Those keys
are hoisted out of the benchmark and placed under ``AIPerfSweep.spec``; the
rest of the YAML becomes ``AIPerfSweep.spec.benchmark``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

from cyclopts import App, Parameter

from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeOptions

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
    cli_config: CLIConfig,
    kube_options: KubeOptions,
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
    an optional top-level ``sweep:`` and/or ``multi_run:`` section. Those are
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

    config_file = cli_config.config_file
    with cli_utils.exit_on_error(title="Error Running Kubernetes Sweep"):
        cr_dict = _build_sweep_cr_dict(
            config_file=config_file,
            kube_options=kube_options,
            multi_run_trials=multi_run_trials,
            cooldown_seconds=cooldown_seconds,
            convergence_metric=convergence_metric,
            convergence_min_runs=convergence_min_runs,
            convergence_max_runs=convergence_max_runs,
            convergence_threshold=convergence_threshold,
        )
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
            namespace=kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE,
        )


# Envelope-only fields that don't belong on the benchmark body.
_ENVELOPE_ONLY_KEYS = (
    "variables",
    "random_seed",
    "randomSeed",
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
    """Raw ``multi_run:``/``multiRun:`` block, or None when absent."""

    child_metadata: Any
    """AIPerfJob-CR-only ``childMetadata`` passthrough, or None."""

    envelope_extras: dict[str, Any]
    """Envelope-only keys (see ``_ENVELOPE_ONLY_KEYS``)."""


def _split_job_cr(raw: dict[str, Any]) -> _SweepYamlParts:
    """Hoist sweep/multi-run/envelope sections out of an AIPerfJob CR dict."""
    cr_spec = dict(raw.get("spec") or {})
    benchmark_raw = cr_spec.get("benchmark") or {}
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
        benchmark_raw, sweep_cfg, multirun_cfg, child_metadata, envelope_extras
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
    return _SweepYamlParts(bench_dict, sweep_cfg, multirun_cfg, None, envelope_extras)


def _normalized_benchmark_body(
    bench_dict: dict[str, Any], envelope_extras: dict[str, Any]
) -> dict[str, Any]:
    """Render templates and normalise the benchmark body to canonical long form.

    Renders Jinja2 / ${ENV_VAR} in the benchmark portion before submission so
    unresolved `{{ ... }}` literals never trip AIPerfSweepSpec.model_validate
    or reach the operator, then validates via AIPerfConfig so the body
    normalises to the canonical long-form the operator expects -- matching
    the path `kube profile` takes for CR-shaped input.
    """
    import yaml

    from aiperf.config import AIPerfConfig, dump_config
    from aiperf.config.loader import expand_config_dict

    # `variables:` is an envelope-level field — temporarily reattach it so
    # `expand_config_dict` can use it as the Jinja context, then strip back out.
    if "variables" in envelope_extras:
        bench_dict = {**bench_dict, "variables": envelope_extras["variables"]}
    bench_dict = expand_config_dict(bench_dict)
    bench_dict.pop("variables", None)
    config = AIPerfConfig.model_validate({"benchmark": bench_dict})
    return yaml.safe_load(dump_config(config)).get("benchmark", {})


def _assemble_spec(
    *,
    kube_options: KubeOptions,
    bench_dict: dict[str, Any],
    sweep_cfg: Any,
    child_metadata: Any,
    envelope_extras: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the flat AIPerfSweep spec envelope (deployment + benchmark).

    AIPerfWorkloadSpec is a flat envelope (AIPerfConfig +
    DeploymentConfig); there is no `template.spec` wrapping.
    """
    deployment = kube_options.to_deployment_config()
    deployment_dict = deployment.model_dump(
        mode="json", by_alias=True, exclude_defaults=True
    )
    spec: dict[str, Any] = {
        **deployment_dict,
        "image": kube_options.image,
        "benchmark": bench_dict,
    }
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
    """Merge the YAML ``multi_run`` config with CLI flag overrides.

    Convergence is a nested object on multi_run (Task 1-3 schema redo). CLI
    maps --convergence-metric/--convergence-threshold/--min-runs to the
    canonical ConvergenceConfig (mode defaults to ci_width); --max-runs maps
    to multi_run.num_runs (the hard cap on trials).
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
    """Round-trip spec through AIPerfSweepSpec and re-stamp the sweep type.

    Validates before submission so users see the curated AIPerfSweepSpec
    error messages, not a raw apiserver CRD validation 422. The round-trip
    through Pydantic with by_alias=True makes snake_case keys from downstream
    converters serialise as the CRD-canonical camelCase the apiserver expects
    under strict decoding (e.g. `numRuns` not `num_runs`). Without this, the
    apiserver rejects with `strict decoding error: unknown field` even though
    our local validator (which has populate_by_name=True) accepts the
    snake_case form.
    """
    from aiperf.operator.models import AIPerfSweepSpec

    validated = AIPerfSweepSpec.model_validate(spec)
    dumped = validated.model_dump(
        mode="json", by_alias=True, exclude_defaults=True, exclude_none=True
    )
    # Re-stamp the SweepConfig discriminator. ``exclude_defaults=True`` strips
    # ``type`` because every variant declares a Literal default
    # (``"adaptive_search" | "grid" | "scenarios"``), and the AIPerfSweep CRD
    # currently treats ``spec.sweep`` as a preserve-unknown-fields blob
    # without a default on the ``type`` property. The operator round-trips
    # the CR through ``AIPerfSweepSpec.model_validate`` on read; that
    # discriminated-union validation requires ``type``, so a sweep CR
    # without it fails the operator-side validator with a confusing
    # "unable to extract tag using discriminator 'type'" error.
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
    top-level ``sweep:`` and ``multi_run:`` keys are extracted and placed under
    ``AIPerfSweep.spec``; the remainder becomes ``spec.benchmark``.

    Raises:
        ValueError: ``config_file`` is None — `aiperf kube sweep` requires a
            YAML config (no flag-only invocation supported).
    """
    if config_file is None:
        raise ValueError(
            "`aiperf kube sweep` requires --config <file> with a base AIPerfConfig "
            "and optional top-level `sweep:`/`multi_run:` keys."
        )

    import yaml

    raw = yaml.safe_load(config_file.read_text()) or {}

    # `kube sweep` accepts three YAML shapes; `kube init` produces #2 today, so
    # users who follow the "init -> sweep" path land here without rewriting.
    #
    # 1. Bare AIPerfConfig YAML with optional top-level `sweep:`/`multi_run:`.
    # 2. AIPerfJob CR (apiVersion + kind=AIPerfJob): benchmark lives under
    #    `spec.benchmark`; sweep/multi_run may be there if the user added them.
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

    spec = _assemble_spec(
        kube_options=kube_options,
        bench_dict=_normalized_benchmark_body(parts.bench_dict, parts.envelope_extras),
        sweep_cfg=parts.sweep_cfg,
        child_metadata=parts.child_metadata,
        envelope_extras=parts.envelope_extras,
    )

    multirun_cfg = _merged_multirun_config(
        multirun_cfg_from_yaml=parts.multirun_cfg,
        multi_run_trials=multi_run_trials,
        cooldown_seconds=cooldown_seconds,
        convergence_metric=convergence_metric,
        convergence_min_runs=convergence_min_runs,
        convergence_max_runs=convergence_max_runs,
        convergence_threshold=convergence_threshold,
    )
    if multirun_cfg:
        sweep_cfg = parts.sweep_cfg
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
        the 30-char truncation, so a cut that lands inside a hyphen run does
        not produce ``...---sweep``), with ``"aiperf"`` substituted when the
        stem sanitizes to empty (e.g. ``___.yaml``).

    The returned string always matches DNS-1123 label rules
    (``[a-z0-9]([-a-z0-9]*[a-z0-9])?``, <=63 chars). The 30-char stem cap
    leaves head-room for ``-sweep`` (6) and any downstream
    ``-vNNNN-tNN`` suffixes (10) that ``derive_child_name`` appends.
    """
    stem = config_file.stem.lower()
    sanitized = re.sub(r"[^a-z0-9-]", "-", stem)
    # Collapse runs of '-' into a single '-' so '__a__b__' doesn't become
    # '--a--b--' which then survives the strip with embedded '--'.
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    safe_stem = sanitized[:30].rstrip("-")
    # Fall back when sanitization eats everything (all-underscore stems,
    # leading-only-special stems, empty stems). DNS-1123 labels must start
    # with [a-z0-9].
    if not safe_stem:
        safe_stem = "aiperf"
    return f"{safe_stem}-sweep"


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
        )
