# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf kube sweep` subcommand: submit an AIPerfSweep CR to the cluster.

The sweep command reads a YAML config file that contains both the base
benchmark config (the same shape as `aiperf kube profile -f ...`) and one or
both of the optional top-level keys ``sweep:`` and ``multi_run:``. Those keys
are hoisted out of the benchmark and placed under ``AIPerfSweep.spec``; the
rest of the YAML becomes ``AIPerfSweep.spec.template.spec.benchmark``.

For v1, the command only submits the CR and returns. Live tailing/attach to
sweep progress is future work.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import App, Parameter

from aiperf.config.kube import KubeOptions
from aiperf.config.v1 import ServiceConfig, UserConfig

if TYPE_CHECKING:
    from pathlib import Path

app = App(name="sweep")


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
    help="Multi-run trials per sweep cell; overrides multi_run.trials in the YAML.",
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
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
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
    level; the rest becomes ``spec.template.spec.benchmark``.

    Examples:
        # Parameter sweep declared in YAML
        aiperf kube sweep -f sweep.yaml --image aiperf:latest

        # Multi-run repeats with cooldown, no parameter axis
        aiperf kube sweep -f bench.yaml --image aiperf:latest \\
            --trials 5 --cooldown 30
    """
    from aiperf import cli_utils
    from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

    if service_config is None:
        service_config = ServiceConfig()
    # service_config is currently only consumed when we eventually load+validate
    # the YAML through the v1->v2 path; v1 sweep submits the raw YAML directly,
    # so service_config is reserved here for parity with profile/generate.
    _ = service_config

    config_file = getattr(user_config, "config_file", None)
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
    ``AIPerfSweep.spec``; the remainder becomes ``spec.template.spec.benchmark``.

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

    from aiperf.config import AIPerfConfig, dump_config
    from aiperf.config.loader import expand_config_dict

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
        cr_spec = dict(raw.get("spec") or {})
        benchmark_raw = cr_spec.get("benchmark") or {}
        sweep_cfg = cr_spec.pop("sweep", None) or benchmark_raw.pop("sweep", None)
        multirun_cfg_from_yaml = (
            cr_spec.pop("multiRun", None)
            or cr_spec.pop("multi_run", None)
            or benchmark_raw.pop("multi_run", None)
            or benchmark_raw.pop("multiRun", None)
        )
        bench_dict = benchmark_raw
    else:
        sweep_cfg = raw.pop("sweep", None)
        multirun_cfg_from_yaml = raw.pop("multi_run", None) or raw.pop("multiRun", None)
        bench_dict = raw

    # Render Jinja2 / ${ENV_VAR} in the benchmark portion before submission so
    # unresolved `{{ ... }}` literals never trip AIPerfSweepSpec.model_validate
    # below or reach the operator. Mirrors `aiperf kube profile -f`'s pipeline.
    bench_dict = expand_config_dict(bench_dict)
    # Validate via AIPerfConfig so v1->v2 shorthand promotions
    # (`model:`/`dataset:`/`phases:`) expand to the long-form the operator
    # expects -- matching the path `kube profile` takes for CR-shaped input.
    config = AIPerfConfig.model_validate(bench_dict)
    bench_dict = yaml.safe_load(dump_config(config))

    deployment = kube_options.to_deployment_config()
    deployment_dict = deployment.model_dump(
        mode="json", by_alias=True, exclude_defaults=True
    )
    template_spec: dict[str, Any] = {
        **deployment_dict,
        "image": kube_options.image,
        "benchmark": bench_dict,
    }

    spec: dict[str, Any] = {"template": {"spec": template_spec}}
    if sweep_cfg is not None:
        spec["sweep"] = sweep_cfg

    multirun_cfg: dict[str, Any] | None = None
    if multirun_cfg_from_yaml is not None:
        multirun_cfg = dict(multirun_cfg_from_yaml)
    if multi_run_trials is not None:
        multirun_cfg = multirun_cfg or {}
        # CLI flag overrides YAML, matching the documented behaviour of
        # _TRIALS_PARAM ("overrides multi_run.trials in the YAML"). Earlier
        # we used setdefault, which made YAML win silently.
        multirun_cfg["trials"] = multi_run_trials
    if cooldown_seconds:
        multirun_cfg = multirun_cfg or {}
        multirun_cfg["cooldownSeconds"] = cooldown_seconds
    if multirun_cfg:
        spec["multiRun"] = multirun_cfg

    if convergence_metric is not None:
        spec["convergence"] = {
            "metric": convergence_metric,
            "minRuns": convergence_min_runs,
            "maxRuns": convergence_max_runs,
            "threshold": convergence_threshold,
        }

    name = kube_options.name or _name_from_config_file(config_file)
    cr_dict: dict[str, Any] = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfSweep",
        "metadata": {"name": name},
        "spec": spec,
    }
    # Validate before submission so users see the curated AIPerfSweepSpec
    # error messages, not a raw apiserver CRD validation 422.
    from aiperf.kubernetes.sweep_models import AIPerfSweepSpec

    AIPerfSweepSpec.model_validate(spec)
    return cr_dict


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
    (``[a-z0-9]([-a-z0-9]*[a-z0-9])?``, ≤63 chars). The 30-char stem cap
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
