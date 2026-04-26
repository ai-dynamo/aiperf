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

from aiperf.config.cli_model import CLIModel
from aiperf.config.kube import KubeOptions

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
    cli_model: CLIModel,
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

    config_file = getattr(cli_model, "config_file", None)
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

            print(orjson.dumps(cr_dict, option=orjson.OPT_INDENT_2).decode())
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

    raw = yaml.safe_load(config_file.read_text()) or {}
    sweep_cfg = raw.pop("sweep", None)
    multirun_cfg_from_yaml = raw.pop("multi_run", None) or raw.pop("multiRun", None)

    deployment = kube_options.to_deployment_config()
    deployment_dict = deployment.model_dump(by_alias=True, exclude_defaults=True)
    template_spec: dict[str, Any] = {
        **deployment_dict,
        "image": kube_options.image,
        "benchmark": raw,
    }

    spec: dict[str, Any] = {"template": {"spec": template_spec}}
    if sweep_cfg is not None:
        spec["sweep"] = sweep_cfg

    multirun_cfg: dict[str, Any] | None = None
    if multirun_cfg_from_yaml is not None:
        multirun_cfg = dict(multirun_cfg_from_yaml)
    if multi_run_trials is not None:
        multirun_cfg = multirun_cfg or {}
        multirun_cfg.setdefault("trials", multi_run_trials)
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
    return {
        "apiVersion": "aiperf.nvidia.com/v1",
        "kind": "AIPerfSweep",
        "metadata": {"name": name},
        "spec": spec,
    }


def _name_from_config_file(config_file: Path) -> str:
    """Derive a DNS-safe AIPerfSweep name from `config_file.stem`.

    Args:
        config_file: Path to the user's sweep YAML.

    Returns:
        ``"<stem>-sweep"`` lowercased and sanitized to ``[a-z0-9-]`` with
        leading/trailing hyphens stripped, stem capped at 30 chars to leave
        room for the suffix within Kubernetes' 63-char DNS-label limit.
    """
    stem = config_file.stem.lower()
    sanitized = re.sub(r"[^a-z0-9-]", "-", stem).strip("-")
    return f"{sanitized[:30]}-sweep"


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
        await custom.create_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1",
            namespace=namespace,
            plural="aiperfsweeps",
            body=cr_dict,
        )
        kube_console.console.print(
            f"AIPerfSweep {namespace}/{cr_dict['metadata']['name']} created."
        )
