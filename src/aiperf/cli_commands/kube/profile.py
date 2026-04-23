# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube profile command: create an AIPerfJob CR to run a benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import App, Parameter

from aiperf.config.cli_model import CLIModel
from aiperf.config.kube import KubeOptions

if TYPE_CHECKING:
    from aiperf.config import AIPerfConfig

app = App(name="profile")

AIPERF_KIND = "AIPerfJob"

_DETACH_PARAM = Parameter(
    name=["-d", "--detach"],
    help="Exit immediately after deploying (don't wait for completion). Automatically enabled in non-interactive environments (pipes, CI/CD).",
)
_NO_WAIT_PARAM = Parameter(
    name="--no-wait",
    negative=(),
    help="Don't wait for pods to be ready before attaching (advanced).",
)
_ATTACH_PORT_PARAM = Parameter(
    name="--attach-port",
    help="Local port for API port-forward (default: 0 = ephemeral).",
)
_SKIP_ENDPOINT_CHECK_PARAM = Parameter(
    name="--skip-endpoint-check",
    negative=(),
    help="Skip endpoint health validation before deploying.",
)
_DRY_RUN_PARAM = Parameter(
    name="--dry-run",
    negative=(),
    help="Print the AIPerfJob CR as JSON without submitting it.",
)
_NO_OPERATOR_PARAM = Parameter(
    name="--no-operator",
    negative=(),
    help="Force direct deployment without the operator. Automatically enabled if the AIPerfJob CRD is not installed on the cluster.",
)


def _try_load_aiperfjob_cr(path: Any) -> dict | None:
    """Parse path as YAML and return the raw dict if it is an AIPerfJob CR.

    Returns None if the file cannot be parsed or is not an AIPerfJob CR.
    The caller owns the single file read; no further reads are needed.
    """
    import yaml

    try:
        raw = yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 - any YAML/IO failure means "not an AIPerfJob CR"; caller handles other paths
        return None
    if (
        isinstance(raw, dict)
        and raw.get("apiVersion", "").startswith("aiperf.nvidia.com")
        and raw.get("kind") == AIPERF_KIND
    ):
        return raw
    return None


def _build_cr_spec_and_config(raw: dict, kube_options: Any) -> tuple[dict, Any]:
    """Build (overlaid_spec, AIPerfConfig) from a parsed AIPerfJob CR dict.

    Extracts benchmark config from the CR spec, then overlays CLI K8s
    deployment options (image, podTemplate, workers, etc.) on top.
    The returned spec is ready to submit to the operator.
    """
    import copy
    import math

    from aiperf.operator.spec_converter import extract_benchmark_config

    spec = copy.deepcopy(dict(raw.get("spec", {})))
    config = extract_benchmark_config(spec)

    dc = kube_options.to_deployment_config()
    dc_dict = dc.model_dump(mode="json", by_alias=True, exclude_defaults=True)

    concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.phases.values()),
        default=1,
    )
    dc_dict["connectionsPerWorker"] = max(
        1, math.ceil(concurrency / kube_options.workers)
    )

    spec.update(dc_dict)
    return spec, config


def generate_benchmark_name(config: AIPerfConfig) -> str:
    """Generate a short benchmark name from config.

    Used by both profile and generate commands.

    Args:
        config: AIPerfConfig instance.

    Returns:
        A short hyphenated name like "qwen3-openai-throughput".
    """
    import re

    model_name = config.get_model_names()[0].split("/")[-1].lower()
    endpoint_type = str(config.endpoint.type)
    first_phase = next(iter(config.phases.values()))
    phase_type = str(first_phase.type)
    raw = "-".join([model_name, endpoint_type, phase_type])
    # Sanitize to valid DNS label: replace invalid chars, strip leading/trailing hyphens
    return re.sub(r"[^a-z0-9-]", "-", raw).strip("-")[:40]


def _resolve_config(cli_model: Any, config_file: Any) -> Any:
    """Return AIPerfConfig from a plain YAML file or CLI flags."""
    if config_file is not None:
        from aiperf.config.loader import load_config

        return load_config(config_file)
    from aiperf.config.cli_converter import build_aiperf_config

    return build_aiperf_config(cli_model)


def _resolve_spec_and_name(
    cli_model: CLIModel, kube_options: KubeOptions
) -> tuple[dict, Any, str]:
    """Resolve the AIPerfJob spec, AIPerfConfig, and benchmark name.

    Handles both paths: a raw AIPerfJob CR YAML file (CR-format) and
    plain CLI flags / benchmark config (flag-format).
    """
    config_file = getattr(cli_model, "config_file", None)
    cr_raw = _try_load_aiperfjob_cr(config_file) if config_file is not None else None
    if cr_raw is not None:
        spec, config = _build_cr_spec_and_config(cr_raw, kube_options)
        cr_name = cr_raw.get("metadata", {}).get("name")
        name = kube_options.name or cr_name or generate_benchmark_name(config)
    else:
        config = _resolve_config(cli_model, config_file)
        spec = kube_options.to_crd_spec(config)
        name = kube_options.name or generate_benchmark_name(config)
    return spec, config, name


def _print_memory_estimate(config: Any, kube_options: KubeOptions, spec: dict) -> None:
    """Compute and display the memory estimate for the planned benchmark."""
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.memory_estimator import estimate_memory, format_estimate

    mem_est = estimate_memory(
        config,
        total_workers=kube_options.workers,
        workers_per_pod=config.runtime.workers_per_pod,
        connections_per_worker=spec.get("connectionsPerWorker", 100),
    )
    kube_console.console.print(format_estimate(mem_est), highlight=False)


@app.default
async def profile(
    *,
    cli_model: CLIModel,
    kube_options: KubeOptions,
    detach: Annotated[bool, _DETACH_PARAM] = False,
    no_wait: Annotated[bool, _NO_WAIT_PARAM] = False,
    attach_port: Annotated[int, _ATTACH_PORT_PARAM] = 0,
    skip_endpoint_check: Annotated[bool, _SKIP_ENDPOINT_CHECK_PARAM] = False,
    dry_run: Annotated[bool, _DRY_RUN_PARAM] = False,
    no_operator: Annotated[bool, _NO_OPERATOR_PARAM] = False,
) -> None:
    """Run a benchmark in Kubernetes.

    Auto-detects whether the AIPerf operator is installed. If the AIPerfJob
    CRD exists, creates a CR and lets the operator handle deployment. Otherwise,
    falls back to direct manifest creation (JobSet, ConfigMap, RBAC).
    Use --no-operator to force direct mode.

    Examples:
        # Auto-detect (operator if available, direct otherwise)
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --workers-max 10

        # Force direct mode (no operator)
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --no-operator

        # CI/CD: deploy and exit immediately
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --detach
    """

    from aiperf import cli_utils
    from aiperf.cli_commands.kube.profile_deploy import (
        deploy_via_operator,
        operator_available,
    )
    from aiperf.cli_commands.kube.profile_deploy_direct import deploy_direct

    with cli_utils.exit_on_error(title="Error Running Kubernetes Benchmark"):
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        spec, config, name = _resolve_spec_and_name(cli_model, kube_options)
        namespace = kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE
        _print_memory_estimate(config, kube_options, spec)

        use_operator = not no_operator
        if use_operator and not dry_run:
            use_operator = await operator_available(kube_options)

        deploy_kwargs: dict[str, Any] = {
            "dry_run": dry_run,
            "detach": detach,
            "no_wait": no_wait,
            "attach_port": attach_port,
        }
        if use_operator:
            await deploy_via_operator(
                spec, kube_options, config, name, namespace, **deploy_kwargs
            )
        else:
            await deploy_direct(config, kube_options, name, namespace, **deploy_kwargs)
