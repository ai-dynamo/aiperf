# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube profile command: create an AIPerfJob CR to run a benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import App, Parameter

from aiperf.cli_commands.kube._kube_common import (
    generate_benchmark_name,
    print_memory_estimate,
)
from aiperf.common.path_safety import safe_read_template_path
from aiperf.config.flags.kube_cli_config import KubeCLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.kube import KubeOptions

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config import AIPerfConfig

# Re-exported for back-compat — external callers historically imported it from here.
__all__ = ["app", "generate_benchmark_name"]

app = App(name="profile")

AIPERF_KIND = "AIPerfJob"

_DETACH_PARAM = Parameter(
    name=["-d", "--detach"],
    help="Exit immediately after deploying (don't wait for completion). Automatically enabled in non-interactive environments (pipes, CI/CD).",
)
_NO_WAIT_PARAM = Parameter(
    name="--no-wait",
    negative=(),
    help="Don't wait for pods to be ready before attaching (advanced). In operator mode this returns as soon as the AIPerfJob is created.",
)
_ATTACH_PORT_PARAM = Parameter(
    name="--attach-port",
    help="Local port for API port-forward (default: 0 = ephemeral). Direct mode only; operator mode watches the AIPerfJob instead of port-forwarding.",
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
_OPERATOR_PARAM = Parameter(
    name="--operator",
    negative=(),
    help="Force operator deployment without probing the cluster-scoped AIPerfJob CRD. Use this with a pre-provisioned namespace when RBAC is namespace-scoped.",
)


def _try_load_aiperfjob_cr(path: Any) -> dict | None:
    """Parse path as YAML and return the raw dict if it is an AIPerfJob CR.

    Returns None if the file cannot be parsed or is not an AIPerfJob CR.
    The caller owns the single file read; no further reads are needed.
    """
    import yaml

    text = safe_read_template_path(str(path))
    if text is None:
        return None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if (
        isinstance(raw, dict)
        and raw.get("apiVersion", "").startswith("aiperf.nvidia.com")
        and raw.get("kind") == AIPERF_KIND
    ):
        return raw
    return None


def _build_cr_spec_and_config(
    raw: dict,
    kube_options: Any,
    cli_config: Any | None = None,
) -> tuple[dict, Any]:
    """Build (overlaid_spec, AIPerfConfig) from a parsed AIPerfJob CR dict.

    Extracts Config-v2 from the CR spec (rendering env vars + Jinja2), applies
    the same explicit benchmark CLI overrides as a plain config file, then
    overlays CLI Kubernetes deployment options. The complete rendered config
    envelope replaces its raw CR fields so aliases and override precedence are
    identical across both input shapes.
    """
    import copy
    import math

    from aiperf.config import AIPerfConfig
    from aiperf.config.flags.resolver import apply_cli_overrides, deep_merge
    from aiperf.kubernetes.spec_converter import extract_benchmark_config

    spec = copy.deepcopy(dict(raw.get("spec", {})))
    config = extract_benchmark_config(spec)
    if cli_config is not None:
        config = apply_cli_overrides(config, cli_config)

    # Replace every Config-v2 envelope field, not just benchmark. This keeps
    # variables/randomSeed/plot aliases canonical and ensures CLI envelope
    # overrides reach the workload instead of only affecting local sizing.
    envelope = config.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )
    kube_options.apply_total_workers_override(envelope)
    for name, model_field in AIPerfConfig.model_fields.items():
        spec.pop(name, None)
        if model_field.alias is not None:
            spec.pop(model_field.alias, None)
    spec.update(envelope)

    dc = kube_options.to_deployment_config()
    dc_dict = dc.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )

    # Only derive connectionsPerWorker when the user actually asked for a
    # worker count. `total_workers` defaults to 10, so deriving unconditionally
    # silently overwrote whatever the CR YAML declared -- a CR carrying
    # connectionsPerWorker: 500 was submitted as 200. KubeOptions.to_crd_spec
    # guards on model_fields_set for exactly this reason.
    if (
        "total_workers" in kube_options.model_fields_set
        and kube_options.total_workers > 0
    ):
        concurrency = max(
            (
                getattr(phase, "concurrency", 1) or 1
                for phase in config.benchmark.phases
            ),
            default=1,
        )
        dc_dict["connectionsPerWorker"] = max(
            1, math.ceil(concurrency / kube_options.total_workers)
        )

    # A CLI node selector must not erase unrelated YAML podTemplate fields
    # such as affinity, volumes, or securityContext.
    spec = deep_merge(spec, dc_dict)
    return spec, config


def _resolve_config(
    cli_config: KubeCLIConfig,
    config_file: Path | None,
) -> AIPerfConfig:
    """Backwards-compatible alias for `_kube_common.resolve_config`."""
    return resolve_config(cli_config, config_file)


def _resolve_spec_and_name(
    cli_config: KubeCLIConfig,
    kube_options: KubeOptions,
) -> tuple[dict, Any, str]:
    """Resolve the AIPerfJob spec, AIPerfConfig, and benchmark name.

    Handles both paths: a raw AIPerfJob CR YAML file (CR-format) and
    plain CLI flags / benchmark config (flag-format).
    """
    config_file = cli_config.config_file
    cr_raw = _try_load_aiperfjob_cr(config_file) if config_file is not None else None
    if cr_raw is not None:
        spec, config = _build_cr_spec_and_config(
            cr_raw, kube_options, cli_config=cli_config
        )
        cr_name = cr_raw.get("metadata", {}).get("name")
        name = kube_options.name or cr_name or generate_benchmark_name(config)
    else:
        config = resolve_config(cli_config, config_file)
        spec = kube_options.to_crd_spec(config)
        name = kube_options.name or generate_benchmark_name(config)
    return spec, config, name


def _print_memory_estimate(config: Any, kube_options: KubeOptions, spec: dict) -> None:
    """Backwards-compatible alias for `_kube_common.print_memory_estimate`."""
    print_memory_estimate(config, kube_options, spec)


def _check_no_sweep_keys(config_dict: dict, *, source: str) -> None:
    """Hand off parameter sweeps and real multi-run configs to ``kube sweep``.

    Explicit single-run multi-run defaults remain valid for AIPerfJob; only
    ``numRuns > 1`` or convergence requires cluster-side orchestration.

    Args:
        config_dict: Parsed YAML dict from the user's config file.
        source: Path-or-label used in the error message to identify the file.

    Raises:
        SystemExit: when any forbidden key is present (via
            `cli_utils.raise_startup_error_and_exit`).
    """
    sweep_key = "sweep" if "sweep" in config_dict else None
    multi_run_key = next(
        (key for key in ("multi_run", "multiRun") if key in config_dict), None
    )
    multi_run_requires_controller = False
    if multi_run_key is not None:
        from pydantic import ValidationError

        from aiperf.config.sweep.multi_run import MultiRunConfig
        from aiperf.kubernetes.sweep_routing import requires_multiple_trials

        try:
            multi_run = MultiRunConfig.model_validate(config_dict[multi_run_key] or {})
        except (TypeError, ValidationError):
            multi_run_requires_controller = True
        else:
            multi_run_requires_controller = requires_multiple_trials(multi_run)

    if sweep_key is None and not multi_run_requires_controller:
        return
    from aiperf import cli_utils

    key = sweep_key or multi_run_key
    workload = "parameter sweep" if sweep_key is not None else "multi-run workload"
    cli_utils.raise_startup_error_and_exit(
        f"This config ({source}) defines a {workload} with '{key}:', but "
        f"`aiperf kube profile` submits one AIPerfJob.\n"
        f"Use `aiperf kube sweep -f <config>` to run it through the Kubernetes "
        f"sweep controller, or remove the '{key}:' settings to run once.",
        title="Kubernetes sweep controller required",
    )


def _check_config_file_for_sweep_keys(config_file: Path | None) -> None:
    """Enforce the profile/sweep boundary for plain YAML and workload CRs.

    AIPerfJob input is inspected for envelope-level multi-run settings;
    existing AIPerfSweep input is redirected to ``kubectl apply``.
    """
    if config_file is None:
        return
    import yaml

    text = safe_read_template_path(str(config_file))
    if text is None:
        return
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return
    if not isinstance(raw, dict):
        return
    if (
        raw.get("apiVersion", "").startswith("aiperf.nvidia.com")
        and raw.get("kind") == "AIPerfSweep"
    ):
        from aiperf import cli_utils

        cli_utils.raise_startup_error_and_exit(
            f"This config ({config_file}) is an AIPerfSweep CR, but "
            f"`aiperf kube profile` only handles single AIPerfJob benchmarks.\n"
            f"Use `kubectl apply -f {config_file}` to submit the existing CR.",
            title="AIPerfSweep CR detected",
        )
    if (
        raw.get("apiVersion", "").startswith("aiperf.nvidia.com")
        and raw.get("kind") == AIPERF_KIND
    ):
        _check_no_sweep_keys(raw.get("spec") or {}, source=str(config_file))
        return
    _check_no_sweep_keys(raw, source=str(config_file))


def _check_resolved_config_for_sweep(config: AIPerfConfig) -> None:
    """Reject CLI-generated orchestration that was absent from the YAML file."""
    from aiperf.kubernetes.sweep_routing import requires_sweep_controller

    if not requires_sweep_controller(config):
        return
    from aiperf import cli_utils

    cli_utils.raise_startup_error_and_exit(
        "The resolved CLI/config combination defines a parameter sweep or "
        "multi-run workload, but `aiperf kube profile` submits one AIPerfJob.\n"
        "Use `aiperf kube sweep -f <config>` or `aiperf kube generate --operator` "
        "for cluster-side orchestration.",
        title="Kubernetes sweep controller required",
    )


@app.default
async def profile(
    *,
    cli_config: KubeCLIConfig,
    kube_options: KubeOptions | None = None,
    detach: Annotated[bool, _DETACH_PARAM] = False,
    no_wait: Annotated[bool, _NO_WAIT_PARAM] = False,
    attach_port: Annotated[int, _ATTACH_PORT_PARAM] = 0,
    skip_endpoint_check: Annotated[bool, _SKIP_ENDPOINT_CHECK_PARAM] = False,
    dry_run: Annotated[bool, _DRY_RUN_PARAM] = False,
    operator: Annotated[bool, _OPERATOR_PARAM] = False,
    no_operator: Annotated[bool, _NO_OPERATOR_PARAM] = False,
) -> None:
    """Run a benchmark in Kubernetes.

    Auto-detects whether the AIPerf operator is installed. If the AIPerfJob
    CRD exists, creates a CR and lets the operator handle deployment. Otherwise,
    falls back to direct manifest creation (JobSet, ConfigMap, RBAC).
    Use --operator to force operator mode without CRD discovery, or
    --no-operator to force direct mode.

    Examples:
        # Auto-detect (operator if available, direct otherwise)
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --total-workers 10

        # Force direct mode (no operator)
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest --no-operator

        # Force operator mode for namespace-scoped cluster access
        aiperf kube profile --model Qwen/Qwen3-0.6B \\
            --url http://server:8000 --image aiperf:latest \\
            --namespace team-a --operator

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

    if operator and no_operator:
        cli_utils.raise_startup_error_and_exit(
            "Cannot use both --operator and --no-operator",
            title="Error Running Kubernetes Benchmark",
        )

    kube_options = kube_options or KubeOptions()
    with cli_utils.exit_on_error(title="Error Running Kubernetes Benchmark"):
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        _check_config_file_for_sweep_keys(cli_config.config_file)
        spec, config, name = _resolve_spec_and_name(cli_config, kube_options)
        _check_resolved_config_for_sweep(config)
        namespace = kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE
        _print_memory_estimate(config, kube_options, spec)

        use_operator = not no_operator
        if use_operator and not operator and not dry_run:
            use_operator = await operator_available(kube_options)

        deploy_kwargs: dict[str, Any] = {
            "dry_run": dry_run,
            "detach": detach,
            "no_wait": no_wait,
            "attach_port": attach_port,
            "skip_endpoint_check": skip_endpoint_check,
        }
        if use_operator:
            await deploy_via_operator(
                spec, kube_options, config, name, namespace, **deploy_kwargs
            )
        else:
            await deploy_direct(
                config,
                kube_options,
                name,
                namespace,
                deployment_spec=spec,
                **deploy_kwargs,
            )
