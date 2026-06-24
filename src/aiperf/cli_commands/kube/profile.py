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
from aiperf.config.flags import CLIConfig
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

    Extracts benchmark config from the CR spec (rendering env vars + Jinja2
    templates the same way ``aiperf kube show`` does), then overlays CLI K8s
    deployment options (image, podTemplate, workers, etc.) on top. The
    rendered benchmark replaces the raw, template-laden ``spec.benchmark``
    so the submitted CR contains only resolved scalars (no ``{{ ... }}``
    literals that would fail operator-side Pydantic validation).
    """
    import copy
    import math

    import yaml

    from aiperf.config import dump_config
    from aiperf.operator.spec_converter import extract_benchmark_config

    spec = copy.deepcopy(dict(raw.get("spec", {})))
    config = extract_benchmark_config(spec)
    # Mirror aiperf/cli_commands/kube/show.py: replace the raw benchmark
    # block with a fully-rendered version derived from the validated
    # AIPerfConfig so unresolved Jinja templates never reach the operator.
    spec["benchmark"] = yaml.safe_load(dump_config(config)).get("benchmark", {})

    dc = kube_options.to_deployment_config()
    dc_dict = dc.model_dump(mode="json", by_alias=True, exclude_defaults=True)

    concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.benchmark.phases),
        default=1,
    )
    dc_dict["connectionsPerWorker"] = max(
        1, math.ceil(concurrency / kube_options.workers)
    )

    spec.update(dc_dict)
    return spec, config


def _resolve_config(
    cli_config: CLIConfig,
    config_file: Path | None,
) -> AIPerfConfig:
    """Backwards-compatible alias for `_kube_common.resolve_config`."""
    return resolve_config(cli_config, config_file)


def _resolve_spec_and_name(
    cli_config: CLIConfig,
    kube_options: KubeOptions,
) -> tuple[dict, Any, str]:
    """Resolve the AIPerfJob spec, AIPerfConfig, and benchmark name.

    Handles both paths: a raw AIPerfJob CR YAML file (CR-format) and
    plain CLI flags / benchmark config (flag-format).
    """
    config_file = cli_config.config_file
    cr_raw = _try_load_aiperfjob_cr(config_file) if config_file is not None else None
    if cr_raw is not None:
        spec, config = _build_cr_spec_and_config(cr_raw, kube_options)
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
    """Fail fast with a hand-off message if `config_dict` has sweep/multi_run keys.

    `aiperf kube profile` runs a single benchmark; sweep/multi_run configs
    must go through `aiperf kube sweep` instead. Detected on any of:
    ``sweep``, ``multi_run``, or ``multiRun``.

    Args:
        config_dict: Parsed YAML dict from the user's config file.
        source: Path-or-label used in the error message to identify the file.

    Raises:
        SystemExit: when any forbidden key is present (via
            `cli_utils.raise_startup_error_and_exit`).
    """
    forbidden = [k for k in ("sweep", "multi_run", "multiRun") if k in config_dict]
    if not forbidden:
        return
    from aiperf import cli_utils

    cli_utils.raise_startup_error_and_exit(
        f"This config ({source}) has '{forbidden[0]}:' set, but "
        f"`aiperf kube profile` runs a single benchmark.\n"
        f"Use `aiperf kube sweep -f <config>` to run it as a parameter sweep, "
        f"or remove the '{forbidden[0]}:' key to run a single benchmark.",
        title="Sweep config detected",
    )


def _check_config_file_for_sweep_keys(config_file: Path | None) -> None:
    """If `config_file` is a plain YAML config (not an AIPerfJob CR), enforce no-sweep.

    Skips when no config file was given, when the file is unparsable, or
    when the file is itself an AIPerfJob CR (which is handled by the CR path).
    Redirects when the file is an AIPerfSweep CR — those belong to
    `aiperf kube sweep`, not `aiperf kube profile`.
    """
    if config_file is None:
        return
    import yaml

    try:
        raw = yaml.safe_load(config_file.read_text())
    except Exception:  # noqa: BLE001 - unparsable YAML surfaces later via load_config
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
            f"Use `aiperf kube sweep -f {config_file}` to submit it.",
            title="AIPerfSweep CR detected",
        )
    if (
        raw.get("apiVersion", "").startswith("aiperf.nvidia.com")
        and raw.get("kind") == AIPERF_KIND
    ):
        return
    _check_no_sweep_keys(raw, source=str(config_file))


@app.default
async def profile(
    *,
    cli_config: CLIConfig,
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

        _check_config_file_for_sweep_keys(cli_config.config_file)
        spec, config, name = _resolve_spec_and_name(cli_config, kube_options)
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
            "skip_endpoint_check": skip_endpoint_check,
        }
        if use_operator:
            await deploy_via_operator(
                spec, kube_options, config, name, namespace, **deploy_kwargs
            )
        else:
            await deploy_direct(config, kube_options, name, namespace, **deploy_kwargs)
