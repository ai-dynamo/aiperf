# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube setup command: prepare a cluster to run AIPerf benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

app = App(name="setup")

SOURCE_CHART_PATH = Path("deploy/helm/aiperf-operator")
"""Chart location when running from the repository root."""

PACKAGED_CHART_PATH = (
    Path(__file__).resolve().parents[2] / "kubernetes" / "helm" / "aiperf-operator"
)
"""Chart location populated by the wheel build's force-include rule."""


def _default_chart_path() -> Path:
    """Resolve the bundled chart, retaining source-checkout convenience."""
    if PACKAGED_CHART_PATH.is_dir():
        return PACKAGED_CHART_PATH
    return SOURCE_CHART_PATH


def _kubectl_base(manage_options: KubeManageOptions) -> list[str]:
    """kubectl invocation carrying the caller's kubeconfig/context selection."""
    cmd = ["kubectl"]
    if manage_options.kubeconfig:
        cmd += ["--kubeconfig", manage_options.kubeconfig]
    if manage_options.kube_context:
        cmd += ["--context", manage_options.kube_context]
    return cmd


async def _ensure_jobset(
    manage_options: KubeManageOptions,
    kubectl: list[str],
    *,
    jobset_version: str | None,
    dry_run: bool,
) -> None:
    """Install the JobSet CRD when the selected cluster does not have it."""
    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
    from aiperf.kubernetes.jobset_urls import (
        JOBSET_FALLBACK_VERSION,
        get_jobset_manifest_url,
        get_latest_jobset_version,
    )
    from aiperf.kubernetes.subproc import run_command

    kube_console.print_step("Checking JobSet CRD")
    async with k8s_client(
        kubeconfig=manage_options.kubeconfig,
        context=manage_options.kube_context,
    ) as api:
        custom = k8s_client_mod.CustomObjectsApi(api)
        try:
            await custom.list_cluster_custom_object(
                group=JOBSET_GROUP, version=JOBSET_VERSION, plural=JOBSET_PLURAL
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
        else:
            kube_console.print_success("JobSet CRD already installed")
            return

    version = jobset_version or await get_latest_jobset_version()
    if version is None:
        version = JOBSET_FALLBACK_VERSION
        kube_console.print_warning(
            f"Could not reach GitHub for the latest JobSet release; "
            f"using the pinned fallback {version}"
        )
    if dry_run:
        kube_console.print_info(f"--dry-run: would install JobSet {version}")
        return

    kube_console.print_step(f"Installing JobSet {version}")
    result = await run_command(
        [*kubectl, "apply", "--server-side", "-f", get_jobset_manifest_url(version)]
    )
    if result.returncode != 0:
        kube_console.print_error(
            f"JobSet install failed: {result.stderr.strip() or result.stdout.strip()}"
        )
        raise SystemExit(1)
    kube_console.print_success(f"Installed JobSet {version}")


async def _ensure_namespaces(
    manage_options: KubeManageOptions,
    kubectl: list[str],
    namespaces: tuple[str, ...],
    *,
    dry_run: bool,
) -> None:
    """Create each missing namespace once, preserving caller order."""
    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.subproc import run_command

    for namespace in dict.fromkeys(namespaces):
        async with k8s_client(
            kubeconfig=manage_options.kubeconfig,
            context=manage_options.kube_context,
        ) as api:
            core = k8s_client_mod.CoreV1Api(api)
            try:
                await core.read_namespace(name=namespace)
            except ApiException as exc:
                if exc.status != 404:
                    raise
            else:
                kube_console.print_success(f"Namespace {namespace} already exists")
                continue
        if dry_run:
            kube_console.print_info(f"--dry-run: would create namespace {namespace}")
            continue
        result = await run_command([*kubectl, "create", "namespace", namespace])
        if result.returncode != 0:
            kube_console.print_error(
                f"Could not create namespace {namespace}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
            raise SystemExit(1)
        kube_console.print_success(f"Created namespace {namespace}")


async def _install_operator(
    manage_options: KubeManageOptions,
    operator_namespace: str,
    benchmark_namespace: str,
    *,
    chart: Path | None,
    skip_operator: bool,
    dry_run: bool,
) -> None:
    """Install or upgrade the operator after prerequisites are ready."""
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.subproc import run_command

    if skip_operator:
        kube_console.print_info("--skip-operator: leaving the operator alone")
        _print_next_steps(kube_console, benchmark_namespace)
        return

    chart_path = chart or _default_chart_path()
    if not chart_path.is_dir():
        kube_console.print_error(
            f"Helm chart not found at {chart_path}. Reinstall AIPerf to restore "
            "the bundled chart, or pass --chart PATH to a valid aiperf-operator chart."
        )
        raise SystemExit(1)

    helm = [
        "helm",
        "upgrade",
        "--install",
        "aiperf-operator",
        str(chart_path),
        "--namespace",
        operator_namespace,
        "--set-string",
        f"benchmarkNamespace.name={benchmark_namespace}",
        "--set",
        "benchmarkNamespace.create=false",
    ]
    if manage_options.kubeconfig:
        helm += ["--kubeconfig", manage_options.kubeconfig]
    if manage_options.kube_context:
        helm += ["--kube-context", manage_options.kube_context]
    if dry_run:
        kube_console.print_info(f"--dry-run: would run `{' '.join(helm)}`")
        _print_next_steps(kube_console, benchmark_namespace)
        return

    kube_console.print_step(f"Installing the AIPerf operator into {operator_namespace}")
    result = await run_command(helm, timeout=300.0)
    if result.returncode != 0:
        kube_console.print_error(
            f"Operator install failed: {result.stderr.strip() or result.stdout.strip()}"
        )
        raise SystemExit(1)
    kube_console.print_success(f"AIPerf operator installed into {operator_namespace}")
    _print_next_steps(kube_console, benchmark_namespace)


@app.default
async def setup(
    *,
    manage_options: KubeManageOptions | None = None,
    jobset_version: Annotated[
        str | None,
        Parameter(
            name="--jobset-version",
            help="JobSet release tag to install (default: latest release, falling back to a known-good pin).",
        ),
    ] = None,
    operator_namespace: Annotated[
        str | None,
        Parameter(
            name="--operator-namespace",
            help=f"Namespace for the AIPerf operator (default: {DEFAULT_OPERATOR_NAMESPACE}).",
        ),
    ] = None,
    chart: Annotated[
        Path | None,
        Parameter(
            name="--chart",
            help="Path to an alternate aiperf-operator Helm chart (default: chart bundled with AIPerf).",
        ),
    ] = None,
    skip_operator: Annotated[
        bool,
        Parameter(
            name="--skip-operator",
            help="Install only the JobSet CRD and namespaces; leave the operator alone.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Report what would be installed and exit without changing the cluster.",
        ),
    ] = False,
) -> None:
    """Install AIPerf's cluster prerequisites.

    Idempotent: each step checks first and skips what is already present, so
    re-running against a prepared cluster is safe and reports "already
    installed" rather than failing.

    Steps:
      1. JobSet CRD -- AIPerf runs every benchmark as a JobSet.
      2. Namespaces -- one for the operator, one for benchmarks.
      3. AIPerf operator -- the Helm chart that owns the AIPerfJob and
         AIPerfSweep CRDs and reconciles them.

    Examples:
        # Prepare a cluster
        aiperf kube setup

        # See what is missing without touching anything
        aiperf kube setup --dry-run

        # Pin JobSet and install only prerequisites
        aiperf kube setup --jobset-version v0.5.2 --skip-operator
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Preparing Cluster"):
        from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

        operator_ns = operator_namespace or DEFAULT_OPERATOR_NAMESPACE
        benchmark_ns = manage_options.namespace or DEFAULT_BENCHMARK_NAMESPACE
        kubectl = _kubectl_base(manage_options)
        await _ensure_jobset(
            manage_options,
            kubectl,
            jobset_version=jobset_version,
            dry_run=dry_run,
        )
        await _ensure_namespaces(
            manage_options,
            kubectl,
            (operator_ns, benchmark_ns),
            dry_run=dry_run,
        )
        await _install_operator(
            manage_options,
            operator_ns,
            benchmark_ns,
            chart=chart,
            skip_operator=skip_operator,
            dry_run=dry_run,
        )


def _print_next_steps(kube_console, benchmark_ns: str) -> None:
    """Tell the user what to run once the cluster is prepared."""
    kube_console.print_info("Cluster ready. Next:")
    kube_console.print_info("  aiperf kube preflight     # verify the cluster")
    kube_console.print_info(
        f"  aiperf kube profile -f benchmark.yaml --namespace {benchmark_ns}"
    )
