# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot diagnostic analysis of Kubernetes benchmark deployments."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from cyclopts import App, Parameter

from aiperf.cli_commands.kube._debug_extract import (
    _extract_pod_info,
    _get_serializer,
)
from aiperf.cli_commands.kube._debug_report import (
    _get_event_severity_style,
    _print_report,
)
from aiperf.config.kube import KubeManageOptions

# Re-exports for tests/importers that reference these by their historical
# ``aiperf.cli_commands.kube.debug`` paths.
__all__ = [
    "_extract_pod_info",
    "_get_event_severity_style",
    "_get_namespace_events",
    "_get_node_resources",
    "_get_problem_pod_logs",
    "_print_report",
    "app",
    "debug",
]

app = App(name="debug")
logger = logging.getLogger("aiperf.kube")


async def _get_namespace_events(
    api: Any,
    namespace: str,
) -> list[dict[str, Any]]:
    """Fetch recent events from a namespace.

    Args:
        api: kubernetes_asyncio ``ApiClient``.
        namespace: Namespace to query.

    Returns:
        List of event dicts sorted by last timestamp (newest first).
    """
    from kubernetes_asyncio import client as k8s_client_mod

    try:
        core = k8s_client_mod.CoreV1Api(api)
        event_list = await core.list_namespaced_event(namespace)
    except Exception:  # noqa: BLE001 - diagnostics best-effort; return [] on any API/serializer error
        return []

    serializer = _get_serializer(api)
    result = []
    for event in event_list.items:
        try:
            raw = serializer.sanitize_for_serialization(event) or {}
        except Exception:  # noqa: BLE001 - diagnostics best-effort; skip malformed events
            logger.debug(
                "Skipping malformed Kubernetes event during debug aggregation",
                exc_info=True,
            )
            continue
        involved = raw.get("involvedObject", {})
        result.append(
            {
                "type": raw.get("type", "Normal"),
                "reason": raw.get("reason", ""),
                "message": raw.get("message", ""),
                "object": f"{involved.get('kind', '')}/{involved.get('name', '')}",
                "count": raw.get("count", 1),
                "last_seen": raw.get("lastTimestamp", raw.get("eventTime", "")),
            }
        )

    result.sort(key=lambda e: e["last_seen"] or "", reverse=True)
    return result


async def _get_node_resources(api: Any) -> list[dict[str, Any]]:
    """Fetch node resource info.

    Args:
        api: kubernetes_asyncio ``ApiClient``.

    Returns:
        List of node resource dicts with capacity and conditions.
    """
    from kubernetes_asyncio import client as k8s_client_mod

    try:
        core = k8s_client_mod.CoreV1Api(api)
        node_list = await core.list_node()
    except Exception:  # noqa: BLE001 - diagnostics best-effort; return [] on any API/serializer error
        return []

    serializer = _get_serializer(api)
    result = []
    for node in node_list.items:
        try:
            raw = serializer.sanitize_for_serialization(node) or {}
        except Exception:  # noqa: BLE001 - diagnostics best-effort; skip malformed nodes
            logger.debug(
                "Skipping malformed Kubernetes node during debug aggregation",
                exc_info=True,
            )
            continue
        status = raw.get("status", {})
        capacity = status.get("capacity", {})
        allocatable = status.get("allocatable", {})
        conditions = status.get("conditions", [])

        pressure_conditions = []
        for cond in conditions:
            if cond.get("status") == "True" and cond.get("type") in (
                "MemoryPressure",
                "DiskPressure",
                "PIDPressure",
            ):
                pressure_conditions.append(cond["type"])

        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True" for c in conditions
        )

        result.append(
            {
                "name": raw.get("metadata", {}).get("name", ""),
                "ready": ready,
                "cpu_capacity": capacity.get("cpu", "0"),
                "memory_capacity": capacity.get("memory", "0"),
                "gpu_capacity": capacity.get("nvidia.com/gpu", "0"),
                "cpu_allocatable": allocatable.get("cpu", "0"),
                "memory_allocatable": allocatable.get("memory", "0"),
                "gpu_allocatable": allocatable.get("nvidia.com/gpu", "0"),
                "pressure": pressure_conditions,
            }
        )

    return result


async def _get_problem_pod_logs(
    api: Any,
    pod_infos: list[dict[str, Any]],
    tail_lines: int = 20,
) -> dict[str, dict[str, str]]:
    """Fetch recent logs from pods with problems.

    Args:
        api: kubernetes_asyncio ``ApiClient``.
        pod_infos: List of extracted pod info dicts (from ``_extract_pod_info``).
        tail_lines: Number of log lines to fetch per container.

    Returns:
        Dict mapping pod_name -> {container_name: log_text}.
    """
    from kubernetes_asyncio import client as k8s_client_mod
    from kubernetes_asyncio.client.exceptions import ApiException

    core = k8s_client_mod.CoreV1Api(api)
    result: dict[str, dict[str, str]] = {}

    problem_pods = [info for info in pod_infos if info["problems"]]

    for info in problem_pods:
        pod_name = info["name"]
        namespace = info.get("namespace") or ""
        if not namespace:
            continue

        container_logs: dict[str, str] = {}
        for cs in info["container_statuses"]:
            container_name = cs.get("name", "unknown")
            try:
                log_text = await core.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    container=container_name,
                    tail_lines=tail_lines,
                )
                if log_text:
                    container_logs[container_name] = log_text.rstrip("\n")
            except ApiException:
                container_logs[container_name] = "<logs unavailable>"
            except Exception:  # noqa: BLE001 - diagnostic log fetch best-effort; any k8s client error is recorded as placeholder
                container_logs[container_name] = "<error fetching logs>"

        if container_logs:
            result[pod_name] = container_logs

    return result


async def _resolve_target_namespaces(
    api: Any,
    *,
    namespace: str | None,
    job_id: str | None,
    all_namespaces: bool,
) -> list[str] | None:
    """Resolve the list of namespaces to inspect for `debug`.

    Returns None when the user-facing error/warning has already been printed
    and the caller should exit silently.
    """
    from aiperf.kubernetes import client as kube_client_mod
    from aiperf.kubernetes import console as kube_console

    if all_namespaces:
        jobsets = await kube_client_mod.list_jobsets(api, all_namespaces=True)
        target = list({js.namespace for js in jobsets})
        if not target:
            kube_console.print_warning("No AIPerf deployments found in any namespace")
            return None
        return target

    if job_id:
        job_info = await kube_client_mod.find_aiperf_job(api, job_id, namespace)
        if job_info:
            return [job_info.namespace]
        sweep_info = await kube_client_mod.find_aiperf_sweep(api, job_id, namespace)
        if sweep_info:
            return [sweep_info.namespace]
        jobset_info = await kube_client_mod.find_jobset(api, job_id, namespace)
        if jobset_info:
            return [jobset_info.namespace]
        kube_console.print_error(f"No AIPerf job found with ID: {job_id}")
        return None

    if namespace:
        return [namespace]

    from aiperf.kubernetes.cli_helpers import resolve_job_id_and_namespace

    resolved = resolve_job_id_and_namespace(None, None)
    if not resolved:
        return None
    _, ns = resolved
    return [ns or "default"]


@app.default
async def debug(
    *,
    manage_options: KubeManageOptions | None = None,
    job_id: Annotated[
        str | None,
        Parameter(
            name=["-j", "--job-id"],
            help="Specific AIPerf job ID or AIPerfSweep name to diagnose.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Parameter(
            name=["-v", "--verbose"], help="Show detailed output including pod logs."
        ),
    ] = False,
    all_namespaces: Annotated[
        bool,
        Parameter(
            name=["-A", "--all-namespaces"],
            help="Inspect all namespaces with AIPerf deployments.",
        ),
    ] = False,
    variation: Annotated[
        int | None,
        Parameter(
            name=["--variation"],
            help="When --job-id is an AIPerfSweep name, target child variation index (00..99). Resolves to <sweep>-v<idx:02d>[-t<trial>]. Note: -v is reserved for --verbose; use the long form here.",
        ),
    ] = None,
    trial: Annotated[
        int | None,
        Parameter(
            name=["-t", "--trial"],
            help="Trial index (0..9) within a sweep variation. Requires --variation.",
        ),
    ] = None,
) -> None:
    """Run diagnostic analysis on a benchmark deployment.

    Inspects pod states, events, node resources, and container logs to
    identify problems. Outputs a structured report with suggestions.

    Examples:
        aiperf kube debug -n my-benchmark
        aiperf kube debug --job-id abc123 -v
        aiperf kube debug -A
        aiperf kube debug --job-id my-sweep --variation 7
        aiperf kube debug --job-id my-sweep --variation 5 -t 0
    """
    from aiperf import cli_utils
    from aiperf.cli_commands.kube._kube_common import resolve_child_name

    manage_options = manage_options or KubeManageOptions()

    if job_id is not None:
        child = resolve_child_name(job_id, variation=variation, trial=trial)
        if child is not None:
            job_id = child

    with cli_utils.exit_on_error(title="Error Running Diagnostics"):
        from aiperf.kubernetes import client as kube_client_mod

        async with kube_client_mod.k8s_client(
            kubeconfig=manage_options.kubeconfig,
            context=manage_options.kube_context,
        ) as api:
            target_namespaces = await _resolve_target_namespaces(
                api,
                namespace=manage_options.namespace,
                job_id=job_id,
                all_namespaces=all_namespaces,
            )
            if target_namespaces is None:
                return

            node_resources = await _get_node_resources(api)

            for ns in sorted(target_namespaces):
                await _debug_namespace(
                    api,
                    ns=ns,
                    job_id=job_id,
                    verbose=verbose,
                    node_resources=node_resources,
                )


async def _debug_namespace(
    api: Any,
    *,
    ns: str,
    job_id: str | None,
    verbose: bool,
    node_resources: Any,
) -> None:
    """Collect and print the diagnostic report for a single namespace."""
    from aiperf.kubernetes import client as kube_client_mod
    from aiperf.kubernetes.constants import AIPerfLabels

    label_selector = (
        kube_client_mod.job_selector(job_id) if job_id else AIPerfLabels.SELECTOR
    )

    pods = await kube_client_mod.get_pods(api, ns, label_selector)
    pod_infos = [_extract_pod_info(pod, api) for pod in pods]
    events = await _get_namespace_events(api, ns)

    pod_logs: dict[str, dict[str, str]] = {}
    if verbose:
        pod_logs = await _get_problem_pod_logs(api, pod_infos)

    _print_report(
        ns,
        pod_infos=pod_infos,
        events=events,
        node_resources=node_resources,
        pod_logs=pod_logs,
        verbose=verbose,
    )
