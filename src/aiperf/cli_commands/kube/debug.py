# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot diagnostic analysis of Kubernetes benchmark deployments."""

from __future__ import annotations

from typing import Annotated, Any

from cyclopts import App, Parameter

app = App(name="debug")

# Lazy module-scoped ApiClient used only for sanitize_for_serialization.
# Reused across calls to avoid leaking an aiohttp session per invocation;
# sanitize_for_serialization does not require the client to be started.
_serializer: Any = None


def _get_serializer() -> Any:
    """Return a cached ``ApiClient`` for ``sanitize_for_serialization``.

    The client is allocated on first use and never explicitly closed --
    it is only used for synchronous object-to-dict serialization, so no
    aiohttp session is opened beyond the default one aiohttp lazily
    creates (which is cleaned up at process exit).
    """
    global _serializer
    if _serializer is None:
        from kubernetes_asyncio.client import ApiClient

        _serializer = ApiClient()
    return _serializer


# Container states that indicate problems
_PROBLEM_STATES: dict[str, tuple[str, str]] = {
    "CrashLoopBackOff": (
        "CRITICAL",
        "Container is crash-looping. Check logs for the root cause.",
    ),
    "ImagePullBackOff": (
        "CRITICAL",
        "Image cannot be pulled. Verify the image name and registry access.",
    ),
    "ErrImagePull": (
        "CRITICAL",
        "Image pull failed. Check image name, tag, and pull secrets.",
    ),
    "OOMKilled": (
        "CRITICAL",
        "Container was killed due to out-of-memory. Increase memory limits.",
    ),
    "CreateContainerConfigError": (
        "ERROR",
        "Container config error. Check ConfigMaps, Secrets, and volume mounts.",
    ),
    "RunContainerError": (
        "ERROR",
        "Failed to run container. Check security context and resource limits.",
    ),
}


def _pod_to_raw(pod: Any) -> tuple[str, dict[str, Any]]:
    """Normalize V1Pod or legacy ``.raw``-style mock to (name, raw_dict)."""
    raw = getattr(pod, "raw", None)
    if raw is not None:
        name = getattr(pod, "name", None) or raw.get("metadata", {}).get("name", "")
        return (name, raw)
    # Real V1Pod from kubernetes_asyncio -- serialize to dict shape.
    # Use a module-scoped serializer to avoid leaking an aiohttp session per
    # pod; sanitize_for_serialization only reads class-level attrs so the
    # singleton is safe to reuse and never needs closing.
    raw = _get_serializer().sanitize_for_serialization(pod) or {}
    name = raw.get("metadata", {}).get("name", "")
    return (name, raw)


def _extract_pod_info(pod: Any) -> dict[str, Any]:
    """Extract diagnostic info from a Pod object.

    Args:
        pod: V1Pod (kubernetes_asyncio) or any object exposing ``.name`` and
            a ``.raw`` dict (legacy test mocks).

    Returns:
        Dict with pod name, phase, conditions, container statuses, and problems.
    """
    pod_name, raw = _pod_to_raw(pod)
    status = raw.get("status", {})
    phase = status.get("phase", "Unknown")
    container_statuses = status.get("containerStatuses", [])
    init_container_statuses = status.get("initContainerStatuses", [])
    conditions = status.get("conditions", [])

    problems: list[dict[str, str]] = []
    restarts = 0

    all_statuses = init_container_statuses + container_statuses
    for cs in all_statuses:
        restarts += cs.get("restartCount", 0)
        container_name = cs.get("name", "unknown")

        waiting = cs.get("state", {}).get("waiting", {})
        if waiting:
            reason = waiting.get("reason", "")
            if reason in _PROBLEM_STATES:
                severity, suggestion = _PROBLEM_STATES[reason]
                problems.append(
                    {
                        "container": container_name,
                        "state": reason,
                        "severity": severity,
                        "suggestion": suggestion,
                        "message": waiting.get("message", ""),
                    }
                )
            elif reason and phase == "Pending":
                problems.append(
                    {
                        "container": container_name,
                        "state": reason,
                        "severity": "WARNING",
                        "suggestion": f"Container is waiting: {reason}",
                        "message": waiting.get("message", ""),
                    }
                )

        terminated = cs.get("state", {}).get("terminated", {})
        if terminated:
            reason = terminated.get("reason", "")
            if reason == "OOMKilled":
                severity, suggestion = _PROBLEM_STATES["OOMKilled"]
                problems.append(
                    {
                        "container": container_name,
                        "state": reason,
                        "severity": severity,
                        "suggestion": suggestion,
                        "message": terminated.get("message", ""),
                    }
                )

        last_terminated = cs.get("lastState", {}).get("terminated", {})
        if last_terminated:
            reason = last_terminated.get("reason", "")
            if reason == "OOMKilled":
                severity, suggestion = _PROBLEM_STATES["OOMKilled"]
                problems.append(
                    {
                        "container": container_name,
                        "state": f"{reason} (previous)",
                        "severity": severity,
                        "suggestion": suggestion,
                        "message": last_terminated.get("message", ""),
                    }
                )

    for cond in conditions:
        if (
            cond.get("type") == "PodScheduled"
            and cond.get("status") == "False"
            and cond.get("reason") == "Unschedulable"
        ):
            problems.append(
                {
                    "container": "-",
                    "state": "Unschedulable",
                    "severity": "CRITICAL",
                    "suggestion": (
                        "Pod cannot be scheduled. "
                        "Check node resources, taints/tolerations, and node selectors."
                    ),
                    "message": cond.get("message", ""),
                }
            )

    metadata = raw.get("metadata", {})
    return {
        "name": pod_name,
        "namespace": metadata.get("namespace", ""),
        "phase": phase,
        "restarts": restarts,
        "problems": problems,
        "container_statuses": all_statuses,
        "node": raw.get("spec", {}).get("nodeName", ""),
    }


def _get_event_severity_style(event_type: str) -> str:
    """Return Rich style for event type.

    Args:
        event_type: Kubernetes event type (Normal, Warning).

    Returns:
        Rich style string.
    """
    if event_type == "Warning":
        return "yellow"
    return "dim"


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

    serializer = _get_serializer()
    result = []
    for event in event_list.items:
        raw = serializer.sanitize_for_serialization(event) or {}
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

    serializer = _get_serializer()
    result = []
    for node in node_list.items:
        raw = serializer.sanitize_for_serialization(node) or {}
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


def _print_report(
    namespace: str,
    *,
    pod_infos: list[dict[str, Any]],
    events: list[dict[str, Any]],
    node_resources: list[dict[str, Any]],
    pod_logs: dict[str, dict[str, str]],
    verbose: bool,
) -> None:
    """Print the structured diagnostic report.

    Args:
        namespace: Namespace being analyzed.
        pod_infos: List of extracted pod info dicts.
        events: List of event dicts.
        node_resources: List of node resource dicts.
        pod_logs: Dict of pod logs (pod_name -> container -> text).
        verbose: Whether to show verbose output.
    """
    from rich.table import Table
    from rich.text import Text

    from aiperf.kubernetes.console import (
        console,
        print_error,
        print_header,
        print_info,
        print_success,
        print_warning,
    )

    print_header(f"Diagnostic Report: {namespace}", style="bold cyan")

    if pod_infos:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("POD", style="cyan")
        table.add_column("STATUS")
        table.add_column("RESTARTS", justify="right")
        table.add_column("NODE", style="dim")
        table.add_column("ISSUES", justify="right")

        for info in pod_infos:
            phase = info["phase"]
            if phase in ("Running", "Succeeded"):
                phase_style = "green"
            elif phase in ("Failed", "Unknown"):
                phase_style = "red"
            else:
                phase_style = "yellow"

            restart_style = "red" if info["restarts"] > 0 else "dim"
            issue_count = len(info["problems"])
            issue_style = "red bold" if issue_count > 0 else "dim"

            table.add_row(
                info["name"],
                Text(phase, style=phase_style),
                Text(str(info["restarts"]), style=restart_style),
                info["node"] or "-",
                Text(str(issue_count), style=issue_style),
            )

        console.print(table)
    else:
        print_warning("No pods found")

    all_problems = [
        (info["name"], problem) for info in pod_infos for problem in info["problems"]
    ]

    if all_problems:
        print_header("Problems Found", style="bold red")
        for pod_name, problem in all_problems:
            severity = problem["severity"]
            if severity == "CRITICAL":
                print_error(
                    f"[{pod_name}] {problem['state']} "
                    f"(container: {problem['container']})"
                )
            else:
                print_warning(
                    f"[{pod_name}] {problem['state']} "
                    f"(container: {problem['container']})"
                )
            print_info(f"  Suggestion: {problem['suggestion']}")
            if problem["message"]:
                print_info(f"  Detail: {problem['message']}")
    else:
        print_header("Problems", style="bold green")
        print_success("No problems detected")

    warning_events = [e for e in events if e["type"] == "Warning"]
    display_events = events[:30] if verbose else warning_events[:15]

    if display_events:
        label = "Recent Events" if verbose else "Warning Events"
        print_header(label, style="bold yellow")

        event_table = Table(show_header=True, header_style="bold", box=None)
        event_table.add_column("TYPE")
        event_table.add_column("REASON", style="dim")
        event_table.add_column("OBJECT", style="dim")
        event_table.add_column("MESSAGE", max_width=60)
        event_table.add_column("COUNT", justify="right", style="dim")

        for event in display_events:
            style = _get_event_severity_style(event["type"])
            event_table.add_row(
                Text(event["type"], style=style),
                event["reason"],
                event["object"],
                event["message"][:120],
                str(event["count"]),
            )

        console.print(event_table)
    elif not verbose:
        print_info("No warning events found")

    if node_resources:
        print_header("Node Resources", style="bold cyan")

        node_table = Table(show_header=True, header_style="bold", box=None)
        node_table.add_column("NODE", style="cyan")
        node_table.add_column("READY")
        node_table.add_column("CPU", justify="right")
        node_table.add_column("MEMORY", justify="right")
        node_table.add_column("GPU", justify="right")
        node_table.add_column("PRESSURE", style="dim")

        for node in node_resources:
            ready_text = Text(
                "Yes" if node["ready"] else "No",
                style="green" if node["ready"] else "red",
            )

            gpu_cap = node["gpu_capacity"]
            gpu_alloc = node["gpu_allocatable"]
            gpu_str = f"{gpu_alloc}/{gpu_cap}" if gpu_cap != "0" else "-"

            pressure_str = ", ".join(node["pressure"]) if node["pressure"] else "-"
            pressure_style = "red" if node["pressure"] else "dim"

            node_table.add_row(
                node["name"],
                ready_text,
                f"{node['cpu_allocatable']}/{node['cpu_capacity']}",
                f"{node['memory_allocatable']}/{node['memory_capacity']}",
                gpu_str,
                Text(pressure_str, style=pressure_style),
            )

        console.print(node_table)

    if verbose and pod_logs:
        print_header("Problem Pod Logs", style="bold yellow")
        for pod_name, containers in pod_logs.items():
            for container_name, log_text in containers.items():
                print_info(f"--- {pod_name}/{container_name} ---")
                console.print(f"[dim]{log_text}[/dim]")

    total_pods = len(pod_infos)
    problem_pods = sum(1 for info in pod_infos if info["problems"])
    running_pods = sum(1 for info in pod_infos if info["phase"] == "Running")

    print_header("Summary", style="bold cyan")
    print_info(
        f"Pods: {total_pods} total, {running_pods} running, {problem_pods} with issues"
    )
    if warning_events:
        print_info(f"Warning events: {len(warning_events)}")
    nodes_with_pressure = [n for n in node_resources if n["pressure"]]
    if nodes_with_pressure:
        print_warning(
            f"Nodes under pressure: {', '.join(n['name'] for n in nodes_with_pressure)}"
        )


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
    namespace: Annotated[
        str | None,
        Parameter(name=["-n", "--namespace"], help="Kubernetes namespace to inspect."),
    ] = None,
    job_id: Annotated[
        str | None,
        Parameter(name=["-j", "--job-id"], help="Specific AIPerf job ID to diagnose."),
    ] = None,
    kubeconfig: Annotated[
        str | None, Parameter(name="--kubeconfig", help="Path to kubeconfig file.")
    ] = None,
    context: Annotated[
        str | None, Parameter(name="--kube-context", help="Kubernetes context to use.")
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
) -> None:
    """Run diagnostic analysis on a benchmark deployment.

    Inspects pod states, events, node resources, and container logs to
    identify problems. Outputs a structured report with suggestions.

    Examples:
        aiperf kube debug -n my-benchmark
        aiperf kube debug --job-id abc123 -v
        aiperf kube debug -A
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Running Diagnostics"):
        from aiperf.kubernetes import client as kube_client_mod

        async with kube_client_mod.k8s_client(
            kubeconfig=kubeconfig,
            context=context,
        ) as api:
            target_namespaces = await _resolve_target_namespaces(
                api,
                namespace=namespace,
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
    pod_infos = [_extract_pod_info(pod) for pod in pods]
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
