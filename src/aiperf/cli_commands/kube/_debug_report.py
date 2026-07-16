# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report-printing helpers for ``aiperf kube debug``.

Separated from ``debug.py`` to split the large ``_print_report`` function
into focused per-section helpers.
"""

from __future__ import annotations

from typing import Any


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


def _print_pods_table(pod_infos: list[dict[str, Any]]) -> None:
    """Print the pod overview table, or a warning if empty."""
    from rich.table import Table
    from rich.text import Text

    from aiperf.kubernetes.console import console, print_warning

    if not pod_infos:
        print_warning("No pods found")
        return

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


def _print_problems(pod_infos: list[dict[str, Any]]) -> None:
    """Print the aggregated problems section."""
    from aiperf.kubernetes.console import (
        print_error,
        print_header,
        print_info,
        print_success,
        print_warning,
    )

    all_problems = [
        (info["name"], problem) for info in pod_infos for problem in info["problems"]
    ]

    if not all_problems:
        print_header("Problems", style="bold green")
        print_success("No problems detected")
        return

    print_header("Problems Found", style="bold red")
    for pod_name, problem in all_problems:
        severity = problem["severity"]
        header = f"[{pod_name}] {problem['state']} (container: {problem['container']})"
        if severity == "CRITICAL":
            print_error(header)
        else:
            print_warning(header)
        print_info(f"  Suggestion: {problem['suggestion']}")
        if problem["message"]:
            print_info(f"  Detail: {problem['message']}")


def _print_events(events: list[dict[str, Any]], *, verbose: bool) -> None:
    """Print the events section (Recent in verbose mode, Warnings otherwise)."""
    from rich.table import Table
    from rich.text import Text

    from aiperf.kubernetes.console import console, print_header, print_info

    warning_events = [e for e in events if e["type"] == "Warning"]
    display_events = events[:30] if verbose else warning_events[:15]

    if not display_events:
        if not verbose:
            print_info("No warning events found")
        return

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


def _print_node_resources(node_resources: list[dict[str, Any]]) -> None:
    """Print the node-resources section, if any nodes were collected."""
    from rich.table import Table
    from rich.text import Text

    from aiperf.kubernetes.console import console, print_header

    if not node_resources:
        return

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


def _print_pod_logs(pod_logs: dict[str, dict[str, str]]) -> None:
    """Print per-container logs for problem pods (verbose mode only)."""
    from aiperf.kubernetes.console import console, print_header, print_info

    if not pod_logs:
        return

    print_header("Problem Pod Logs", style="bold yellow")
    for pod_name, containers in pod_logs.items():
        for container_name, log_text in containers.items():
            print_info(f"--- {pod_name}/{container_name} ---")
            console.print(f"[dim]{log_text}[/dim]")


def _print_summary(
    pod_infos: list[dict[str, Any]],
    events: list[dict[str, Any]],
    node_resources: list[dict[str, Any]],
) -> None:
    """Print the final summary footer."""
    from aiperf.kubernetes.console import print_header, print_info, print_warning

    total_pods = len(pod_infos)
    problem_pods = sum(1 for info in pod_infos if info["problems"])
    running_pods = sum(1 for info in pod_infos if info["phase"] == "Running")
    warning_events = [e for e in events if e["type"] == "Warning"]

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
    from aiperf.kubernetes.console import print_header

    print_header(f"Diagnostic Report: {namespace}", style="bold cyan")

    _print_pods_table(pod_infos)
    _print_problems(pod_infos)
    _print_events(events, verbose=verbose)
    _print_node_resources(node_resources)
    if verbose:
        _print_pod_logs(pod_logs)
    _print_summary(pod_infos, events, node_resources)
