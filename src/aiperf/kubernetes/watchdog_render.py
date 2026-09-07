# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text dashboards and final reports for the benchmark watchdog.

Pure rendering: takes the watchdog's accumulated state and produces the
multi-line strings logged to stdout. Kept separate from the monitoring
loop so ``BenchmarkWatchdog`` stays focused on the control flow.
"""

from __future__ import annotations

import time

from aiperf.kubernetes.watchdog_models import (
    PodTimeline,
    ProblemSeverity,
    WatchdogPodSnapshot,
    WatchdogProblem,
)

# Box drawing for dashboards
_W = 72
_DLINE = "+" + "=" * _W + "+"


def _row(text: str) -> str:
    """Format a line inside the box, padded to fixed width."""
    return f"| {text:<{_W - 2}} |"


def _progress_bar(pct: float, width: int = 30) -> str:
    """Render a text progress bar."""
    filled = int(width * min(pct, 100) / 100)
    empty = width - filled
    bar = "#" * filled + "-" * empty
    return f"[{bar}] {pct:.0f}%"


def _short_pod_name(name: str, max_len: int = 38) -> str:
    """Shorten pod name, keeping the unique suffix visible."""
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3) :]


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m{secs:02d}s"


def _phase_icon(phase: str) -> str:
    """Map phase to a compact visual indicator."""
    return {
        "Pending": "...",
        "Running": ">>>",
        "Succeeded": "[OK]",
        "Failed": "[!!]",
        "Unknown": "[??]",
        "Completed": "[OK]",
    }.get(phase, "   ")


def render_status_dashboard(
    *,
    namespace: str,
    start_time: float,
    timeout: float | None,
    node_cpu_pct: int | None,
    node_mem_pct: int | None,
    stale_ns_count: int,
    pods: list[WatchdogPodSnapshot],
    pod_timelines: dict[str, PodTimeline],
    problems: list[WatchdogProblem],
) -> str:
    """Render the periodic status dashboard text block."""
    elapsed = time.time() - start_time
    lines = [_DLINE, _row(f"WATCHDOG  |  ns={namespace}")]

    if timeout is not None:
        remaining = max(0, timeout - elapsed)
        pct = min(100, (elapsed / timeout) * 100) if timeout > 0 else 0
        lines.append(
            _row(
                f"time: {_fmt_duration(elapsed)} elapsed, "
                f"{_fmt_duration(remaining)} remaining  "
                f"{_progress_bar(pct, 20)}"
            )
        )
    else:
        lines.append(_row(f"time: {_fmt_duration(elapsed)} elapsed"))

    if node_cpu_pct is not None:
        node_line = f"node: cpu={node_cpu_pct}%"
        if node_mem_pct is not None:
            node_line += f"  mem={node_mem_pct}%"
        if stale_ns_count > 0:
            node_line += f"  stale_ns={stale_ns_count}"
        lines.append(_row(node_line))

    lines.append(_row("-" * (_W - 2)))
    lines.append(
        _row(f"{'':>4} {'POD':<36} {'PHASE':<12} {'RDY':<5} {'RST':>3} {'AGE':>6}")
    )
    lines.append(_row("-" * (_W - 2)))

    for pod in pods:
        tl = pod_timelines.get(pod.name)
        age = (time.time() - tl.first_seen) if tl else 0
        icon = _phase_icon(pod.phase)
        short = _short_pod_name(pod.name, 36)
        ready_str = "Y" if pod.ready else "N"
        lines.append(
            _row(
                f"{icon} {short:<36} {pod.phase:<12} {ready_str:<5} "
                f"{pod.restarts:>3} {_fmt_duration(age):>6}"
            )
        )

    crits = sum(1 for p in problems if p.severity == ProblemSeverity.CRITICAL)
    warns = sum(1 for p in problems if p.severity == ProblemSeverity.WARNING)
    if crits or warns:
        lines.append(_row("-" * (_W - 2)))
        parts = []
        if crits:
            parts.append(f"{crits} CRITICAL")
        if warns:
            parts.append(f"{warns} WARNING")
        lines.append(_row(f"issues: {', '.join(parts)}"))

    lines.append(_DLINE)
    return "[WATCHDOG]\n" + "\n".join(lines)


def render_final_report(
    *,
    namespace: str,
    start_time: float,
    timeout: float | None,
    node_cpu_pct: int | None,
    node_mem_pct: int | None,
    pod_timelines: dict[str, PodTimeline],
    problems: list[WatchdogProblem],
) -> str:
    """Render the comprehensive end-of-run report text block."""
    elapsed = time.time() - start_time
    total_pods = len(pod_timelines)
    succeeded = sum(
        1
        for tl in pod_timelines.values()
        if tl.last_phase in ("Succeeded", "Completed")
    )
    failed = sum(1 for tl in pod_timelines.values() if tl.last_phase == "Failed")
    total_restarts = sum(tl.restart_count for tl in pod_timelines.values())

    lines = [
        "",
        _DLINE,
        _row("WATCHDOG FINAL REPORT"),
        _row(f"Namespace:  {namespace}"),
        _row(
            f"Duration:   {_fmt_duration(elapsed)}"
            + (
                f" (timeout was {_fmt_duration(timeout)})"
                if timeout is not None
                else ""
            )
        ),
        _row(
            f"Pods:       {total_pods} tracked, "
            f"{succeeded} succeeded, {failed} failed, "
            f"{total_restarts} total restarts"
        ),
    ]

    if node_cpu_pct is not None:
        lines.append(
            _row(f"Node:       CPU {node_cpu_pct}%, Memory {node_mem_pct or '?'}%")
        )

    if pod_timelines:
        lines.append(_row("-" * (_W - 2)))
        lines.append(_row("POD LIFECYCLE:"))
        for tl in pod_timelines.values():
            short = tl.name.split("-")[-1] if "-" in tl.name else tl.name
            phase_times = []
            for i, (ts, phase) in enumerate(tl.phase_history):
                if i + 1 < len(tl.phase_history):
                    dt = tl.phase_history[i + 1][0] - ts
                    phase_times.append(f"{phase}({_fmt_duration(dt)})")
                else:
                    phase_times.append(phase)
            timing = " -> ".join(phase_times)
            rst_str = f" [{tl.restart_count}rst]" if tl.restart_count else ""
            lines.append(_row(f"  {tl.role[:4]}({short}): {timing}{rst_str}"))

    crits = [p for p in problems if p.severity == ProblemSeverity.CRITICAL]
    warns = [p for p in problems if p.severity == ProblemSeverity.WARNING]

    lines.append(_row("-" * (_W - 2)))
    if crits or warns:
        lines.append(_row(f"ISSUES: {len(crits)} critical, {len(warns)} warnings"))
        for p in crits:
            msg = p.message[: (_W - 14)]
            lines.append(_row(f"  [CRIT] {msg}"))
        for p in warns:
            msg = p.message[: (_W - 14)]
            lines.append(_row(f"  [WARN] {msg}"))
    else:
        lines.append(_row("STATUS: Clean run - no problems detected"))

    lines.append(_DLINE)
    return "[WATCHDOG]" + "\n".join(lines)
