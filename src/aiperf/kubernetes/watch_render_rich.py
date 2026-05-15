# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rich TUI renderer for aiperf kube watch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from aiperf.kubernetes.watch_models import WatchSnapshot

_HEALTH_COLORS = {
    "healthy": "green",
    "degraded": "yellow",
    "stalled": "yellow",
    "failing": "red",
    "completed": "cyan",
    "failed": "red",
}

_PHASE_COLORS = {
    "Pending": "yellow",
    "Running": "green",
    "Completed": "cyan",
    "Failed": "red",
    "Cancelled": "yellow",
}


class RichRenderer:
    """Renders WatchSnapshot as a Rich terminal dashboard."""

    def __init__(self) -> None:
        self._console = Console(stderr=True)
        self._started = False

    def start(self) -> None:
        """Clear screen on first render."""
        self._started = True

    def stop(self) -> None:
        """No-op."""

    def render(self, snapshot: WatchSnapshot) -> None:
        """Render a full dashboard frame."""
        if self._started:
            self._console.clear()

        sections = [
            self._render_header(snapshot),
            self._render_progress(snapshot),
            self._render_metrics(snapshot),
            self._render_pods(snapshot),
        ]

        if snapshot.diagnosis.issues:
            sections.append(self._render_problems(snapshot))

        if snapshot.events:
            sections.append(self._render_events(snapshot))

        if snapshot.phase in ("Completed", "Failed") and snapshot.results:
            sections.append(self._render_completion(snapshot))

        health = snapshot.diagnosis.health
        color = _HEALTH_COLORS.get(health, "white")
        phase_color = _PHASE_COLORS.get(snapshot.phase, "white")

        elapsed = _format_duration(snapshot.elapsed_seconds)
        title = (
            f"AIPerf Watch: [bold]{snapshot.job_id}[/] "
            f"([dim]{snapshot.namespace}[/]) "
            f"[{phase_color}]{snapshot.phase}[/] "
            f"[dim]{elapsed}[/]"
        )
        subtitle = f"[{color}]Health: {health}[/]"

        panel = Panel(
            Group(*sections),
            title=title,
            subtitle=subtitle,
            border_style=color,
        )
        self._console.print(panel)

    def _render_header(self, snap: WatchSnapshot) -> Text:
        parts = []
        if snap.model:
            parts.append(f"Model: {snap.model}")
        if snap.endpoint:
            parts.append(f"Endpoint: {snap.endpoint}")
        if snap.current_phase:
            parts.append(f"Phase: {snap.current_phase}")
        return Text("  |  ".join(parts), style="dim") if parts else Text("")

    def _render_progress(self, snap: WatchSnapshot) -> Panel:
        if snap.progress and snap.progress.percent > 0:
            bar = ProgressBar(total=100, completed=snap.progress.percent, width=40)
            pct = f" {snap.progress.percent:.1f}%"
            eta = ""
            if snap.progress.eta_seconds is not None:
                eta = f"  ETA: {_format_duration(snap.progress.eta_seconds)}"
            reqs = ""
            if snap.progress.requests_completed:
                reqs = f"  ({snap.progress.requests_completed:,} / {snap.progress.requests_total:,})"
            content = Group(bar, Text(f"{pct}{reqs}{eta}", style="dim"))
        else:
            content = Text("Waiting for progress data...", style="dim italic")
        return Panel(content, title="Progress", border_style="dim")

    def _render_metrics(self, snap: WatchSnapshot) -> Panel:
        if not snap.metrics:
            return Panel(
                Text("No metrics available yet", style="dim italic"),
                title="Metrics",
                border_style="dim",
            )
        m = snap.metrics
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_column(style="bold")
        table.add_column(justify="right")

        table.add_row(
            "Throughput",
            f"{m.request_throughput_rps:,.1f} req/s",
            "Goodput",
            f"{m.goodput_rps:,.1f} req/s",
        )
        table.add_row(
            "Output Tokens",
            f"{m.output_token_throughput_tps:,.1f} tok/s",
            "Total Tokens",
            f"{m.total_token_throughput_tps:,.1f} tok/s",
        )

        table.add_row(
            "Latency avg",
            f"{m.request_latency_avg_ms:.2f} ms",
            "Latency p99",
            f"{m.request_latency_p99_ms:.2f} ms",
        )

        # TTFT (streaming)
        streaming_tag = " [cyan](streaming)[/]" if m.streaming else ""
        table.add_row(
            "TTFT avg",
            f"{m.ttft_avg_ms:.2f} ms",
            "TTFT p99",
            f"{m.ttft_p99_ms:.2f} ms",
        )

        # ITL + per-user throughput
        if m.inter_token_latency_avg_ms > 0 or m.time_to_second_token_avg_ms > 0:
            table.add_row(
                "ITL avg",
                f"{m.inter_token_latency_avg_ms:.4f} ms",
                "TTST avg",
                f"{m.time_to_second_token_avg_ms:.2f} ms",
            )

        if m.prefill_throughput_per_user_tps > 0:
            table.add_row(
                "Prefill/user",
                f"{m.prefill_throughput_per_user_tps:,.0f} tok/s",
                "Decode/user",
                f"{m.output_token_throughput_per_user_tps:,.0f} tok/s",
            )

        table.add_row(
            "Requests",
            f"{m.request_count:,}",
            "Errors",
            f"{m.error_count:,}",
        )

        title = f"Live Metrics{streaming_tag}"
        return Panel(table, title=title, border_style="blue")

    def _render_pods(self, snap: WatchSnapshot) -> Panel:
        if not snap.pods:
            workers = snap.workers
            return Panel(
                Text(f"Workers: {workers.ready}/{workers.total}", style="dim"),
                title="Pods",
                border_style="dim",
            )
        table = Table(box=None, padding=(0, 1))
        table.add_column("Pod", style="bold")
        table.add_column("Status")
        table.add_column("Restarts", justify="right")
        table.add_column("Issues")
        for pod in snap.pods:
            status_style = "green" if pod.ready else "yellow"
            issues = []
            if pod.oom_killed:
                issues.append("[red]OOMKilled[/]")
            if pod.restarts > 3:
                issues.append("[red]CrashLoop?[/]")
            table.add_row(
                _short_pod_name(pod.name),
                Text(pod.status, style=status_style),
                str(pod.restarts),
                " ".join(issues) if issues else "-",
            )
        return Panel(table, title="Pods", border_style="dim")

    def _render_problems(self, snap: WatchSnapshot) -> Panel:
        lines = []
        for issue in snap.diagnosis.issues:
            icon = "[red]x[/]" if issue.severity == "critical" else "[yellow]![/]"
            lines.append(Text.from_markup(f"  {icon} {issue.title}: {issue.detail}"))
            lines.append(Text(f"    -> {issue.suggested_fix}", style="dim"))
        return Panel(
            Group(*lines) if lines else Text("None"),
            title="Problems",
            border_style="red"
            if any(i.severity == "critical" for i in snap.diagnosis.issues)
            else "yellow",
        )

    def _render_events(self, snap: WatchSnapshot) -> Panel:
        table = Table(box=None, padding=(0, 1), show_header=False)
        table.add_column("Type", width=8)
        table.add_column("Reason", width=20)
        table.add_column("Message")
        for evt in snap.events[-8:]:
            style = "yellow" if evt.type == "Warning" else "dim"
            table.add_row(
                Text(evt.type, style=style),
                evt.reason,
                evt.message[:80],
            )
        return Panel(
            table, title=f"Events (last {min(len(snap.events), 8)})", border_style="dim"
        )

    def _render_completion(self, snap: WatchSnapshot) -> Panel:
        if snap.phase == "Failed":
            return Panel(
                Text(snap.error or "Unknown error", style="red bold"),
                title="Failed",
                border_style="red",
            )
        results = snap.results or {}
        lines = []
        for key in [
            "request_throughput",
            "request_latency",
            "time_to_first_token",
            "output_token_throughput",
        ]:
            if key in results:
                val = results[key]
                avg = val.get("avg", 0)
                unit = val.get("unit", "")
                lines.append(f"  {key}: {avg:,.2f} {unit}")
        return Panel(
            Text("\n".join(lines) if lines else "Results available"),
            title="Completed",
            border_style="cyan",
        )


def _format_duration(seconds: float) -> str:
    from aiperf.ui.utils import format_elapsed_time

    return format_elapsed_time(seconds)


def _short_pod_name(name: str) -> str:
    parts = name.split("-")
    if len(parts) > 3:
        return "-".join(parts[-4:])
    return name
