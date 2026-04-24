# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube watch command: unified real-time benchmark monitoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

if TYPE_CHECKING:
    from aiperf.kubernetes.watch_orchestrator import WatchRenderer

app = App(name="watch")


@app.default
async def watch(
    job_id: Annotated[
        str | None,
        Parameter(help="Job to watch (default: last deployed / auto-detect)."),
    ] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    all_jobs: Annotated[
        bool,
        Parameter(
            name=["-A", "--all"],
            help="Watch all running jobs.",
        ),
    ] = False,
    output: Annotated[
        str,
        Parameter(
            name=["-o", "--output"],
            help="Output format: rich (TUI), text (plain log lines), or json (NDJSON).",
        ),
    ] = "rich",
    interval: Annotated[
        float,
        Parameter(
            name=["-i", "--interval"],
            help="Refresh interval in seconds.",
        ),
    ] = 2.0,
    follow_logs: Annotated[
        bool,
        Parameter(
            name=["-f", "--follow-logs"],
            help="Include live log tail in output.",
        ),
    ] = False,
) -> None:
    """Watch a running benchmark with live status, metrics, and diagnostics.

    Provides a unified real-time view combining status, metrics, pod health,
    and self-debugging diagnostics. Use --output json for AI agent consumption
    (NDJSON: one JSON object per line per refresh interval).

    Examples:
        # Watch last deployed benchmark (Rich TUI)
        aiperf kube watch

        # Watch a specific benchmark
        aiperf kube watch my-benchmark

        # AI agent mode (NDJSON output)
        aiperf kube watch --output json

        # Watch all running benchmarks
        aiperf kube watch --all

        # Include log tail in JSON output
        aiperf kube watch --output json --follow-logs
    """
    await _run_watch(
        job_id=job_id,
        manage_options=manage_options or KubeManageOptions(),
        all_jobs=all_jobs,
        output=output,
        interval=interval,
        follow_logs=follow_logs,
    )


def _build_renderer(output: Literal["json", "text", "rich"]) -> WatchRenderer:
    from aiperf.kubernetes.watch_render_json import JsonRenderer
    from aiperf.kubernetes.watch_render_rich import RichRenderer
    from aiperf.kubernetes.watch_render_text import TextRenderer

    if output == "json":
        return JsonRenderer()
    if output == "text":
        return TextRenderer()
    return RichRenderer()


async def _run_watch(
    *,
    job_id: str | None,
    manage_options: KubeManageOptions,
    all_jobs: bool,
    output: Literal["json", "text", "rich"],
    interval: float,
    follow_logs: bool,
) -> None:
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Watching Benchmark"):
        from aiperf.kubernetes.watch_orchestrator import WatchOrchestrator

        orchestrator = WatchOrchestrator(
            job_id=job_id,
            namespace=manage_options.namespace,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
            all_jobs=all_jobs,
            renderer=_build_renderer(output),
            interval=interval,
            follow_logs=follow_logs,
        )
        await orchestrator.run()
