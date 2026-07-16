# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube attach command: connect to a running benchmark."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="attach")


@app.default
async def attach(
    job_id: Annotated[
        str | None,
        Parameter(
            help="The AIPerf job ID or AIPerfSweep name to attach to (default: last deployed job)."
        ),
    ] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    port: Annotated[
        int,
        Parameter(
            name=["-p", "--port"],
            help="Local port for port-forward (default: 0 = ephemeral).",
        ),
    ] = 0,
    variation: Annotated[
        int | None,
        Parameter(
            name=["-v", "--variation"],
            help="When job_id is an AIPerfSweep name, target child variation index (00..99). Resolves to <sweep>-v<idx:02d>[-t<trial>].",
        ),
    ] = None,
    trial: Annotated[
        int | None,
        Parameter(
            name=["-t", "--trial"],
            help="Trial index (0..9) within a sweep variation. Requires -v.",
        ),
    ] = None,
) -> None:
    """Attach to a running AIPerf benchmark and stream progress.

    Connects to a running benchmark via port-forward and streams real-time
    progress updates via WebSocket. Press Ctrl+C to disconnect.

    If no job_id is specified, uses the last deployed benchmark.

    Examples:
        # Attach to the last deployed benchmark
        aiperf kube attach

        # Attach to a specific job
        aiperf kube attach abc123

        # Attach to a job in a specific namespace
        aiperf kube attach abc123 --namespace aiperf-bench

        # Use a different local port
        aiperf kube attach abc123 --port 9091

        # Target a specific sweep variation
        aiperf kube attach my-sweep -v 7
        aiperf kube attach my-sweep -v 5 -t 0
    """
    from aiperf import cli_utils
    from aiperf.cli_commands.kube._kube_common import resolve_child_name
    from aiperf.kubernetes import attach as kube_attach
    from aiperf.kubernetes import cli_helpers

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Attaching to Benchmark"):
        if job_id is not None:
            child = resolve_child_name(job_id, variation=variation, trial=trial)
            if child is not None:
                job_id = child

        resolved = await cli_helpers.resolve_job(
            job_id,
            manage_options.namespace,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )
        if not resolved:
            return

        try:
            await kube_attach.attach_to_benchmark(
                resolved.job_id,
                resolved.namespace,
                port,
                resolved.api,
                phase=resolved.job_info.phase,
                kubeconfig=manage_options.kubeconfig,
                kube_context=manage_options.kube_context,
            )
        finally:
            await resolved.aclose()
