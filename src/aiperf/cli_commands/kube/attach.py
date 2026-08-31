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
            help="When job_id is an AIPerfSweep name, target child variation index (0..199). Resolves to <sweep>-v<idx:02d>[-t<trial>].",
        ),
    ] = None,
    trial: Annotated[
        int | None,
        Parameter(
            name=["-t", "--trial"],
            help="Trial index (0..9) within a sweep variation. Requires -v.",
        ),
    ] = None,
    ignore_not_found: Annotated[
        bool,
        Parameter(
            name="--ignore-not-found",
            help="Exit 0 instead of 1 when the benchmark does not exist (mirrors kubectl).",
        ),
    ] = False,
) -> None:
    """Attach to a running AIPerf benchmark and stream progress.

    Connects to a running benchmark via port-forward and streams real-time
    progress updates via WebSocket. Press Ctrl+C to disconnect.

    If no job_id is specified, uses the last deployed benchmark.

    Exits 1 when the target cannot be addressed at all (no such AIPerfJob,
    AIPerfSweep or JobSet, or no job_id and no last-benchmark record), so the
    command works as a CI existence check. Pass ``--ignore-not-found`` to keep
    exit 0 in that case.

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

        # Tolerate an already-deleted benchmark in a cleanup script
        aiperf kube attach abc123 --ignore-not-found
    """
    from aiperf import cli_utils
    from aiperf.cli_commands.kube._kube_common import resolve_child_target
    from aiperf.kubernetes import attach as kube_attach
    from aiperf.kubernetes import cli_helpers

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Attaching to Benchmark"):
        job_id = resolve_child_target(
            job_id, variation=variation, trial=trial, command="kube attach"
        )

        resolved = await cli_helpers.resolve_job(
            job_id,
            manage_options.namespace,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )
        if not resolved:
            cli_helpers.exit_target_not_found(ignore_not_found=ignore_not_found)
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
