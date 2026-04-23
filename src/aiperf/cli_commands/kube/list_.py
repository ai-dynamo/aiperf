# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube list command: list benchmark jobs and their status."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

if TYPE_CHECKING:
    from aiperf.kubernetes.models import AIPerfJobInfo

app = App(name="list")


@app.default
async def list_jobs(
    job_id: Annotated[str | None, Parameter(help="Specific job ID to check.")] = None,
    *,
    all_namespaces: Annotated[
        bool,
        Parameter(name=["-A", "--all-namespaces"], help="Search in all namespaces."),
    ] = True,
    running: Annotated[
        bool, Parameter(name=["--running"], help="Show only running jobs.")
    ] = False,
    completed: Annotated[
        bool, Parameter(name=["--completed"], help="Show only completed jobs.")
    ] = False,
    failed: Annotated[
        bool, Parameter(name=["--failed"], help="Show only failed jobs.")
    ] = False,
    wide: Annotated[
        bool,
        Parameter(
            name=["-w", "--wide"],
            help="Show additional columns (model, endpoint, error).",
        ),
    ] = False,
    watch: Annotated[
        bool,
        Parameter(
            name=["--watch"],
            help="Refresh the list every few seconds until interrupted.",
        ),
    ] = False,
    interval: Annotated[
        int,
        Parameter(
            name=["--interval"],
            help="Refresh interval in seconds (used with --watch).",
        ),
    ] = 5,
    manage_options: KubeManageOptions | None = None,
) -> None:
    """List AIPerf benchmark jobs and their status.

    Lists AIPerfJob custom resources with operator-provided status.
    By default searches all namespaces. Use --namespace to limit scope.

    Examples:
        # List all AIPerf jobs (all namespaces)
        aiperf kube list

        # Watch jobs with live refresh
        aiperf kube list --watch

        # List only running jobs
        aiperf kube list --running

        # List jobs in a specific namespace
        aiperf kube list --namespace aiperf-bench

        # Get status of a specific job
        aiperf kube list abc123
    """
    import asyncio

    from aiperf import cli_utils
    from aiperf.kubernetes import client as kube_client_mod
    from aiperf.kubernetes import console as kube_console

    manage_options = manage_options or KubeManageOptions()

    active_filters = [
        ("Running", running),
        ("Completed", completed),
        ("Failed", failed),
    ]
    selected = [name for name, active in active_filters if active]
    if len(selected) > 1:
        kube_console.print_error(
            f"Only one status filter can be used at a time, got: {', '.join(f'--{s.lower()}' for s in selected)}"
        )
        raise SystemExit(1)
    status_filter = selected[0] if selected else None

    search_all = all_namespaces and manage_options.namespace is None

    with cli_utils.exit_on_error(title="Error Listing Kubernetes Jobs"):
        async with kube_client_mod.k8s_client(
            kubeconfig=manage_options.kubeconfig,
            context=manage_options.kube_context,
        ) as api:
            while True:
                jobs = await _fetch_jobs(
                    api,
                    manage_options=manage_options,
                    search_all=search_all,
                    status_filter=status_filter,
                    job_id=job_id,
                )

                if not jobs:
                    filter_msg = (
                        f" with phase '{status_filter}'" if status_filter else ""
                    )
                    kube_console.print_info(f"No AIPerf jobs found{filter_msg}")
                    if not watch:
                        return
                else:
                    if watch:
                        # Clear screen for live refresh
                        print("\033[2J\033[H", end="", flush=True)
                    kube_console.print_aiperfjob_table(jobs, wide=wide)

                if not watch:
                    return

                try:
                    await asyncio.sleep(interval)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    return


async def _fetch_jobs(
    api: object,
    *,
    manage_options: KubeManageOptions,
    search_all: bool,
    status_filter: str | None,
    job_id: str | None,
) -> list[AIPerfJobInfo]:
    """Fetch job list from cluster."""
    from aiperf.kubernetes.client import (
        find_aiperf_job,
        list_aiperf_jobs,
        list_jobsets,
    )

    if job_id:
        job_info = await find_aiperf_job(api, job_id, manage_options.namespace)
        return [job_info] if job_info else []

    jobs = await list_aiperf_jobs(
        api,
        namespace=manage_options.namespace,
        all_namespaces=search_all,
        status_filter=status_filter,
    )

    if not jobs:
        from aiperf.kubernetes.models import AIPerfJobInfo

        jobsets = await list_jobsets(
            api,
            namespace=manage_options.namespace,
            all_namespaces=search_all,
            job_id=job_id,
            status_filter=status_filter,
        )
        jobs = [
            AIPerfJobInfo(
                name=js.name,
                namespace=js.namespace,
                phase=js.status,
                job_id=js.job_id,
                jobset_name=js.name,
                created=js.created,
                model=js.model,
                endpoint=js.endpoint,
            )
            for js in jobsets
        ]
    return jobs
