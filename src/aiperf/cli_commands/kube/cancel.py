# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube cancel command: request cancellation of a running benchmark."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="cancel")

_TERMINAL_PHASES = frozenset({"Completed", "Failed", "Cancelled"})


@app.default
async def cancel(
    job_id: Annotated[
        str | None,
        Parameter(
            help="The AIPerf job ID or AIPerfSweep name to cancel (default: last deployed job)."
        ),
    ] = None,
    *,
    manage_options: KubeManageOptions | None = None,
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
    kind: Annotated[
        Literal["job", "sweep"] | None,
        Parameter(
            name="--kind",
            help="Target kind when an AIPerfJob and AIPerfSweep share a name.",
        ),
    ] = None,
) -> None:
    """Cancel a running AIPerf benchmark.

    Patches ``spec.cancel: true`` on the CR. The operator's cancel handler
    tears down the JobSet, stamps ``status.phase=Cancelled``, and emits a
    Cancelled event. Cancelling an already-terminal benchmark is a no-op.

    Examples:
        # Cancel the last deployed job
        aiperf kube cancel

        # Cancel a specific job
        aiperf kube cancel abc123

        # Cancel a whole sweep, or one variation of it
        aiperf kube cancel my-sweep
        aiperf kube cancel my-sweep -v 7
    """
    from aiperf import cli_utils
    from aiperf.cli_commands.kube._kube_common import resolve_child_name

    manage_options = manage_options or KubeManageOptions()
    use_last_benchmark = job_id is None
    if job_id is not None:
        child = resolve_child_name(job_id, variation=variation, trial=trial)
        if child is not None:
            job_id = child

    with cli_utils.exit_on_error(title="Error Cancelling Benchmark"):
        from kubernetes_asyncio import client as k8s_client_mod

        from aiperf.cli_commands.kube._kube_delete import (
            AmbiguousAIPerfTargetError,
            find_aiperf_cr,
            workload_kind_from_cli,
        )
        from aiperf.kubernetes import cli_helpers
        from aiperf.kubernetes import console as kube_console
        from aiperf.kubernetes.client import k8s_client
        from aiperf.kubernetes.cr_refs import (
            AIPERF_JOB_GROUP,
            AIPERF_JOB_VERSION,
        )

        resolved = cli_helpers.resolve_job_id_and_namespace(
            job_id, manage_options.namespace
        )
        if not resolved:
            return
        job_id, namespace = resolved
        requested_kind = workload_kind_from_cli(kind)
        if requested_kind is None and use_last_benchmark:
            last = kube_console.get_last_benchmark()
            requested_kind = last.kind if last is not None else None

        async with k8s_client(
            kubeconfig=manage_options.kubeconfig,
            context=manage_options.kube_context,
        ) as api:
            custom = k8s_client_mod.CustomObjectsApi(api)
            try:
                found = await find_aiperf_cr(
                    custom,
                    namespace=namespace,
                    name=job_id,
                    kind=requested_kind,
                )
            except AmbiguousAIPerfTargetError as error:
                kube_console.print_error(str(error))
                return
            if found is None:
                expected = requested_kind or "AIPerfJob or AIPerfSweep"
                kube_console.print_error(
                    f"No {expected} named {job_id!r} in namespace {namespace}"
                )
                return

            plural, cr = found
            phase = (cr.get("status") or {}).get("phase")
            if phase in _TERMINAL_PHASES:
                kube_console.print_info(
                    f"{job_id} is already {phase}; nothing to cancel."
                )
                return
            await custom.patch_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=plural,
                namespace=namespace,
                name=job_id,
                body={"spec": {"cancel": True}},
            )
            kube_console.print_success(
                f"Cancellation requested for {job_id} in namespace {namespace}"
            )
            kube_console.print_info(
                f"Watch it wind down with: aiperf kube list {job_id} --watch"
            )
