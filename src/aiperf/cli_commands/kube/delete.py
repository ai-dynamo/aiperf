# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube delete command: remove a benchmark and its backing resources."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="delete")


@app.default
async def delete(
    job_id: Annotated[
        str | None,
        Parameter(
            help="The AIPerf job ID or AIPerfSweep name to delete (default: last deployed job)."
        ),
    ] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    force: Annotated[
        bool,
        Parameter(name=["-f", "--force"], help="Skip the confirmation prompt."),
    ] = False,
    kind: Annotated[
        Literal["job", "sweep"] | None,
        Parameter(
            name="--kind",
            help="Target kind when an AIPerfJob and AIPerfSweep share a name.",
        ),
    ] = None,
) -> None:
    """Delete an AIPerf benchmark.

    Deletes the AIPerfJob or AIPerfSweep custom resource. The JobSet, pods and
    ConfigMap carry ownerReferences back to it, so Kubernetes garbage-collects
    them; results already harvested onto the operator's PVC are untouched.

    The namespace itself is never touched: AIPerf does not create it.

    Examples:
        # Delete the last deployed benchmark (asks first)
        aiperf kube delete

        # Delete a specific one without prompting
        aiperf kube delete abc123 --force
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Deleting Benchmark"):
        from kubernetes_asyncio import client as k8s_client_mod

        from aiperf.cli_commands.kube._kube_delete import (
            AmbiguousAIPerfTargetError,
            confirm_action,
            delete_aiperf_cr,
            find_aiperf_cr,
            kind_for_plural,
            workload_kind_from_cli,
        )
        from aiperf.kubernetes import cli_helpers
        from aiperf.kubernetes import console as kube_console
        from aiperf.kubernetes.client import k8s_client

        use_last_benchmark = job_id is None
        resolved = cli_helpers.resolve_job_id_and_namespace(
            job_id,
            manage_options.namespace,
            kubeconfig=manage_options.kubeconfig,
            context=manage_options.kube_context,
        )
        if not resolved:
            raise SystemExit(1)
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
                raise SystemExit(1) from None
            if found is None:
                expected = requested_kind or "AIPerfJob or AIPerfSweep"
                kube_console.print_error(
                    f"No {expected} named {job_id!r} in namespace {namespace}"
                )
                raise SystemExit(1)
            plural, cr = found
            found_kind = kind_for_plural(plural)

            target = f"{found_kind} {job_id} in namespace {namespace}"
            if not force and not confirm_action(f"Delete {target}?"):
                kube_console.print_info("Aborted.")
                return

            await delete_aiperf_cr(
                custom,
                plural=plural,
                namespace=namespace,
                name=job_id,
                kind=found_kind,
                cr=cr,
            )
