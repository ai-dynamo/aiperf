# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube shutdown command: retire a finished benchmark's controller pod."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="shutdown")


@app.default
async def shutdown(
    job_id: Annotated[
        str | None,
        Parameter(
            help="The AIPerf job ID whose controller should shut down (default: last deployed job)."
        ),
    ] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    local_port: Annotated[
        int,
        Parameter(
            name="--local-port",
            help="Local port for the API port-forward (0 = ephemeral).",
        ),
    ] = 0,
) -> None:
    """Ask a completed benchmark's API service to exit.

    In Kubernetes the controller pod deliberately stays up after the run so it
    can serve results. This tells it to stop, letting the pod exit cleanly and
    release its CPU and memory reservation.

    The API refuses while the benchmark is still running -- use
    ``aiperf kube cancel`` to stop a run in progress.

    Examples:
        # Retire the last deployed benchmark's controller
        aiperf kube shutdown

        # Retire a specific one
        aiperf kube shutdown abc123 --namespace aiperf-bench
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Shutting Down Benchmark"):
        import aiohttp

        from aiperf.kubernetes import cli_helpers
        from aiperf.kubernetes import console as kube_console
        from aiperf.kubernetes.client import k8s_client
        from aiperf.kubernetes.client_pods import find_controller_pod
        from aiperf.kubernetes.port_forward import port_forward_to_controller

        resolved = cli_helpers.resolve_job_id_and_namespace(
            job_id, manage_options.namespace
        )
        if not resolved:
            return
        job_id, namespace = resolved

        async with k8s_client(
            kubeconfig=manage_options.kubeconfig,
            context=manage_options.kube_context,
        ) as api:
            pod = await find_controller_pod(api, namespace, job_id)
            if pod is None:
                kube_console.print_error(
                    f"No controller pod found for {job_id} in namespace {namespace}. "
                    f"It may already have exited."
                )
                return
            pod_name, _phase = pod

        async with (
            port_forward_to_controller(
                namespace,
                pod_name,
                local_port=local_port,
                kubeconfig=manage_options.kubeconfig,
                kube_context=manage_options.kube_context,
            ) as port,
            aiohttp.ClientSession() as session,
            session.post(
                f"http://localhost:{port}/api/shutdown",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp,
        ):
            if resp.status == 409:
                kube_console.print_warning(
                    f"{job_id} is still running; the API will not shut down "
                    f"mid-benchmark. Use `aiperf kube cancel {job_id}` to stop it."
                )
                return
            if resp.status >= 400:
                kube_console.print_error(
                    f"API returned HTTP {resp.status}: {await resp.text()}"
                )
                return

        kube_console.print_success(f"Shutdown requested for {job_id}")
        kube_console.print_info(
            "The controller pod exits once the API service stops; results already "
            "harvested by the operator remain available via `aiperf kube results`."
        )
