# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach and auto-attach workflows for kube commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import (
    find_controller_pod,
    find_jobset,
    k8s_client,
    wait_for_controller_pod_ready,
)
from aiperf.kubernetes.console import (
    logger,
    print_action,
    print_benchmark_complete,
    print_error,
    print_info,
    print_results_summary,
    print_success,
    print_warning,
)
from aiperf.kubernetes.constants import Containers
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.logs import save_pod_logs
from aiperf.kubernetes.port_forward import port_forward_with_status
from aiperf.kubernetes.results import (
    retrieve_all_artifacts,
    stream_controller_logs,
)
from aiperf.kubernetes.ui_dispatch import API_WS_PATH, stream_progress
from aiperf.kubernetes.watch import watch_job

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

__all__ = [
    "attach_to_benchmark",
    "auto_attach_workflow",
    "retrieve_and_display_results",
    "watch_job",
]


async def _fetch_and_print_pod_logs(
    api: ApiClient,
    namespace: str,
    job_id: str,
    *,
    tail: int = 30,
) -> None:
    """Best-effort fetch and display of controller pod logs.

    Args:
        api: Connected kubernetes_asyncio ApiClient.
        namespace: Kubernetes namespace.
        job_id: AIPerf job ID.
        tail: Number of log lines to display.
    """
    try:
        pod_info = await find_controller_pod(api, namespace, job_id)
        if not pod_info:
            return
        pod_name, _ = pod_info
        core = client.CoreV1Api(api)
        log_text = await core.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            tail_lines=tail,
        )
        if log_text.strip():
            logger.info("")
            logger.info(f"[dim]Last {tail} lines from controller pod {pod_name}:[/dim]")
            for line in log_text.strip().splitlines():
                logger.info(f"[dim]  {line}[/dim]")
    except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError):
        # Best-effort diagnostic: never fail the caller because logs are
        # unavailable (pod deleted mid-read, API unreachable, etc.).
        return


async def attach_to_benchmark(
    job_id: str,
    namespace: str,
    local_port: int,
    api: ApiClient,
    *,
    phase: str | None = None,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Attach to a running benchmark and stream progress.

    Args:
        job_id: The job ID to attach to.
        namespace: Namespace containing the job.
        local_port: Local port for port-forward.
        api: Connected kubernetes_asyncio ApiClient (from resolve_job).
        phase: Current job phase (from CR status), used for early exit.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.
    """
    kube_creds = {"kubeconfig": kubeconfig, "kube_context": kube_context}

    if phase == "Completed":
        print_warning(f"Job {job_id} has already completed.")
        print_action("Use 'aiperf kube results' to retrieve results.")
        return
    if phase == "Failed":
        print_error(f"Job {job_id} has failed.")
        await _fetch_and_print_pod_logs(api, namespace, job_id)
        print_action("Use 'aiperf kube logs' to investigate.")
        return

    pod_info = await find_controller_pod(api, namespace, job_id)
    if not pod_info:
        print_warning(
            f"No controller pod found for job {job_id}. The benchmark may still be starting."
        )
        return

    pod_name, pod_phase = pod_info
    if pod_phase != PodPhase.RUNNING:
        print_warning(f"Controller pod {pod_name} is not ready (status: {pod_phase})")
        return

    print_info(f"Attaching to job {job_id} in namespace {namespace}")
    print_success(f"Controller pod: {pod_name}")

    async with port_forward_with_status(
        namespace, pod_name, local_port, **kube_creds
    ) as port:
        ws_url = f"ws://localhost:{port}{API_WS_PATH}"
        await stream_progress(ws_url)


async def _resolve_controller_pod(
    api: ApiClient,
    namespace: str,
    job_id: str,
    *,
    wait_for_ready: bool,
) -> str:
    """Return the controller pod name, optionally waiting for Running."""
    if wait_for_ready:
        pod_name = await wait_for_controller_pod_ready(
            api, namespace, job_id, timeout=300
        )
        print_success(f"Controller pod ready: {pod_name}")
        return pod_name

    result = await find_controller_pod(api, namespace, job_id)
    if not result:
        raise RuntimeError(
            f"No controller pod found for job {job_id}. "
            f"Remove --no-wait to wait for pod readiness."
        )
    pod_name, _ = result
    return pod_name


async def _stream_progress_or_logs(
    namespace: str,
    pod_name: str,
    attach_port: int,
    *,
    stream_ws: bool,
    kube_creds: dict[str, str | None],
) -> None:
    """Stream live progress via WebSocket (with port-forward) or by tailing logs."""
    if stream_ws:
        async with port_forward_with_status(
            namespace, pod_name, attach_port, **kube_creds
        ) as port:
            ws_url = f"ws://localhost:{port}{API_WS_PATH}"
            await stream_progress(ws_url)
    else:
        logger.info("")
        await stream_controller_logs(
            namespace, pod_name, container=Containers.CONTROL_PLANE, **kube_creds
        )


async def auto_attach_workflow(
    job_id: str,
    namespace: str,
    attach_port: int,
    *,
    wait_for_ready: bool = True,
    stream_ws: bool = False,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Execute the post-deploy auto-attach workflow: wait, stream, retrieve results.

    After ``aiperf kube profile`` creates the AIPerfJob CR, this helper:

    1. Optionally waits for the controller pod to reach ``Running``
       (``wait_for_ready=True``).
    2. Streams live progress to the terminal — via the controller's
       WebSocket if ``stream_ws=True``, otherwise by tailing the
       controller pod's stdout.
    3. On benchmark completion, prints the completion banner and
       downloads all result artifacts into ``./artifacts/<job_id>/``.

    Side effects:
        - Creates the directory ``./artifacts/<job_id>/`` in the caller's cwd.
        - Writes progress / status lines to the ``aiperf.kubernetes.console``
          CLI logger (``print_info``, ``print_success``, ``print_action``).
        - Opens a ``kubectl port-forward`` subprocess when ``stream_ws=True``.
        - Downloads result files and pod-logs archives into the artifacts dir.

    Args:
        job_id: AIPerfJob CR name to attach to.
        namespace: Namespace containing the AIPerfJob.
        attach_port: Local port for the port-forward (pass ``0`` for an
            ephemeral port).
        wait_for_ready: If True, wait up to 300s for the controller pod to
            reach ``Running``. If False and no pod exists yet, raises
            ``RuntimeError``.
        stream_ws: If True, stream progress via the controller's WebSocket
            (requires port-forward). If False, tail controller pod logs.
        kubeconfig: Path to kubeconfig file (falls back to in-cluster /
            default kubeconfig resolution via :func:`k8s_client`).
        kube_context: Kubernetes context name.

    Raises:
        RuntimeError: ``wait_for_ready=False`` and no controller pod found.
        TimeoutError: ``wait_for_ready=True`` and the controller pod did
            not reach ``Running`` within 300s.
        ConnectionError: WebSocket streaming failed after all retries
            (raised from :func:`stream_progress_from_api`).
        ApiException: Underlying Kubernetes API error.

    Example:
        >>> await auto_attach_workflow(
        ...     job_id="aiperf-bench-7f2a",
        ...     namespace="aiperf-bench",
        ...     attach_port=0,
        ...     wait_for_ready=True,
        ...     stream_ws=False,
        ... )
        # ...live controller logs stream to terminal...
        # Benchmark complete. Retrieving results...
        # Results saved to ./artifacts/aiperf-bench-7f2a/
    """
    kube_creds: dict[str, str | None] = {
        "kubeconfig": kubeconfig,
        "kube_context": kube_context,
    }

    async with k8s_client(kubeconfig=kubeconfig, context=kube_context) as api:
        pod_name = await _resolve_controller_pod(
            api, namespace, job_id, wait_for_ready=wait_for_ready
        )
        await _stream_progress_or_logs(
            namespace,
            pod_name,
            attach_port,
            stream_ws=stream_ws,
            kube_creds=kube_creds,
        )

        print_benchmark_complete()
        print_info("Retrieving results...")
        await retrieve_and_display_results(job_id, namespace, api, **kube_creds)


async def retrieve_and_display_results(
    job_id: str,
    namespace: str,
    api: ApiClient,
    *,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Retrieve all artifacts from API service and display summary."""
    output_dir = Path(f"./artifacts/{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    jobset_info = await find_jobset(api, job_id, namespace)

    success = await retrieve_all_artifacts(
        job_id,
        namespace,
        output_dir,
        jobset_info,
        api,
        local_port=0,
        kubeconfig=kubeconfig,
        kube_context=kube_context,
    )

    await save_pod_logs(
        job_id,
        namespace,
        output_dir,
        api,
        kubeconfig=kubeconfig,
        kube_context=kube_context,
    )

    if success:
        print_results_summary(str(output_dir))
    else:
        print_warning("Results not yet available from API")
        print_action(f"Try: aiperf kube results {job_id} --namespace {namespace}")
