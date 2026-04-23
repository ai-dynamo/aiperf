# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach and auto-attach workflows for kube commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
)
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.logs import save_pod_logs
from aiperf.kubernetes.port_forward import port_forward_with_status
from aiperf.kubernetes.results import (
    retrieve_all_artifacts,
    stream_controller_logs,
)
from aiperf.kubernetes.ui_dispatch import API_WS_PATH, stream_progress

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient


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


async def _poll_cr_status(
    custom: client.CustomObjectsApi,
    namespace: str,
    job_id: str,
) -> dict | None:
    """Fetch AIPerfJob CR, returning None on 404 (caller should retry)."""
    try:
        return await custom.get_namespaced_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            namespace=namespace,
            name=job_id,
        )
    except ApiException as e:
        if e.status == 404:
            return None
        raise


def _log_new_conditions(
    cli_logger: Any,
    conditions: list[dict],
    prev_count: int,
    elapsed: float,
) -> int:
    """Log conditions appended since the previous poll; return new count."""
    if len(conditions) <= prev_count:
        return prev_count
    for cond in conditions[prev_count:]:
        icon = (
            "[green]PASS[/green]" if cond.get("status") == "True" else "[red]FAIL[/red]"
        )
        cli_logger.info(
            f"  [{elapsed:>3.0f}s] {icon} {cond.get('type', '')}: "
            f"{cond.get('message', '')[:100]}"
        )
    return len(conditions)


async def watch_job(
    namespace: str,
    job_id: str,
    *,
    timeout: int = 600,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> dict:
    """Watch an AIPerfJob CR until it reaches a terminal phase.

    Runs the production :class:`BenchmarkWatchdog` as a background task
    while polling the AIPerfJob CR every 2s for ``status.phase``,
    ``status.conditions``, and ``status.workers``. Newly-observed
    conditions are logged incrementally; a phase/worker heartbeat is
    logged every 10s. Returns when the phase is ``"Completed"``,
    ``"Failed"``, or ``"Cancelled"``.

    Side effects:
        - Writes progress lines to the ``aiperf.kubernetes.console`` logger.
        - Calls ``print_success``/``print_error`` on terminal phase.
        - Starts and tears down a :class:`BenchmarkWatchdog` background task.

    Args:
        namespace: Kubernetes namespace containing the AIPerfJob CR.
        job_id: The AIPerfJob CR name (same as the job id).
        timeout: Maximum seconds to wait for a terminal phase before
            raising ``TimeoutError``.
        kubeconfig: Path to kubeconfig file (falls back to in-cluster /
            default kubeconfig resolution via :func:`k8s_client`).
        kube_context: Kubernetes context name.

    Returns:
        The terminal ``status`` dict from the AIPerfJob CR. Common keys:

        - ``phase`` (``str``): ``"Completed"``, ``"Failed"``, ``"Cancelled"``.
        - ``conditions`` (``list[dict]``): CR condition entries with
          ``type``, ``status``, ``message``.
        - ``workers`` (``dict``): ``{"ready": int, "total": int}``.
        - ``error`` (``str``): present when phase is ``"Failed"``.
        - ``jobId`` (``str``): echoed job id.

    Raises:
        TimeoutError: ``timeout`` seconds elapsed without a terminal phase.
        ApiException: Non-404 Kubernetes API error while polling the CR.

    Example:
        >>> status = await watch_job(
        ...     namespace="aiperf-bench",
        ...     job_id="aiperf-bench-7f2a",
        ...     timeout=600,
        ... )
        >>> status["phase"]
        'Completed'
        >>> status["workers"]
        {'ready': 8, 'total': 8}
    """
    import asyncio

    from aiperf.kubernetes.watchdog import BenchmarkWatchdog, K8sWatchdogSource

    prev_cond_count = 0
    terminal_phases = {"Completed", "Failed", "Cancelled"}

    from aiperf.kubernetes.console import logger as cli_logger

    async with k8s_client(kubeconfig=kubeconfig, context=kube_context) as api:
        source = K8sWatchdogSource(api)
        custom = client.CustomObjectsApi(api)

        async with BenchmarkWatchdog(
            source,
            namespace,
            timeout=timeout,
            poll_interval=5.0,
            status_interval=10.0,
            log=cli_logger,
        ):
            start = asyncio.get_running_loop().time()
            last_status_log = 0.0
            last_poll_error: Exception | None = None

            while True:
                elapsed = asyncio.get_running_loop().time() - start

                try:
                    raw = await _poll_cr_status(custom, namespace, job_id)
                    if raw is None:
                        if elapsed > 30:
                            cli_logger.warning(
                                f"[{elapsed:.0f}s] AIPerfJob {job_id} not found"
                            )
                        await asyncio.sleep(5)
                        continue

                    cr_status = raw.get("status", {})
                    phase = cr_status.get("phase", "Pending")
                    conditions = cr_status.get("conditions", [])
                    workers = cr_status.get("workers", {})

                    prev_cond_count = _log_new_conditions(
                        cli_logger, conditions, prev_cond_count, elapsed
                    )

                    # Phase/worker status every 10s
                    if elapsed - last_status_log >= 10:
                        w_ready = workers.get("ready", 0)
                        w_total = workers.get("total", "?")
                        cli_logger.info(
                            f"  [{elapsed:>3.0f}s] phase=[cyan]{phase}[/cyan]  "
                            f"workers={w_ready}/{w_total}"
                        )
                        last_status_log = elapsed

                    if phase in terminal_phases:
                        if phase == "Completed":
                            print_success(f"Benchmark completed ({elapsed:.0f}s)")
                        elif phase == "Failed":
                            error = cr_status.get("error", "unknown error")
                            print_error(f"Benchmark failed: {error}")
                        return cr_status

                except (
                    ApiException,
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    OSError,
                ) as e:
                    last_poll_error = e
                    cli_logger.warning(f"[{elapsed:.0f}s] CR poll error: {e}")

                if elapsed > timeout:
                    raise TimeoutError(
                        f"Benchmark {job_id} in {namespace} did not complete "
                        f"after {timeout}s (last CR poll error: {last_poll_error!r}). "
                        f"Check: kubectl get pods -n {namespace}"
                    ) from last_poll_error

                await asyncio.sleep(2)


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
    kube_creds = {"kubeconfig": kubeconfig, "kube_context": kube_context}

    async with k8s_client(kubeconfig=kubeconfig, context=kube_context) as api:
        if wait_for_ready:
            pod_name = await wait_for_controller_pod_ready(
                api, namespace, job_id, timeout=300
            )
            print_success(f"Controller pod ready: {pod_name}")
        else:
            result = await find_controller_pod(api, namespace, job_id)
            if not result:
                raise RuntimeError(
                    f"No controller pod found for job {job_id}. "
                    f"Remove --no-wait to wait for pod readiness."
                )
            pod_name, _ = result

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
