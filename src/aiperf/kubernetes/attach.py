# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach and auto-attach workflows for kube commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
    except Exception:
        pass


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


async def watch_job(
    namespace: str,
    job_id: str,
    timeout: int = 600,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> dict:
    """Watch an AIPerfJob CR until terminal, with full watchdog monitoring.

    Runs the production watchdog in the background while polling the CR
    for phase transitions. Returns the final CR status dict.
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

            while True:
                elapsed = asyncio.get_running_loop().time() - start

                # Poll CR status
                try:
                    try:
                        raw = await custom.get_namespaced_custom_object(
                            group=AIPERF_JOB_GROUP,
                            version=AIPERF_JOB_VERSION,
                            plural=AIPERF_JOB_PLURAL,
                            namespace=namespace,
                            name=job_id,
                        )
                    except ApiException as e:
                        if e.status == 404:
                            if elapsed > 30:
                                cli_logger.warning(
                                    f"[{elapsed:.0f}s] AIPerfJob {job_id} not found"
                                )
                            await asyncio.sleep(5)
                            continue
                        raise

                    cr_status = raw.get("status", {})
                    phase = cr_status.get("phase", "Pending")
                    conditions = cr_status.get("conditions", [])
                    workers = cr_status.get("workers", {})

                    # Print new conditions as they appear
                    if len(conditions) > prev_cond_count:
                        for cond in conditions[prev_cond_count:]:
                            icon = (
                                "[green]PASS[/green]"
                                if cond.get("status") == "True"
                                else "[red]FAIL[/red]"
                            )
                            cli_logger.info(
                                f"  [{elapsed:>3.0f}s] {icon} {cond.get('type', '')}: "
                                f"{cond.get('message', '')[:100]}"
                            )
                        prev_cond_count = len(conditions)

                    # Phase/worker status every 10s
                    if elapsed - last_status_log >= 10:
                        w_ready = workers.get("ready", 0)
                        w_total = workers.get("total", "?")
                        cli_logger.info(
                            f"  [{elapsed:>3.0f}s] phase=[cyan]{phase}[/cyan]  "
                            f"workers={w_ready}/{w_total}"
                        )
                        last_status_log = elapsed

                    # Terminal?
                    if phase in terminal_phases:
                        if phase == "Completed":
                            print_success(f"Benchmark completed ({elapsed:.0f}s)")
                        elif phase == "Failed":
                            error = cr_status.get("error", "unknown error")
                            print_error(f"Benchmark failed: {error}")
                        return cr_status

                except Exception as e:
                    cli_logger.warning(f"[{elapsed:.0f}s] CR poll error: {e}")

                if elapsed > timeout:
                    raise TimeoutError(
                        f"Benchmark did not complete after {timeout}s. "
                        f"Check: kubectl get pods -n {namespace}"
                    )

                await asyncio.sleep(2)


async def auto_attach_workflow(
    job_id: str,
    namespace: str,
    attach_port: int,
    wait_for_ready: bool = True,
    stream_ws: bool = False,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Execute auto-attach workflow after deployment.

    Waits for controller pod, then streams progress via WebSocket or
    controller logs to the terminal. Retrieves results on completion.
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
