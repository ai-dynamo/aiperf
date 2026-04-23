# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Poll an AIPerfJob CR to terminal phase while running the BenchmarkWatchdog."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.console import (
    print_error,
    print_success,
)
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
)

_TERMINAL_PHASES = {"Completed", "Failed", "Cancelled"}


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


def _log_terminal_phase(cr_status: dict, phase: str, elapsed: float) -> None:
    """Print the completion/failure banner for a terminal CR phase."""
    if phase == "Completed":
        print_success(f"Benchmark completed ({elapsed:.0f}s)")
    elif phase == "Failed":
        error = cr_status.get("error", "unknown error")
        print_error(f"Benchmark failed: {error}")


def _log_phase_heartbeat(
    cli_logger: Any, phase: str, workers: dict, elapsed: float
) -> None:
    """Log a one-line phase/worker heartbeat."""
    w_ready = workers.get("ready", 0)
    w_total = workers.get("total", "?")
    cli_logger.info(
        f"  [{elapsed:>3.0f}s] phase=[cyan]{phase}[/cyan]  workers={w_ready}/{w_total}"
    )


async def _process_cr_poll(
    custom: client.CustomObjectsApi,
    namespace: str,
    job_id: str,
    *,
    cli_logger: Any,
    state: dict,
    elapsed: float,
) -> dict | None:
    """Poll CR once; return terminal ``cr_status`` or None to keep looping.

    Mutates ``state`` keys ``prev_cond_count`` and ``last_status_log``.
    """
    raw = await _poll_cr_status(custom, namespace, job_id)
    if raw is None:
        if elapsed > 30:
            cli_logger.warning(f"[{elapsed:.0f}s] AIPerfJob {job_id} not found")
        await asyncio.sleep(5)
        return None

    cr_status = raw.get("status", {})
    phase = cr_status.get("phase", "Pending")
    conditions = cr_status.get("conditions", [])
    workers = cr_status.get("workers", {})

    state["prev_cond_count"] = _log_new_conditions(
        cli_logger, conditions, state["prev_cond_count"], elapsed
    )

    if elapsed - state["last_status_log"] >= 10:
        _log_phase_heartbeat(cli_logger, phase, workers, elapsed)
        state["last_status_log"] = elapsed

    if phase in _TERMINAL_PHASES:
        _log_terminal_phase(cr_status, phase, elapsed)
        return cr_status
    return None


async def _poll_until_terminal(
    custom: client.CustomObjectsApi,
    namespace: str,
    job_id: str,
    *,
    cli_logger: Any,
    timeout: int,
) -> dict:
    """Poll the AIPerfJob CR every 2s until terminal or timeout."""
    state: dict[str, Any] = {"prev_cond_count": 0, "last_status_log": 0.0}
    start = asyncio.get_running_loop().time()
    last_poll_error: Exception | None = None

    while True:
        elapsed = asyncio.get_running_loop().time() - start
        try:
            terminal = await _process_cr_poll(
                custom,
                namespace,
                job_id,
                cli_logger=cli_logger,
                state=state,
                elapsed=elapsed,
            )
            if terminal is not None:
                return terminal
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

    Args:
        namespace: Kubernetes namespace containing the AIPerfJob CR.
        job_id: The AIPerfJob CR name (same as the job id).
        timeout: Maximum seconds to wait for a terminal phase before
            raising ``TimeoutError``.
        kubeconfig: Path to kubeconfig file (falls back to in-cluster /
            default kubeconfig resolution via :func:`k8s_client`).
        kube_context: Kubernetes context name.

    Returns:
        The terminal ``status`` dict from the AIPerfJob CR. Common keys
        are ``phase``, ``conditions``, ``workers``, ``error``, ``jobId``.

    Raises:
        TimeoutError: ``timeout`` seconds elapsed without a terminal phase.
        ApiException: Non-404 Kubernetes API error while polling the CR.
    """
    from aiperf.kubernetes.console import logger as cli_logger
    from aiperf.kubernetes.watchdog import BenchmarkWatchdog, K8sWatchdogSource

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
            return await _poll_until_terminal(
                custom,
                namespace,
                job_id,
                cli_logger=cli_logger,
                timeout=timeout,
            )
