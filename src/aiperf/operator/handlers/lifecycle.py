# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle handler logic: on_delete, on_cancel, on_benchmark_complete.

This module contains the business logic only — no kopf decorators.
Decorators live in ``aiperf.operator.main``.
"""

from __future__ import annotations

import logging
from typing import Any

import kopf
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.operator import events
from aiperf.operator.client_cache import (
    close_progress_client,
    get_or_create_progress_client,
    job_key,
    request_cancellation,
    try_claim_completion,
)
from aiperf.operator.handlers.completion import handle_completion
from aiperf.operator.status import Phase, StatusBuilder

logger = logging.getLogger(__name__)


async def on_delete(
    name: str, namespace: str, status: dict[str, Any], **_: Any
) -> None:
    """Handle AIPerfJob CR deletion.

    Side effects:
        - Sets a sticky in-process cancellation flag for this job so any
          in-flight monitor/completion coroutines short-circuit at their
          next await boundary (avoids blocking delete on fetch backoff).
        - Closes the cached ProgressClient (releases aiohttp session).
        - Relies on Kubernetes ownerReferences GC to reap the JobSet,
          ConfigMap, Role, and RoleBinding — this handler does NOT delete
          them directly.

    The cancellation flag is set BEFORE closing the client so concurrent
    observers see the flag before the client-cache entry disappears.
    """
    job_id = status.get("jobId", name)
    key = job_key(namespace, job_id)
    # Request cancellation FIRST so any concurrent monitor/completion work
    # sees the flag before we free the client. close_progress_client also
    # clears the cancellation event, so the request must be made first.
    request_cancellation(key)
    await close_progress_client(key)
    logger.info(f"Deleting AIPerfJob {namespace}/{name}")


async def on_cancel(
    body: dict[str, Any],
    spec: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle cancellation request via ``spec.cancel`` field.

    Fires on every ``spec.cancel`` update; no-ops unless ``spec.cancel`` is
    truthy and the CR is not already terminal.

    Side effects:
        - Deletes the JobSet custom object (logs a warning on non-404
          failures but does not re-raise).
        - Closes the cached ProgressClient for this job.
        - Patches ``status.phase`` to ``Cancelled`` and sets completion time.
        - Emits a ``Cancelled`` kopf event on the CR.

    Unlike ``on_delete`` this does NOT set the in-process cancellation
    flag — the CR is staying around in ``Cancelled`` phase for user
    inspection; there are no in-flight completion paths to abort because
    the monitor handler will see the terminal phase on its next tick.
    """
    if not spec.get("cancel"):
        return

    current_phase = status.get("phase", Phase.PENDING)
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return  # Already terminal

    job_id = status.get("jobId", name)
    jobset_name = status.get("jobSetName")

    logger.info(f"Cancelling AIPerfJob {namespace}/{name}")

    sb = StatusBuilder(patch, status)

    if jobset_name:
        try:
            async with k8s_client() as api:
                await client.CustomObjectsApi(api).delete_namespaced_custom_object(
                    group=JOBSET_GROUP,
                    version=JOBSET_VERSION,
                    plural=JOBSET_PLURAL,
                    namespace=namespace,
                    name=jobset_name,
                )
            logger.info(f"Deleted JobSet {jobset_name}")
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Failed to delete JobSet: {e}")

    await close_progress_client(job_key(namespace, job_id))
    sb.set_phase(Phase.CANCELLED).set_completion_time()
    sb.finalize()
    events.cancelled(body, job_id)


async def on_benchmark_complete(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle benchmark completion signal from controller pod.

    The controller pod patches the ``benchmark-complete`` annotation after
    results are exported. This handler fires immediately via kopf's watch
    mechanism, bypassing the 10-second monitor poll cycle.

    Side effects:
        - Attempts to claim completion via ``try_claim_completion`` (durable
          CR annotation); returns silently if another handler already won.
        - Delegates to ``handle_completion`` (fetches results, patches CR
          status, updates the job index, emits ``Completed``/``ResultsStored``
          events, deletes the JobSet on success).
        - Sends a shutdown signal to the controller pod's HTTP API so it
          exits cleanly; on failure emits a ``ShutdownSignalFailed`` warning
          event but does not re-raise (results are already stored).
        - Closes the cached ProgressClient.
    """
    current_phase = status.get("phase", Phase.PENDING)
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return

    job_id = status.get("jobId", name)
    jobset_name = status.get("jobSetName")
    if not jobset_name:
        return

    key = job_key(namespace, job_id)
    if not await try_claim_completion(namespace, name, body):
        return

    logger.info(
        f"Benchmark completion signal received for {namespace}/{name}, fetching results"
    )

    sb = StatusBuilder(patch, status)
    await handle_completion(body, namespace, jobset_name, job_id, status=status, sb=sb)

    host = controller_dns_name(jobset_name, namespace)
    try:
        progress_client = await get_or_create_progress_client(key)
        await progress_client.send_shutdown(host)
    except Exception as e:
        logger.exception(f"Failed to send shutdown to {host}")
        kopf.event(
            body,
            type="Warning",
            reason="ShutdownSignalFailed",
            message=f"Failed to send shutdown to controller at {host}: {e}",
        )

    await close_progress_client(key)
