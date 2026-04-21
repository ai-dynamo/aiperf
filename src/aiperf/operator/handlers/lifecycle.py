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
import kr8s

from aiperf.kubernetes.client import get_api
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.kr8s_resources import AsyncJobSet
from aiperf.operator import events
from aiperf.operator.client_cache import (
    _shutdown_sent,
    close_progress_client,
    get_or_create_progress_client,
    job_key,
)
from aiperf.operator.handlers.completion import handle_completion
from aiperf.operator.status import Phase, StatusBuilder

logger = logging.getLogger(__name__)


async def on_delete(
    name: str, namespace: str, status: dict[str, Any], **_: Any
) -> None:
    """Handle deletion - clean up cached ProgressClient and let K8s GC handle resources."""
    job_id = status.get("jobId", name)
    await close_progress_client(job_key(namespace, job_id))
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
    """Handle cancellation request via spec.cancel field."""
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
            api = await get_api()
            js = await AsyncJobSet.get(jobset_name, namespace=namespace, api=api)
            await js.delete()
            logger.info(f"Deleted JobSet {jobset_name}")
        except kr8s.NotFoundError:
            pass
        except kr8s.ServerError as e:
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
    results are exported.  This handler fires immediately via kopf's watch
    mechanism, bypassing the 10-second monitor poll cycle.
    """
    current_phase = status.get("phase", Phase.PENDING)
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return

    job_id = status.get("jobId", name)
    jobset_name = status.get("jobSetName")
    if not jobset_name:
        return

    key = job_key(namespace, job_id)
    if key in _shutdown_sent:
        return

    logger.info(
        f"Benchmark completion signal received for {namespace}/{name}, fetching results"
    )
    _shutdown_sent.add(key)

    sb = StatusBuilder(patch, status)
    await handle_completion(body, namespace, jobset_name, job_id, status, sb)

    host = controller_dns_name(jobset_name, namespace)
    try:
        client = await get_or_create_progress_client(key)
        await client.send_shutdown(host)
    except Exception as e:
        logger.exception(f"Failed to send shutdown to {host}")
        kopf.event(
            body,
            type="Warning",
            reason="ShutdownSignalFailed",
            message=f"Failed to send shutdown to controller at {host}: {e}",
        )

    await close_progress_client(key)
