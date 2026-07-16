# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark status reporting for Kubernetes mode (the native push model).

When running in K8s, the run pod (the ``aiperf controller`` frontend) reaches up to
its own parent AIPerfJob CR and PUSHES its state -- live progress into ``.status``
during the run, and a completion annotation at the end. The operator reacts to both
via kopf field handlers (watch stream) instead of polling the run; there is no
per-run progress service to poll.

This is the native successor to the mesh model, where the operator's
``monitor_progress`` timer HTTP-polled each run's ``/api/progress`` service and wrote
the numbers into the CR status itself. Here the run owns that write, using the same
in-cluster client + RBAC (aiperfjobs/status) it already used for completion.

Environment variables (set automatically by the JobSet manifest via
``jobset_helpers.build_cr_identity_env``):
    AIPERF_JOB_ID    - AIPerfJob CR name (= job_id)
    AIPERF_NAMESPACE - Namespace containing the CR
"""

from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_GROUP,
    AIPERF_PLURAL,
    AIPERF_VERSION,
)

logger = logging.getLogger(__name__)


def _cr_identity() -> tuple[str, str] | None:
    """The (job_id, namespace) of this pod's owning AIPerfJob, or None off-cluster."""
    job_id = os.environ.get("AIPERF_JOB_ID")
    namespace = os.environ.get("AIPERF_NAMESPACE")
    if not job_id or not namespace:
        logger.debug("Not in K8s mode (AIPERF_JOB_ID/AIPERF_NAMESPACE not set)")
        return None
    return job_id, namespace


async def report_benchmark_progress(
    *,
    phase: str,
    requests_completed: int,
    requests_total: int | None = None,
    requests_per_second: float | None = None,
    overall_phase: str | None = None,
) -> bool:
    """Patch the owning AIPerfJob CR ``.status`` with live progress (push model).

    Called periodically by the ``aiperf controller`` frontend from the runner's
    progress stream. Writes the ``.status.phases.<phase>`` shape the CLI watcher
    (``watch_pollers._populate_progress``) and the CRD printer columns already
    consume, plus the overall ``.status.phase`` when given, so a run pushes exactly
    what the operator's mesh poller used to write -- letting the operator drop the
    poll and react via ``@kopf.on.field(field="status.phases")``.

    Best-effort: a transient API error logs and returns False; the next tick retries.

    Args:
        phase: Phase name whose progress this is (e.g. ``"warmup"``, ``"profiling"``).
        requests_completed: Requests completed so far in ``phase``.
        requests_total: Phase request budget, when known (enables the percent).
        requests_per_second: Current offered/observed rate, when known.
        overall_phase: The overall ``.status.phase`` (e.g. ``"Profiling"``), if it
            should advance with this update.

    Returns:
        True if the status subresource was patched successfully.
    """
    identity = _cr_identity()
    if identity is None:
        return False
    job_id, namespace = identity

    phase_stats: dict[str, Any] = {"requestsCompleted": requests_completed}
    if requests_total is not None:
        phase_stats["requestsTotal"] = requests_total
        if requests_total > 0:
            phase_stats["requestsProgressPercent"] = round(
                100.0 * requests_completed / requests_total, 1
            )
    if requests_per_second is not None:
        phase_stats["requestsPerSecond"] = requests_per_second

    status: dict[str, Any] = {"phases": {phase: phase_stats}}
    if overall_phase:
        status["phase"] = overall_phase
    patch_body: dict[str, Any] = {"status": status}

    try:
        from kubernetes_asyncio import client

        from aiperf.kubernetes.client import k8s_client

        async with k8s_client() as api:
            await client.CustomObjectsApi(api).patch_namespaced_custom_object_status(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                plural=AIPERF_PLURAL,
                namespace=namespace,
                name=job_id,
                body=patch_body,
                _content_type="application/merge-patch+json",
            )
        return True
    except (ApiException, aiohttp.ClientError, OSError) as e:
        logger.warning(f"Failed to report benchmark progress: {e}")
        return False


async def report_benchmark_snapshot(snapshot: dict[str, Any]) -> bool:
    """Patch a full native-v2-level metric snapshot into the AIPerfJob ``.status.snapshot``.

    Kubernetes-native metric visibility: ``kubectl get aiperfjob -o json`` (and the
    operator's dashboard/API, which watch ``.status``) surface the run's metrics at
    the same fidelity as ``native-v2.json`` -- counters plus the TTFT / ITL / request-
    latency (and any other) distributions with min/max/percentiles -- without
    downloading the results PVC. Called by the ``aiperf controller`` frontend both
    live (from the running cross-cell aggregate the controller emits) and once at
    completion (the final ``native-v2.json``), so the snapshot converges to the
    committed report. The whole ``snapshot`` dict replaces ``.status.snapshot``
    (strategic keys under it are merge-patched); keep it well under etcd's ~1.5 MB
    object cap -- the native-v2 metric summary is a few tens of KB.

    Best-effort: a transient API error logs and returns False; the next tick retries.

    Args:
        snapshot: The native-v2-level metric snapshot to store under ``.status.snapshot``.

    Returns:
        True if the status subresource was patched successfully.
    """
    identity = _cr_identity()
    if identity is None:
        return False
    job_id, namespace = identity
    patch_body: dict[str, Any] = {"status": {"snapshot": snapshot}}

    try:
        from kubernetes_asyncio import client

        from aiperf.kubernetes.client import k8s_client

        async with k8s_client() as api:
            await client.CustomObjectsApi(api).patch_namespaced_custom_object_status(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                plural=AIPERF_PLURAL,
                namespace=namespace,
                name=job_id,
                body=patch_body,
                _content_type="application/merge-patch+json",
            )
        return True
    except (ApiException, aiohttp.ClientError, OSError) as e:
        logger.warning(f"Failed to report benchmark snapshot: {e}")
        return False


async def signal_benchmark_complete() -> bool:
    """Patch the AIPerfJob CR annotation to signal benchmark completion.

    Called by the controller pod after the benchmark finishes and results
    are exported.  The operator's ``on_benchmark_complete`` handler picks
    this up within seconds via kopf's watch mechanism.

    Returns:
        True if the annotation was patched successfully.
    """
    identity = _cr_identity()
    if identity is None:
        return False
    job_id, namespace = identity

    try:
        from kubernetes_asyncio import client

        from aiperf.kubernetes.client import k8s_client

        patch_body: dict[str, Any] = {
            "metadata": {
                "annotations": {
                    Annotations.BENCHMARK_COMPLETE: "true",
                }
            }
        }

        async with k8s_client() as api:
            await client.CustomObjectsApi(api).patch_namespaced_custom_object(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                plural=AIPERF_PLURAL,
                namespace=namespace,
                name=job_id,
                body=patch_body,
                _content_type="application/merge-patch+json",
            )

        logger.info(f"Signaled benchmark completion on AIPerfJob {namespace}/{job_id}")
        return True

    except (ApiException, aiohttp.ClientError, OSError) as e:
        logger.warning(f"Failed to signal benchmark completion: {e}")
        return False
