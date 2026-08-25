# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle handler logic: on_delete, on_cancel, on_benchmark_complete.

This module contains the business logic only — no kopf decorators.
Decorators live in ``aiperf.operator.main``.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import kopf

from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.phase import Phase
from aiperf.operator import events, runs_index
from aiperf.operator.client_cache import (
    close_progress_client,
    get_or_create_progress_client,
    is_cancellation_requested,
    job_key,
    request_cancellation,
    try_claim_completion,
)
from aiperf.operator.handlers._job_identity import (
    StaleAIPerfJobCallback,
    body_uid,
    current_aiperfjob_resource_version,
    delete_owned_aiperfjob_jobset,
)
from aiperf.operator.handlers.cleanup import on_aiperfjob_delete_index_cleanup
from aiperf.operator.handlers.completion import handle_completion
from aiperf.operator.status import StatusBuilder

logger = logging.getLogger(__name__)


def _acknowledge_generation(body: dict[str, Any], patch: kopf.Patch) -> None:
    """Stamp the generation consumed by a successful mutable-field handler."""
    generation = (body.get("metadata") or {}).get("generation")
    if generation is not None:
        patch.status["observedGeneration"] = int(generation)


async def acknowledge_timeout_update(
    *,
    body: dict[str, Any],
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Acknowledge a timeout update consumed by the monitor timer."""
    _acknowledge_generation(body, patch)


async def on_delete(
    name: str,
    namespace: str,
    status: dict[str, Any],
    uid: str | None = None,
    **_: Any,
) -> None:
    """Handle AIPerfJob CR deletion.

    Side effects:
        - Sets a sticky in-process cancellation flag for this job so any
          in-flight monitor/completion coroutines short-circuit at their
          next await boundary (avoids blocking delete on fetch backoff).
        - Closes the cached ProgressClient (releases aiohttp session).
        - Drops every ``runs_index`` row for this job so the index does
          not retain orphaned entries pointing at a deleted CR.
        - Relies on Kubernetes ownerReferences GC to reap the JobSet,
          ConfigMap, Role, and RoleBinding — this handler does NOT delete
          them directly.

    The cancellation flag is set BEFORE closing the client so concurrent
    observers see the flag before the client-cache entry disappears.
    """
    job_id = status.get("jobId", name)
    key = job_key(namespace, job_id, uid)
    # Request cancellation FIRST so any concurrent monitor/completion work
    # sees the flag before we free the client. close_progress_client also
    # clears the cancellation event, so the request must be made first.
    request_cancellation(key)
    await close_progress_client(key)
    await on_aiperfjob_delete_index_cleanup(namespace, name, status)
    logger.info(f"Deleting AIPerfJob {namespace}/{name}")


async def on_cancel(
    body: dict[str, Any],
    spec: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    expected_parent_uid: str | None = None,
    **_: Any,
) -> None:
    """Handle cancellation request via ``spec.cancel`` field.

    Fires on every ``spec.cancel`` update; no-ops unless ``spec.cancel`` is
    truthy and the CR is not already terminal.

    Side effects:
        - Sets the same sticky in-process cancellation flag as ``on_delete`` so
          in-flight completion/fetch paths cannot overwrite Cancelled.
        - Deletes the JobSet custom object; non-404 failures raise
          ``kopf.TemporaryError`` so the cancel field watcher retries.
        - Closes the cached ProgressClient for this job after deletion succeeds.
        - Patches ``status.phase`` to ``Cancelled`` and sets completion time.
        - Emits a ``Cancelled`` kopf event on the CR.
    """
    parent_uid = expected_parent_uid or body_uid(body)
    try:
        # Read for its identity assertion only: it raises on a replaced UID.
        await current_aiperfjob_resource_version(namespace, name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale cancel callback: %s", exc)
        return

    sb = StatusBuilder(patch, status)
    if not spec.get("cancel"):
        _acknowledge_generation(body, patch)
        return

    current_phase = status.get("phase", Phase.PENDING)
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        _acknowledge_generation(body, patch)
        return  # Already terminal

    job_id = status.get("jobId", name)
    key = job_key(namespace, job_id, parent_uid)
    jobset_name = status.get("jobSetName")

    logger.info(f"Cancelling AIPerfJob {namespace}/{name}")
    request_cancellation(key)

    if jobset_name:
        deleted = await delete_owned_aiperfjob_jobset(
            namespace,
            jobset_name,
            parent_name=name,
            parent_uid=parent_uid,
            context="cancel",
        )
        if deleted:
            logger.info(f"Deleted JobSet {jobset_name}")
        else:
            return

    try:
        await current_aiperfjob_resource_version(namespace, name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Discarding stale cancel publication: %s", exc)
        await close_progress_client(key)
        return

    await close_progress_client(key)
    sb.set_phase(Phase.CANCELLED).set_completion_time()
    generation = body.get("metadata", {}).get("generation")
    if generation is not None:
        sb.set_observed_generation(int(generation))
    # Do NOT fence this patch with metadata.resourceVersion (see the same note
    # in ``completion._publish_completion_after_jobset_delete``). The fence makes
    # kopf's single merge PATCH (metadata+status) 409 on any concurrent CR write
    # — a monitor tick or a controller heartbeat annotation is enough — and the
    # status update is then silently dropped. Cancellation cannot recover from
    # that: the JobSet is already deleted above and ``request_cancellation`` is
    # sticky, so every later monitor tick short-circuits and the CR is stranded
    # in its pre-cancel phase forever. Stale-write protection here comes from the
    # UID-fenced ``current_aiperfjob_resource_version`` re-read just above.
    sb.finalize()
    events.cancelled(body, job_id)


async def on_benchmark_complete(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    expected_parent_uid: str | None = None,
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

    parent_uid = expected_parent_uid or body_uid(body)
    try:
        await current_aiperfjob_resource_version(namespace, name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale benchmark-complete callback: %s", exc)
        return

    key = job_key(namespace, job_id, parent_uid)
    # Check cancellation BEFORE claiming. try_claim_completion stamps a
    # durable annotation on the CR; if on_delete fires before the claim
    # is consumed, the CR is GC'd via ownerRef finalizers before the
    # monitor's _recover_orphaned_completion_claim path can reach it,
    # and the claim is silently lost. Pairs with the same check at the
    # top of monitor.py's _reconcile_aiperfjob_event entry point.
    if is_cancellation_requested(key):
        logger.info(
            f"Cancellation requested for {namespace}/{name}, "
            "skipping benchmark-complete handling"
        )
        return

    if not await try_claim_completion(namespace, name, body):
        return

    logger.info(
        f"Benchmark completion signal received for {namespace}/{name}, fetching results"
    )

    sb = StatusBuilder(patch, status)
    await handle_completion(
        body,
        namespace,
        jobset_name,
        job_id,
        status=status,
        sb=sb,
        expected_parent_uid=parent_uid,
    )
    # observedGeneration is a success-path-only stamp: stamp ONLY when
    # handle_completion finalized the CR as COMPLETED. Absence of a phase
    # (handle_completion short-circuited on a mid-completion cancellation
    # before copying its staged patch into sb, so get_phase() is None) or a
    # FAILED/CANCELLED phase must not signal spec acceptance. Mirrors
    # monitor.monitor_progress.
    if sb.get_phase() == str(Phase.COMPLETED):
        generation = body.get("metadata", {}).get("generation")
        if generation is not None:
            sb.set_observed_generation(int(generation))

    await _shutdown_after_completion(
        body=body,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
        key=key,
        parent_uid=parent_uid,
    )


async def _shutdown_after_completion(
    *,
    body: dict[str, Any],
    namespace: str,
    name: str,
    jobset_name: str,
    key: str,
    parent_uid: str | None,
) -> None:
    """Shut down only the controller belonging to the completed CR identity."""
    if is_cancellation_requested(key):
        logger.info(
            "Cancellation requested for %s/%s after completion handling; "
            "skipping controller shutdown",
            namespace,
            name,
        )
        await close_progress_client(key)
        return
    try:
        await current_aiperfjob_resource_version(namespace, name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale controller shutdown: %s", exc)
        await close_progress_client(key)
        return

    host = controller_dns_name(jobset_name, namespace)
    try:
        progress_client = await get_or_create_progress_client(key)
        await progress_client.send_shutdown(host)
    except (TimeoutError, aiohttp.ClientError, OSError) as e:
        logger.exception(f"Failed to send shutdown to {host}")
        kopf.event(
            body,
            type="Warning",
            reason="ShutdownSignalFailed",
            message=f"Failed to send shutdown to controller at {host}: {e}",
        )

    await close_progress_client(key)


async def record_phase_transition(
    namespace: str,
    name: str,
    status: dict[str, Any],
) -> None:
    """Mirror an AIPerfJob phase transition into ``runs_index``.

    Wired from ``main.on_aiperfjob_phase_transition`` (the kopf
    ``status.phase`` field watcher). Best-effort: any failure logs and
    swallows so the index path never blocks a kopf handler tick.

    The index needs ``(namespace, job_id, epoch)`` as the row key.
    ``job_id`` falls back to ``name`` when the controller has not yet
    written ``status.jobId``; ``runEpoch`` is the canonical run-key on the
    CR (set by ``handle_completion`` once results land). Phases observed
    before ``runEpoch`` is set (e.g. ``Pending`` at create) are skipped —
    the create handler already wrote a row with the correct epoch via
    ``upsert_run_created``.
    """
    phase = status.get("phase")
    if not phase:
        return
    job_id = status.get("jobId", name)
    run_epoch = status.get("runEpoch")
    if run_epoch is None:
        return
    epoch = str(run_epoch)
    try:
        await runs_index.upsert_run_phase(namespace, job_id, epoch, phase=phase)
    except Exception as exc:  # noqa: BLE001 - index path must never break the handler
        logger.warning("runs_index.upsert_run_phase failed: %s", exc)
