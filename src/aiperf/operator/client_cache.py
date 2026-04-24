# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-job ProgressClient cache with LRU eviction.

Serializes concurrent access with an asyncio.Lock to prevent
interleaving between the None-check and dict assignment (which
contains an ``await``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
)
from aiperf.operator.progress_client import ProgressClient

logger = logging.getLogger(__name__)

_MAX_CACHE_SIZE = 200


class _JobCacheState:
    """Process-wide kopf handler caches for AIPerfJob reconcile.

    Holds the per-job mutable state that every reconcile tick reads:
    ProgressClient sessions, pod-restart dedup, completion-claim fast
    path, and cancellation flags. Encapsulated on a class so the state
    is discoverable in one place and the module-level names below are
    simple aliases to the class attributes (same dict/set objects).
    """

    # Per-job ProgressClient cache keyed by namespace/job_id. Avoids
    # creating a new aiohttp session every monitor tick.
    progress_clients: dict[str, ProgressClient] = {}
    client_cache_lock: asyncio.Lock = asyncio.Lock()

    # Tracks (pod_name, restart_count) pairs already warned about per
    # job. Prevents emitting the same pod restart event every tick.
    warned_pod_restarts: dict[str, set[tuple[str, int]]] = {}

    # In-process fast-path cache of jobs where completion has already
    # been claimed this operator process. Authoritative dedup lives on
    # the CR as the ``Annotations.COMPLETION_CLAIMED`` annotation,
    # which survives operator pod restart. This set just avoids
    # re-doing the annotation check for claims made by this process.
    shutdown_sent: set[str] = set()

    # Per-job cancellation events set by on_delete. Long-running
    # handler paths check ``is_cancellation_requested`` at await
    # boundaries and short-circuit so CR deletion doesn't have to
    # wait for fetch backoff + JobSet delete.
    cancellation_events: dict[str, asyncio.Event] = {}


# Module-level aliases preserve the historical import surface used by
# operator.handlers.* modules (same dict/set objects as the class
# attributes, so writes through either name are visible to the other).
_progress_clients = _JobCacheState.progress_clients
_client_cache_lock = _JobCacheState.client_cache_lock
_warned_pod_restarts = _JobCacheState.warned_pod_restarts
_shutdown_sent = _JobCacheState.shutdown_sent
_cancellation_events = _JobCacheState.cancellation_events


def request_cancellation(key: str) -> None:
    """Signal that any in-flight handler work for this job should abort.

    Called from on_delete. Long-running paths check
    ``is_cancellation_requested`` at each await boundary and exit early
    (skipping remaining retries, JobSet delete, status patches) so the
    CR deletion doesn't block on tens-of-seconds of fetch backoff.
    """
    event = _cancellation_events.get(key)
    if event is None:
        event = asyncio.Event()
        _cancellation_events[key] = event
    event.set()


def is_cancellation_requested(key: str) -> bool:
    """Return True if cancellation was requested for this job key."""
    event = _cancellation_events.get(key)
    return event is not None and event.is_set()


def clear_cancellation(key: str) -> None:
    """Drop the cancellation flag for a job key.

    `request_cancellation` is sticky by design: once set, a flag stays set
    for the lifetime of the operator process so that in-flight observers
    (fetch-retry, etc.) reliably short-circuit even after the client-cache
    entry is freed. But when a new CR is created with the same
    namespace/name as a previously-deleted one, the key collides and the
    new CR inherits the old flag — every monitor tick skips, the CR stays
    Pending forever, results are never downloaded. Call this from
    `on_create` to give the new CR a clean slate.
    """
    _cancellation_events.pop(key, None)


def job_key(namespace: str, job_id: str) -> str:
    """Create a unique cache key scoped to namespace.

    CRs in different namespaces can share the same name, so cache keys
    and results directories must be namespace-scoped.
    """
    return f"{namespace}/{job_id}"


async def get_or_create_progress_client(key: str) -> ProgressClient:
    """Get a cached ProgressClient for a job, creating one if needed.

    Serialized by _client_cache_lock to prevent concurrent interleaving
    between the None check and dict assignment (which includes an await).
    """
    async with _client_cache_lock:
        client = _progress_clients.get(key)
        if client is None:
            while len(_progress_clients) >= _MAX_CACHE_SIZE:
                oldest_key = next(iter(_progress_clients))
                await _close_unlocked(oldest_key)
            client = ProgressClient()
            await client.__aenter__()
            _progress_clients[key] = client
        return client


async def close_progress_client(key: str) -> None:
    """Close and remove a cached ProgressClient and dedup state for a job."""
    async with _client_cache_lock:
        await _close_unlocked(key)


async def _close_unlocked(key: str) -> None:
    """Close a cached ProgressClient without acquiring the lock (caller holds it)."""
    client = _progress_clients.pop(key, None)
    if client is not None:
        await client.__aexit__(None, None, None)
    _warned_pod_restarts.pop(key, None)
    _shutdown_sent.discard(key)
    # Intentionally DO NOT clear _cancellation_events here: observers may
    # still need to see the cancel flag after the client is freed (the
    # fetch-retry loop, for instance, yields between the close and the
    # next iteration). Once set, the flag stays set until
    # _reset_for_testing is called or the process exits.


def is_completion_claimed(body: dict[str, Any]) -> bool:
    """Return True if the CR body already carries the completion-claimed annotation."""
    annotations = body.get("metadata", {}).get("annotations") or {}
    return bool(annotations.get(Annotations.COMPLETION_CLAIMED))


async def try_claim_completion(
    namespace: str,
    name: str,
    body: dict[str, Any],
) -> bool:
    """Try to claim the completion branch durably via a CR annotation.

    Uses a JSON-patch with a ``test`` op so two concurrent handlers cannot
    both acquire the claim: only the first patch succeeds, the second
    gets a 422/409 and returns False.

    Args:
        namespace: Namespace of the AIPerfJob CR.
        name: Name of the AIPerfJob CR.
        body: The CR body (checked for an existing claim annotation to
            avoid an unnecessary API round-trip on the slow path).

    Returns:
        True iff this call newly won the race and the caller should
        proceed with ``handle_completion``. False if the annotation was
        already present (another handler or a previous operator run
        claimed it) or if the claim attempt fails for any reason
        (fail-safe: don't double complete).

    Raises:
        No exceptions escape — unexpected errors are logged and return
        False. The ``_shutdown_sent`` in-process set is updated on lost
        races so subsequent ticks skip the API call entirely.

    Example:
        >>> if await try_claim_completion(namespace, name, body):
        ...     await handle_completion(
        ...         body, namespace, jobset_name, job_id, status, sb
        ...     )
    """
    key = job_key(namespace, name)

    # In-process fast path: we already claimed this key in this process.
    if key in _shutdown_sent:
        return False

    # Annotation fast path: a previous operator run (or the annotation
    # handler itself) already claimed this job.
    if is_completion_claimed(body):
        _shutdown_sent.add(key)
        return False

    patch_ops = _build_claim_patch_ops(body)
    claimed = await _submit_claim_patch(namespace, name, patch_ops)
    if claimed is True:
        _shutdown_sent.add(key)
        return True
    if claimed is False:
        # Lost the race on a 409/422: remember so subsequent ticks skip
        # the API call. ``None`` means an unexpected error — don't cache.
        _shutdown_sent.add(key)
    return False


def _build_claim_patch_ops(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the JSON-patch ops that atomically claim the completion annotation.

    Using a ``test`` op means a concurrent writer that also sets the
    annotation will cause our patch to fail with 422, and we return
    False (losing the race, which is the safe outcome).
    """
    from aiperf.operator.status import format_timestamp

    # JSON Pointer RFC 6901: escape '/' as '~1' and '~' as '~0'.
    escaped_key = Annotations.COMPLETION_CLAIMED.replace("~", "~0").replace("/", "~1")
    timestamp = format_timestamp()
    current_annotations = body.get("metadata", {}).get("annotations")

    if current_annotations is None:
        return [
            {"op": "test", "path": "/metadata/annotations", "value": None},
            {"op": "add", "path": "/metadata/annotations", "value": {}},
            {
                "op": "add",
                "path": f"/metadata/annotations/{escaped_key}",
                "value": timestamp,
            },
        ]
    return [
        {
            "op": "test",
            "path": f"/metadata/annotations/{escaped_key}",
            "value": None,
        },
        {
            "op": "add",
            "path": f"/metadata/annotations/{escaped_key}",
            "value": timestamp,
        },
    ]


async def _submit_claim_patch(
    namespace: str,
    name: str,
    patch_ops: list[dict[str, Any]],
) -> bool | None:
    """Apply the claim JSON-patch; return True on win, False on lost race, None on error."""
    try:
        async with k8s_client() as api:
            await client.CustomObjectsApi(api).patch_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                name=name,
                body=patch_ops,
                _content_type="application/json-patch+json",
            )
    except ApiException as e:
        status_code = e.status or 0
        if status_code in (409, 422):
            logger.debug(
                "Completion claim for %s/%s lost race (status %s), skipping",
                namespace,
                name,
                status_code,
            )
            return False
        logger.warning(
            "Completion claim patch failed for %s/%s: %s (not claiming)",
            namespace,
            name,
            e,
        )
        return None
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.warning(
            "Unexpected error claiming completion for %s/%s: %s (not claiming)",
            namespace,
            name,
            e,
        )
        return None
    except Exception as e:  # noqa: BLE001 - fail-safe: any error reclaiming must NOT raise into kopf; we prefer 'not claimed' over 'double-claimed'
        logger.warning(
            "Unexpected error claiming completion for %s/%s: %s (not claiming)",
            namespace,
            name,
            e,
        )
        return None
    return True


def _reset_for_testing() -> None:
    """Clear all cached state. For use in tests only."""
    _progress_clients.clear()
    _warned_pod_restarts.clear()
    _shutdown_sent.clear()
    _cancellation_events.clear()
