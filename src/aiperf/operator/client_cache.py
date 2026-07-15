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
from collections.abc import MutableMapping
from typing import Any

import aiohttp
import httpx
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
)
from aiperf.kubernetes.environment import K8sEnvironment
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
        # SET flags deliberately survive _close_unlocked so in-flight
        # observers still short-circuit after the client cache entry is
        # freed (see _close_unlocked). For uniquely-named cancelled jobs
        # (e.g. large sweeps whose <sweep>-vNN-tN children are deleted)
        # that means one retained entry per job for the operator process
        # lifetime, since clear_cancellation only fires on a same-name CR
        # reuse. Bound the dict the same way _progress_clients is bounded:
        # evict the oldest (insertion-order) entries first. The new entry
        # is always the most recent, so an in-flight job's flag is never
        # the one dropped; a stale SET flag for a long-deleted job is
        # harmless to forget.
        while len(_cancellation_events) >= _MAX_CACHE_SIZE:
            oldest_key = next(iter(_cancellation_events))
            _cancellation_events.pop(oldest_key, None)
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
            # Target the results-sidecar port, not the retired mesh API service
            # (9090). Under the native cellular model the controller pod runs no
            # api service — the only HTTP the operator makes to the run pod is
            # results fetch, served by the results-sidecar container on
            # PORTS.RESULTS_SIDECAR (9091) with the same /api/results/* paths the
            # ProgressClient uses. Progress/metrics polling is retired and the
            # (vestigial) shutdown call best-effort no-ops against the exited
            # controller, so a single results-scoped client is correct here.
            client = ProgressClient(port=K8sEnvironment.PORTS.RESULTS_SIDECAR)
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
    # Prune the cancellation event ONLY when it is unset. A SET flag must
    # survive the close so concurrent observers (the fetch-retry loop, for
    # instance, yields between the close and the next iteration) reliably
    # short-circuit even after the client-cache entry is freed. An UNSET
    # entry (every normal completion/cleanup path through here) carries no
    # signal, so dropping it bounds _cancellation_events to currently-
    # cancelling jobs instead of every job key ever seen.
    cancel_event = _cancellation_events.get(key)
    if cancel_event is not None and not cancel_event.is_set():
        _cancellation_events.pop(key, None)


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
    gets a 422/409 and returns False. When the body snapshot already shows
    the claim annotation, the live CR is re-read (the snapshot is
    user-writable and not trusted) and a genuine live claim is treated as a
    lost race — without re-submitting an overwriting claim patch.

    Args:
        namespace: Namespace of the AIPerfJob CR.
        name: Name of the AIPerfJob CR.
        body: The CR body. If its snapshot carries the claim annotation, the
            live CR is re-read to confirm before losing the race; a forged or
            stale snapshot annotation does not by itself suppress completion.

    Returns:
        True iff this call newly won the race and the caller should
        proceed with ``handle_completion``. False if a genuine prior claim
        exists (another handler or a previous operator run claimed it) or if
        the claim attempt fails for any reason (fail-safe: don't double
        complete).

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

    # The CR body's COMPLETION_CLAIMED annotation is user-writable, so a forged
    # value must not be trusted as a skip (that would let an attacker suppress
    # completion). But a genuine prior claim MUST make this call lose the race:
    # handle_completion is not idempotent, so re-running it double-completes
    # (re-fetch results, re-emit events, re-delete the JobSet). When the
    # snapshot shows a claim, resolve both concerns by verifying against LIVE
    # apiserver state (not the snapshot): a real live claim is a decisive lost
    # race; a stale/forged snapshot falls through to the atomic claim below,
    # whose ``test``-op still guards the concurrent first-claim race.
    if is_completion_claimed(body):
        live_claimed = await _read_live_completion_claimed(namespace, name)
        if live_claimed is True:
            _shutdown_sent.add(key)
            from aiperf.operator.metrics import COMPLETION_CLAIM_RACES

            COMPLETION_CLAIM_RACES.inc()
            return False

    from aiperf.operator.status import format_timestamp

    timestamp = format_timestamp()
    patch_ops = _build_claim_patch_ops(body, timestamp)
    claimed = await _submit_claim_patch(namespace, name, patch_ops)
    if claimed is True:
        _shutdown_sent.add(key)
        # Latch the claim into the local snapshot so a same-tick
        # handle_completion -> maybe_raise_for_transient_fetch_failure can read
        # the claim age (the apiserver patch above is not reflected back into
        # this dict). Mirrors the in-process _shutdown_sent fast-path latch.
        #
        # kopf hands handlers a read-only ``Body`` mapping (no ``setdefault``);
        # only plain-dict callers (tests, synthesized bodies) are mutable. When
        # body is immutable, skip the latch: the claim is already durably
        # persisted to the apiserver above, so the transient-fetch retry gate
        # reads it from the apiserver-fresh body on the next monitor tick
        # instead of same-tick. Correctness never depends on the latch.
        if isinstance(body, MutableMapping):
            metadata = body.setdefault("metadata", {})
            annotations = metadata.get("annotations")
            if annotations is None:
                annotations = {}
                metadata["annotations"] = annotations
            annotations[Annotations.COMPLETION_CLAIMED] = timestamp
        await _post_dashboard_refresh()
        return True
    if claimed is False:
        # Lost the race on a 409/422: remember so subsequent ticks skip
        # the API call. ``None`` means an unexpected error — don't cache.
        _shutdown_sent.add(key)
        from aiperf.operator.metrics import COMPLETION_CLAIM_RACES

        COMPLETION_CLAIM_RACES.inc()
    return False


def _build_claim_patch_ops(
    body: dict[str, Any], timestamp: str | None = None
) -> list[dict[str, Any]]:
    """Build the JSON-patch ops that atomically claim the completion annotation.

    Using a ``test`` op means a concurrent writer that also sets the
    annotation will cause our patch to fail with 422, and we return
    False (losing the race, which is the safe outcome).

    ``timestamp`` lets the caller reuse the exact value it later latches into
    the local body snapshot, so the same-tick transient-fetch retry gate sees
    the claim age. When ``None`` (the default) the value is generated here.
    """
    from aiperf.operator.status import format_timestamp

    # JSON Pointer RFC 6901: escape '/' as '~1' and '~' as '~0'.
    escaped_key = Annotations.COMPLETION_CLAIMED.replace("~", "~0").replace("/", "~1")
    if timestamp is None:
        timestamp = format_timestamp()
    metadata = body.get("metadata", {})
    current_annotations = metadata.get("annotations")

    if current_annotations is None:
        precondition_path = "/metadata"
        # Snapshot the metadata dict so a later mutation of body["metadata"]
        # (e.g. the caller latching the claim annotation after a successful
        # patch) cannot retroactively alter this test-op precondition.
        precondition_value: Any = dict(metadata)
        if metadata.get("resourceVersion") is not None:
            precondition_path = "/metadata/resourceVersion"
            precondition_value = metadata["resourceVersion"]
        return [
            {
                "op": "test",
                "path": precondition_path,
                "value": precondition_value,
            },
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
            "path": "/metadata/annotations",
            # Snapshot so a later body["metadata"]["annotations"] mutation
            # (claim-latch by the caller) cannot alter this precondition.
            "value": dict(current_annotations),
        },
        {
            "op": "add",
            "path": f"/metadata/annotations/{escaped_key}",
            "value": timestamp,
        },
    ]


async def _post_dashboard_refresh() -> None:
    """Fire-and-forget POST to the dashboard sidecar's /admin/refresh.

    Called after a successful completion claim so the Plotly Dash view
    picks up the new run on the PVC. Best-effort: failures (sidecar off,
    dashboard disabled, port unreachable) are logged at debug and
    swallowed -- refresh is not load-bearing.
    """
    from aiperf.operator.environment import OperatorEnvironment

    port = OperatorEnvironment.DASHBOARD.PORT
    if port <= 0:
        return
    url = f"http://localhost:{port}/admin/refresh"
    try:
        async with httpx.AsyncClient(timeout=2.0) as http_client:
            await http_client.post(url)
    except (httpx.HTTPError, OSError) as exc:
        logger.debug("dashboard refresh skipped: %s", exc)


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
        if status_code == 409:
            live_claimed = await _read_live_completion_claimed(namespace, name)
            if live_claimed is True:
                logger.debug(
                    "Completion claim for %s/%s lost race (status %s), skipping",
                    namespace,
                    name,
                    status_code,
                )
                return False
            logger.warning(
                "Completion claim patch conflicted for %s/%s without a live claim; "
                "not caching as a lost race so a later tick can retry: %s",
                namespace,
                name,
                e,
            )
            return None
        if status_code == 422:
            logger.warning(
                "Completion claim patch was rejected for %s/%s with status 422; "
                "not caching as a lost race so a later tick can retry: %s",
                namespace,
                name,
                e,
            )
            return None
        logger.warning(
            "Completion claim patch failed for %s/%s: %s (not claiming)",
            namespace,
            name,
            e,
        )
        return None
    except (TimeoutError, aiohttp.ClientError, OSError) as e:
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


async def _read_live_completion_claimed(namespace: str, name: str) -> bool | None:
    """Re-read the CR after a claim conflict and report whether it is claimed."""
    try:
        async with k8s_client() as api:
            live_body = await client.CustomObjectsApi(api).get_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                name=name,
            )
    except ApiException as e:
        logger.warning(
            "Failed to re-read completion claim for %s/%s after conflict: %s",
            namespace,
            name,
            e,
        )
        return None
    except (TimeoutError, aiohttp.ClientError, OSError) as e:
        logger.warning(
            "Unexpected error re-reading completion claim for %s/%s after conflict: %s",
            namespace,
            name,
            e,
        )
        return None
    except Exception as e:  # noqa: BLE001 - fail-safe: do not cache a race on unreadable live state
        logger.warning(
            "Unexpected error re-reading completion claim for %s/%s after conflict: %s",
            namespace,
            name,
            e,
        )
        return None
    return is_completion_claimed(live_body)


def _reset_for_testing() -> None:
    """Clear all cached state. For use in tests only."""
    _progress_clients.clear()
    _warned_pod_restarts.clear()
    _shutdown_sent.clear()
    _cancellation_events.clear()
