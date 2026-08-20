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

    # Claim timestamps latched at the moment this process won the
    # completion race, keyed by job_key. kopf passes a read-only
    # ``kopf.Body`` (a Mapping, not a dict) into handlers, so the claim
    # cannot always be written back into the caller's body snapshot;
    # this registry gives the same-tick claim-age reader
    # (``_completion_retry._claim_age_seconds``) a mutation-free source.
    claim_timestamps: dict[str, str] = {}


# Module-level aliases preserve the historical import surface used by
# operator.handlers.* modules (same dict/set objects as the class
# attributes, so writes through either name are visible to the other).
_progress_clients = _JobCacheState.progress_clients
_client_cache_lock = _JobCacheState.client_cache_lock
_warned_pod_restarts = _JobCacheState.warned_pod_restarts
_shutdown_sent = _JobCacheState.shutdown_sent
_cancellation_events = _JobCacheState.cancellation_events
_claim_timestamps = _JobCacheState.claim_timestamps


def request_cancellation(key: str) -> None:
    """Signal that any in-flight handler work for this job should abort.

    Called from on_delete. Long-running paths check
    ``is_cancellation_requested`` at each await boundary and exit early
    (skipping remaining retries, JobSet delete, status patches) so the
    CR deletion doesn't block on tens-of-seconds of fetch backoff.
    """
    event = _cancellation_events.get(key)
    if event is None:
        # Only unset flags are safe eviction candidates. A large sweep can
        # cancel thousands of children at once, and every SET flag remains a
        # live correctness signal until all in-flight handlers have observed
        # it. Let the cancellation registry exceed the client-cache bound
        # rather than revive work for a CR that is being deleted.
        while len(_cancellation_events) >= _MAX_CACHE_SIZE:
            evictable = next(
                (k for k, e in _cancellation_events.items() if not e.is_set()),
                None,
            )
            if evictable is None:
                break
            _cancellation_events.pop(evictable, None)
        event = asyncio.Event()
        _cancellation_events[key] = event
    event.set()


def is_cancellation_requested(key: str) -> bool:
    """Return True if cancellation was requested for this job key."""
    event = _cancellation_events.get(key)
    return event is not None and event.is_set()


def get_cancellation_event(key: str) -> asyncio.Event:
    """Return the shared cancellation event for an in-flight job handler.

    Awaiting this event lets long-running handler I/O stop as soon as
    ``request_cancellation`` runs instead of waiting for the current network
    timeout or retry delay to expire.
    """
    event = _cancellation_events.get(key)
    if event is None:
        event = asyncio.Event()
        _cancellation_events[key] = event
    return event


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


def job_key(namespace: str, job_id: str, uid: str | None = None) -> str:
    """Create a cache key scoped to one immutable AIPerfJob identity.

    CRs in different namespaces can share the same name, so cache keys
    and results directories must be namespace-scoped. Production operator
    paths also pass the CR UID so delayed callbacks from a deleted job cannot
    cancel, close, or claim state belonging to a same-name replacement. The
    UID remains optional for non-CR callers and compatibility with direct
    unit-test helpers.
    """
    base = f"{namespace}/{job_id}"
    return f"{base}@{uid}" if uid else base


async def get_or_create_progress_client(key: str) -> ProgressClient:
    """Get a cached ProgressClient for a job, creating one if needed.

    Serialized by _client_cache_lock to prevent concurrent interleaving
    between the None check and dict assignment (which includes an await).
    """
    async with _client_cache_lock:
        client = _progress_clients.get(key)
        if client is not None:
            # Mark as most-recently-used. Without this the cache evicted in
            # pure insertion order, so the longest-lived job -- the one most
            # likely still running -- was always the first closed, and a
            # mid-flight fetch on it then raised out of progress_client and
            # stamped the job Failed/ResultsFetchFailed with results ready.
            _progress_clients[key] = _progress_clients.pop(key)
            return client

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


def get_claim_timestamp(key: str) -> str | None:
    """Return the claim timestamp this process latched for ``key``, if any.

    Fallback source for the same-tick claim-age read when the caller's body
    snapshot could not be mutated (kopf hands handlers a read-only
    ``kopf.Body``). Returns None when this process never won the claim.
    """
    return _claim_timestamps.get(key)


def _latch_claim_timestamp(key: str, body: Any, timestamp: str) -> None:
    """Record a freshly-won claim timestamp for same-tick claim-age reads.

    Always records into the process-wide registry. Additionally writes the
    annotation back into ``body`` when the snapshot is mutable: kopf passes a
    read-only ``kopf.Body`` (a ``Mapping``, no ``setdefault``) in production,
    but plain dicts flow through tests and non-kopf call paths, and mutating
    those keeps the body self-describing for downstream readers.
    """
    while len(_claim_timestamps) >= _MAX_CACHE_SIZE:
        _claim_timestamps.pop(next(iter(_claim_timestamps)), None)
    _claim_timestamps[key] = timestamp

    if not isinstance(body, MutableMapping):
        return
    metadata = body.setdefault("metadata", {})
    if not isinstance(metadata, MutableMapping):
        return
    annotations = metadata.get("annotations")
    if annotations is None:
        annotations = {}
        metadata["annotations"] = annotations
    if isinstance(annotations, MutableMapping):
        annotations[Annotations.COMPLETION_CLAIMED] = timestamp


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
    expected_uid = (body.get("metadata") or {}).get("uid")
    key = job_key(
        namespace,
        name,
        str(expected_uid) if expected_uid is not None else None,
    )

    # In-process fast path: we already claimed this key in this process.
    if key in _shutdown_sent:
        return False

    annotations = (body.get("metadata") or {}).get("annotations") or {}
    if annotations.get(Annotations.STARTUP_FAILURE_CLAIMED):
        live_failure_claimed = await _read_live_startup_failure_claimed(
            namespace,
            name,
            expected_uid=str(expected_uid) if expected_uid is not None else None,
        )
        if live_failure_claimed is not False:
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
        live_claimed = await _read_live_completion_claimed(
            namespace,
            name,
            expected_uid=str(expected_uid) if expected_uid is not None else None,
        )
        if live_claimed is True:
            _shutdown_sent.add(key)
            from aiperf.operator.metrics import COMPLETION_CLAIM_RACES

            COMPLETION_CLAIM_RACES.inc()
            return False

    from aiperf.kubernetes.phase import format_timestamp

    timestamp = format_timestamp()
    patch_ops = _build_claim_patch_ops(body, timestamp)
    claimed = await _submit_claim_patch(
        namespace,
        name,
        patch_ops,
        expected_uid=str(expected_uid) if expected_uid is not None else None,
    )
    if claimed is True:
        _shutdown_sent.add(key)
        # Latch the claim so a same-tick handle_completion ->
        # maybe_raise_for_transient_fetch_failure can read the claim age (the
        # apiserver patch above is not reflected back into this snapshot).
        # Mirrors the in-process _shutdown_sent fast-path latch.
        _latch_claim_timestamp(key, body, timestamp)
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
    from aiperf.kubernetes.phase import format_timestamp

    # JSON Pointer RFC 6901: escape '/' as '~1' and '~' as '~0'.
    escaped_key = Annotations.COMPLETION_CLAIMED.replace("~", "~0").replace("/", "~1")
    if timestamp is None:
        timestamp = format_timestamp()
    metadata = body.get("metadata", {})
    patch_ops: list[dict[str, Any]] = []
    expected_uid = metadata.get("uid")
    if expected_uid is not None:
        patch_ops.append(
            {
                "op": "test",
                "path": "/metadata/uid",
                "value": str(expected_uid),
            }
        )
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
        return patch_ops + [
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
    return patch_ops + [
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
    from aiperf.transports.aiohttp_client import create_tcp_connector
    from aiperf.transports.http_defaults import AioHttpDefaults

    try:
        async with (
            aiohttp.ClientSession(
                connector=create_tcp_connector(),
                timeout=aiohttp.ClientTimeout(total=2.0),
                trust_env=AioHttpDefaults.TRUST_ENV,
            ) as session,
            session.post(url) as response,
        ):
            await response.read()
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        logger.debug("dashboard refresh skipped: %s", exc)


async def _submit_claim_patch(
    namespace: str,
    name: str,
    patch_ops: list[dict[str, Any]],
    *,
    expected_uid: str | None = None,
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
            live_claimed = await _read_live_completion_claimed(
                namespace, name, expected_uid=expected_uid
            )
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


async def _read_live_completion_claimed(
    namespace: str,
    name: str,
    *,
    expected_uid: str | None = None,
) -> bool | None:
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
    live_uid = (live_body.get("metadata") or {}).get("uid")
    if expected_uid is not None and live_uid != expected_uid:
        logger.info(
            "Completion claim callback for %s/%s uid=%s is stale; live uid=%s",
            namespace,
            name,
            expected_uid,
            live_uid,
        )
        return None
    return is_completion_claimed(live_body)


async def _read_live_startup_failure_claimed(
    namespace: str,
    name: str,
    *,
    expected_uid: str | None = None,
) -> bool | None:
    """Confirm a startup-failure claim against the live exact CR."""
    try:
        async with k8s_client() as api:
            live_body = await client.CustomObjectsApi(api).get_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                name=name,
            )
    except (ApiException, TimeoutError, aiohttp.ClientError, OSError) as exc:
        logger.warning(
            "Failed to verify startup-failure claim for %s/%s: %s",
            namespace,
            name,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - fail-safe: never complete across an unreadable failure claim
        logger.warning(
            "Unexpected error verifying startup-failure claim for %s/%s: %s",
            namespace,
            name,
            exc,
        )
        return None
    live_metadata = live_body.get("metadata") or {}
    if expected_uid is not None and live_metadata.get("uid") != expected_uid:
        return None
    live_annotations = live_metadata.get("annotations") or {}
    return bool(live_annotations.get(Annotations.STARTUP_FAILURE_CLAIMED))


def _reset_for_testing() -> None:
    """Clear all cached state. For use in tests only."""
    _progress_clients.clear()
    _warned_pod_restarts.clear()
    _shutdown_sent.clear()
    _cancellation_events.clear()
    _claim_timestamps.clear()
