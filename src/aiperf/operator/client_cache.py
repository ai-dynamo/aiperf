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

import kr8s

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.progress_client import ProgressClient

logger = logging.getLogger(__name__)

_MAX_CACHE_SIZE = 200

# Per-job ProgressClient cache keyed by namespace/job_id.
# Avoids creating a new aiohttp session every monitor tick.
_progress_clients: dict[str, ProgressClient] = {}
_client_cache_lock = asyncio.Lock()

# Tracks (pod_name, restart_count) pairs already warned about per job.
# Prevents emitting the same pod restart event every monitor tick.
_warned_pod_restarts: dict[str, set[tuple[str, int]]] = {}

# In-process fast-path cache of jobs where completion has already been
# claimed this operator process. Authoritative dedup lives on the CR as
# the ``Annotations.COMPLETION_CLAIMED`` annotation, which survives
# operator pod restart. This set just avoids re-doing the annotation
# check for claims made by this same process.
_shutdown_sent: set[str] = set()


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

    Returns True if the claim was newly acquired (caller should proceed
    with handle_completion). Returns False if the annotation is already
    present (another handler or a previous operator run claimed it) or
    if the claim attempt fails for any reason (fail-safe: don't double
    complete).

    Uses a JSON-patch with a ``test`` op so two concurrent handlers cannot
    both acquire the claim: only the first patch succeeds, the second
    gets a 422/409 and returns False.
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

    # Slow path: attempt to durably claim by patching the annotation.
    # Import lazily to avoid circular imports with aiperf.kubernetes.client.
    from aiperf.kubernetes.client import get_api
    from aiperf.kubernetes.kr8s_resources import AsyncAIPerfJob
    from aiperf.operator.status import format_timestamp

    # JSON Pointer RFC 6901: escape '/' as '~1' and '~' as '~0'.
    escaped_key = Annotations.COMPLETION_CLAIMED.replace("~", "~0").replace("/", "~1")
    timestamp = format_timestamp()

    current_annotations = body.get("metadata", {}).get("annotations")

    # If annotations dict exists we test against it; otherwise we add it.
    # Using a `test` op means a concurrent writer that also sets the
    # annotation will cause our patch to fail with 422, and we return
    # False (losing the race, which is the safe outcome).
    if current_annotations is None:
        patch_ops: list[dict[str, Any]] = [
            {"op": "test", "path": "/metadata/annotations", "value": None},
            {"op": "add", "path": "/metadata/annotations", "value": {}},
            {
                "op": "add",
                "path": f"/metadata/annotations/{escaped_key}",
                "value": timestamp,
            },
        ]
    else:
        patch_ops = [
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

    try:
        api = await get_api()
        obj = await AsyncAIPerfJob.get(name, namespace=namespace, api=api)
        await obj.patch(patch_ops, type="json")
    except kr8s.ServerError as e:
        status_code = e.response.status_code if e.response else 0
        if status_code in (409, 422):
            logger.debug(
                "Completion claim for %s/%s lost race (status %s), skipping",
                namespace,
                name,
                status_code,
            )
            _shutdown_sent.add(key)
            return False
        logger.warning(
            "Completion claim patch failed for %s/%s: %s (not claiming)",
            namespace,
            name,
            e,
        )
        return False
    except Exception as e:
        logger.warning(
            "Unexpected error claiming completion for %s/%s: %s (not claiming)",
            namespace,
            name,
            e,
        )
        return False

    _shutdown_sent.add(key)
    return True


def _reset_for_testing() -> None:
    """Clear all cached state. For use in tests only."""
    _progress_clients.clear()
    _warned_pod_restarts.clear()
    _shutdown_sent.clear()
