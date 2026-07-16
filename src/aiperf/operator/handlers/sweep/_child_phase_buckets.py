# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase classification + child listing helpers for the sweep rollup handler.

Carved out of ``child_rollup.py`` to keep that module under the 500-line
ergonomics ceiling. ``child_rollup`` retains the kopf-decorated entry
point and the apiserver-write helpers; this module owns the read-side
phase-bucketing and "currently running child" selection logic, plus the
``_api_or_new`` ApiClient context-reuse wrapper shared across helpers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from kubernetes_asyncio.client import ApiClient

__all__ = [
    "_PHASE_BUCKETS",
    "_api_or_new",
    "_count_owned_children",
    "_find_current_child",
    "_tally_children",
]


@asynccontextmanager
async def _api_or_new(
    api: ApiClient | None,
) -> AsyncIterator[ApiClient]:
    """Yield ``api`` if non-None, else open a fresh ``k8s_client()`` context.

    Lets one ``on_child_phase_transition`` tick share a single ApiClient across
    its 3-4 helper calls (list, get, patch, conditional patch) instead of
    opening a separate TCP/TLS handshake per helper. Helpers retain their
    ``api=None`` default for unit tests that call them in isolation.
    """
    from aiperf.kubernetes.client import k8s_client

    if api is not None:
        yield api
        return
    async with k8s_client() as fresh:
        yield fresh


async def _count_owned_children(
    namespace: str,
    sweep_uid: str,
    sweep_name: str,
    *,
    run_epoch: str | None = None,
    api: ApiClient | None = None,
) -> dict[str, Any]:
    """List children with the sweep label and count by terminal phase.

    When called from the kopf entry point, ``api`` is the shared client for
    this tick. Standalone test callers can pass ``api=None`` and a fresh
    client is opened transparently.

    ``run_epoch`` (when provided) is added to the label selector so we
    count only children from a single epoch — without this filter, stale
    children from prior re-applies of the sweep get counted and the UI
    reports e.g. ``completedRuns=5`` against ``totalVariations=3``.
    """
    import aiohttp
    import kopf
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    selector = f"aiperf.nvidia.com/sweep={sweep_name}"
    if run_epoch:
        selector += f",aiperf.nvidia.com/sweep-run-epoch={run_epoch}"
    try:
        async with _api_or_new(api) as client:
            custom = k8s.CustomObjectsApi(client)
            resp = await custom.list_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="aiperfjobs",
                label_selector=selector,
            )
    except ApiException as e:
        raise kopf.TemporaryError(
            f"apiserver rejected list ({e.status}): {e.reason}", delay=15
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during list: {e}", delay=15
        ) from e

    return _tally_children(resp.get("items", []), sweep_uid)


# Phase strings are case-insensitive on the wire; canonical AIPerfJob
# terminal phases are PascalCase (Succeeded / Failed / ...) but children
# mid-write may surface lowercase forms from SystemState. Lowercasing
# the dispatch key tolerates both without a brittle exact-match table.
_PHASE_BUCKETS: dict[str, str] = {
    "succeeded": "completed",
    "completed": "completed",
    "failed": "failed",
    "partiallyfailed": "failed",
    "cancelled": "cancelled",
    "profiling": "running",
    "processing": "running",
    "running": "running",
}


def _tally_children(items: list[Any], sweep_uid: str) -> dict[str, Any]:
    """Bucket owned children by phase. Pending/missing/unknown → pending."""
    counts = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }
    owned: list[dict[str, Any]] = []
    for child in items:
        refs = (child.get("metadata") or {}).get("ownerReferences") or []
        if not any(r.get("uid") == sweep_uid for r in refs):
            continue
        owned.append(child)
        phase = (child.get("status") or {}).get("phase")
        bucket = _PHASE_BUCKETS.get((phase or "").lower(), "pending")
        counts[bucket] += 1

    in_flight = counts["pending"] + counts["running"]
    total = in_flight + counts["completed"] + counts["failed"] + counts["cancelled"]
    terminal_phase = "Aggregating" if (in_flight == 0 and total > 0) else None
    counts["in_flight"] = in_flight
    counts["total_terminal_phase"] = terminal_phase  # type: ignore[assignment]
    counts["owned_children"] = owned  # type: ignore[assignment]
    return counts


def _find_current_child(children: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the AIPerfJob to surface in AIPerfSweep.status.currentChildRef.

    Priority: running > pending (lowest variation-index) > None when all
    terminal. Sweeps run roughly in variation-index order, so the lowest
    in-flight index is the active head.

    Children missing the ``aiperf.nvidia.com/variation-index`` label sort
    to the back via the ``"9999"`` fallback — Task 11 will fix the
    sweep-controller's k8s_executor to always stamp the label.
    """

    def _idx(child: dict[str, Any]) -> int:
        labels = (child.get("metadata") or {}).get("labels") or {}
        try:
            return int(labels.get("aiperf.nvidia.com/variation-index", "9999"))
        except (TypeError, ValueError):
            return 9999

    running = [
        c
        for c in children
        if ((c.get("status") or {}).get("phase") or "").lower()
        in {"profiling", "processing", "running"}
    ]
    if running:
        running.sort(key=_idx)
        return running[0]
    pending = [
        c
        for c in children
        if ((c.get("status") or {}).get("phase") or "").lower() == "pending"
    ]
    if pending:
        pending.sort(key=_idx)
        return pending[0]
    return None
