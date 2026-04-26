# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""@kopf.on.field handler on AIPerfJob.status.phase.

When a child has an AIPerfSweep ownerReference, recompute the parent's
rollup counts. Standalone AIPerfJobs are no-ops.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TERMINAL_PHASES = frozenset(
    {"Succeeded", "Failed", "Cancelled", "PartiallyFailed", "Completed"}
)
# Parent (AIPerfSweep) terminal phases the controller may write; the
# rollup must not clobber these once set.
PARENT_TERMINAL_PHASES = frozenset(
    {"Succeeded", "Failed", "Cancelled", "PartiallyFailed"}
)
# SSA field manager for rollup writes. The sweep-controller pod uses
# "aiperf-sweep-controller" — these two managers are intentionally
# distinct so the apiserver tracks per-field ownership in managedFields.
ROLLUP_FIELD_MANAGER = "aiperf-operator-rollup"

__all__ = ["on_child_phase_transition"]


async def on_child_phase_transition(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """For each AIPerfJob.status.phase change, if the child has an AIPerfSweep
    ownerReference, recompute the parent's rollup counts."""
    parent = _find_sweep_owner(body)
    if parent is None:
        return
    sweep_name, sweep_uid = parent

    counts = await _count_owned_children(namespace, sweep_uid, sweep_name)
    body_patch: dict[str, Any] = {
        "status": {
            "completedRuns": counts["completed"],
            "failedRuns": counts["failed"],
            "lastChildEvent": {
                "name": name,
                "phase": status.get("phase", "Unknown"),
            },
        }
    }
    # Only set phase=Aggregating when all children are terminal AND the parent
    # has not already reached its own terminal phase (the sweep-controller
    # owns terminal phase writes; clobbering them is the race documented in
    # the audit). The list `Aggregating` itself is non-terminal so it is safe
    # to set repeatedly.
    parent_phase = (await _read_parent_phase(namespace, sweep_name)) or ""
    if (
        counts.get("total_terminal_phase")
        and parent_phase not in PARENT_TERMINAL_PHASES
    ):
        body_patch["status"]["phase"] = counts["total_terminal_phase"]

    await _patch_parent_status(
        group="aiperf.nvidia.com",
        version="v1alpha1",
        plural="aiperfsweeps",
        name=sweep_name,
        namespace=namespace,
        body=body_patch,
    )


def _find_sweep_owner(child_body: dict[str, Any]) -> tuple[str, str] | None:
    refs = (child_body.get("metadata") or {}).get("ownerReferences") or []
    for ref in refs:
        if ref.get("kind") == "AIPerfSweep" and ref.get("name") and ref.get("uid"):
            return ref["name"], ref["uid"]
    return None


async def _count_owned_children(
    namespace: str, sweep_uid: str, sweep_name: str
) -> dict[str, Any]:
    """List children with the sweep label and count by terminal phase."""
    import aiohttp
    import kopf
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    completed = failed = in_flight = 0
    try:
        async with k8s_client() as api:
            custom = k8s.CustomObjectsApi(api)
            resp = await custom.list_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="aiperfjobs",
                label_selector=f"aiperf.nvidia.com/sweep={sweep_name}",
            )
    except ApiException as e:
        raise kopf.TemporaryError(
            f"apiserver rejected list ({e.status}): {e.reason}", delay=15
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during list: {e}", delay=15
        ) from e
    for child in resp.get("items", []):
        refs = (child.get("metadata") or {}).get("ownerReferences") or []
        if not any(r.get("uid") == sweep_uid for r in refs):
            continue
        phase = (child.get("status") or {}).get("phase")
        if phase in {"Succeeded", "Completed"}:
            completed += 1
        elif phase in {"Failed", "Cancelled", "PartiallyFailed"}:
            failed += 1
        else:
            in_flight += 1

    total = completed + failed + in_flight
    terminal_phase = None
    if in_flight == 0 and total > 0:
        # All children are terminal; sweep-controller will run aggregation.
        terminal_phase = "Aggregating"
    return {
        "completed": completed,
        "failed": failed,
        "in_flight": in_flight,
        "total_terminal_phase": terminal_phase,
    }


async def _patch_parent_status(
    *,
    group: str,
    version: str,
    plural: str,
    name: str,
    namespace: str,
    body: dict[str, Any],
) -> None:
    """Server-Side Apply of operator-owned rollup fields on AIPerfSweep.status.

    Uses ``application/apply-patch+yaml`` with field manager
    ``aiperf-operator-rollup``. The body's ``status`` content is the operator's
    declared ownership: ``completedRuns``, ``failedRuns``, ``lastChildEvent``,
    and conditionally ``phase``. The sweep-controller writes disjoint fields
    (``currentCell``, ``aggregation``, ``aggregateRef``) under its own field
    manager, so SSA tracks both without clobber. ``force=True`` reclaims fields
    after operator restarts.
    """
    import aiohttp
    import kopf
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    apply_body: dict[str, Any] = {
        "apiVersion": f"{group}/{version}",
        "kind": "AIPerfSweep",
        "metadata": {"name": name, "namespace": namespace},
        **body,
    }
    try:
        async with k8s_client() as api:
            custom = k8s.CustomObjectsApi(api)
            await custom.patch_namespaced_custom_object_status(
                group=group,
                version=version,
                plural=plural,
                namespace=namespace,
                name=name,
                body=apply_body,
                field_manager=ROLLUP_FIELD_MANAGER,
                force=True,
                _content_type="application/apply-patch+yaml",
            )
    except ApiException as e:
        if e.status == 404:
            # Parent CR was deleted between rollup and patch; not retryable.
            return
        raise kopf.TemporaryError(
            f"apiserver rejected status patch ({e.status}): {e.reason}", delay=15
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during status patch: {e}", delay=15
        ) from e


async def _read_parent_phase(namespace: str, name: str) -> str | None:
    """Return parent AIPerfSweep status.phase, or None if missing/unreadable."""
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            custom = k8s.CustomObjectsApi(api)
            cr = await custom.get_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="aiperfsweeps",
                name=name,
            )
    except Exception:  # noqa: BLE001 - best-effort read; worst case we re-set Aggregating once
        return None
    return ((cr.get("status") or {}).get("phase")) or None
