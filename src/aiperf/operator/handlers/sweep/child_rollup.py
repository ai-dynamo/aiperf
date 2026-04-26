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

TERMINAL_PHASES = frozenset({"Succeeded", "Failed", "Cancelled", "PartiallyFailed"})

__all__ = ["on_child_phase_transition"]


async def on_child_phase_transition(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
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
    if counts.get("total_terminal_phase"):
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
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes.client import k8s_client

    completed = failed = in_flight = 0
    async with k8s_client() as api:
        custom = k8s.CustomObjectsApi(api)
        resp = await custom.list_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfjobs",
            label_selector=f"aiperf.nvidia.com/sweep={sweep_name}",
        )
        for child in resp.get("items", []):
            refs = (child.get("metadata") or {}).get("ownerReferences") or []
            if not any(r.get("uid") == sweep_uid for r in refs):
                continue
            phase = (child.get("status") or {}).get("phase")
            if phase in {"Succeeded", "Completed"}:
                completed += 1
            elif phase in {"Failed", "Cancelled"}:
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
    from kubernetes_asyncio import client as k8s

    from aiperf.kubernetes.client import k8s_client

    async with k8s_client() as api:
        custom = k8s.CustomObjectsApi(api)
        # Force merge-patch content-type — kubernetes_asyncio defaults to
        # application/json-patch+json which expects a list of ops, not the dict
        # body we send here. The api_client kwarg name is `_content_type`.
        await custom.patch_namespaced_custom_object_status(
            group=group,
            version=version,
            plural=plural,
            namespace=namespace,
            name=name,
            body=body,
            _content_type="application/merge-patch+json",
        )
