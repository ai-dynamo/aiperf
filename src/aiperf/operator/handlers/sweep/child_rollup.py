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
# Field-manager metadata on the merge-patch. Distinct from the
# sweep-controller's "aiperf-sweep-controller" so kubectl can tell which
# writer last touched each status field. Merge-patch does not enforce
# field ownership — the disjoint-top-level-field invariant between
# operator and controller writers does that.
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
    # Counts + lastChildEvent are this writer's exclusive top-level fields,
    # so a plain merge-patch is safe with no atomicity hand-shake.
    await _patch_parent_status(
        group="aiperf.nvidia.com",
        version="v1alpha1",
        plural="aiperfsweeps",
        name=sweep_name,
        namespace=namespace,
        body=body_patch,
    )

    # ``status.phase`` is co-written by three managers: kopf-default at
    # create (``Pending``), the sweep-controller (terminal ``Succeeded`` /
    # ``Failed`` from ``aggregation_complete``), and this rollup
    # (``Aggregating`` once every child is terminal). Without the
    # JSON-patch ``test`` op below, a TOCTOU between ``_read_parent_phase``
    # and the merge-patch would let this rollup clobber a freshly-written
    # terminal phase back to ``Aggregating``. The ``test`` op makes the
    # write conditional on the apiserver value still matching what we read.
    if not counts.get("total_terminal_phase"):
        return
    # Currently-listed children are all terminal — but the sweep-controller
    # creates child CRs lazily as it walks variations × trials, so an empty
    # in-flight set is NOT proof the sweep is done. The CR's
    # ``status.maxTotalRuns`` (set at create-time from variations × trials)
    # is the authoritative completion target; only flip phase to
    # ``Aggregating`` when ``completedRuns + failedRuns >= maxTotalRuns``.
    # Without this guard the parent flips to ``Aggregating`` after the very
    # first child terminates in a multi-cell sweep and stays there for the
    # rest of the run.
    parent_cr = await _read_parent_status(namespace, sweep_name)
    parent_phase = (parent_cr.get("phase") if parent_cr else "") or ""
    if parent_phase in PARENT_TERMINAL_PHASES:
        return
    max_total_runs = (parent_cr or {}).get("maxTotalRuns")
    if isinstance(max_total_runs, int) and max_total_runs > 0:
        accounted = counts["completed"] + counts["failed"]
        if accounted < max_total_runs:
            return
    await _conditional_phase_set(
        namespace=namespace,
        name=sweep_name,
        expect_phase=parent_phase,
        new_phase=counts["total_terminal_phase"],
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
    """Merge-patch operator-owned rollup fields on AIPerfSweep.status.

    Uses ``application/merge-patch+json`` with field manager
    ``aiperf-operator-rollup`` as observability metadata. The operator owns
    ``completedRuns``, ``failedRuns``, ``lastChildEvent``, and
    conditionally ``phase``; the sweep-controller writes disjoint fields
    (``currentCell``, ``aggregation``, ``aggregateRef``) under its own
    field manager. The disjoint-top-level-field invariant means merge-patch
    is safe — neither writer can clobber the other's fields. (Server-Side
    Apply was tried and reverted: SSA's relinquishment semantics drop a
    manager's own previously-set fields between calls when the new apply
    body doesn't include them, which broke the imperative event-style
    write pattern this code uses.)
    """
    import aiohttp
    import kopf
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            custom = k8s.CustomObjectsApi(api)
            await custom.patch_namespaced_custom_object_status(
                group=group,
                version=version,
                plural=plural,
                namespace=namespace,
                name=name,
                body=body,
                field_manager=ROLLUP_FIELD_MANAGER,
                _content_type="application/merge-patch+json",
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


async def _read_parent_status(namespace: str, name: str) -> dict[str, Any] | None:
    """Return parent AIPerfSweep ``status`` dict, or None if missing/unreadable.

    The rollup needs both ``phase`` (TOCTOU guard) and ``maxTotalRuns``
    (the operator-create-handler-set total target the rollup compares
    completed+failed against before flipping phase to ``Aggregating``).
    A single read avoids two GETs against the apiserver.
    """
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
    return (cr.get("status") or {}) or None


async def _read_parent_phase(namespace: str, name: str) -> str | None:
    """Return parent AIPerfSweep status.phase, or None if missing/unreadable.

    Thin wrapper around ``_read_parent_status`` retained for backwards
    compatibility with existing tests that patch this symbol directly.
    """
    status = await _read_parent_status(namespace, name)
    return (status or {}).get("phase") or None


async def _conditional_phase_set(
    *,
    namespace: str,
    name: str,
    expect_phase: str,
    new_phase: str,
) -> None:
    """Atomically write ``status.phase`` only when the apiserver still
    reflects ``expect_phase``.

    Uses a JSON-patch with a leading ``test`` op so a concurrent terminal
    write from the sweep-controller (between our read and this patch)
    flips the apiserver value, the test fails, and the apiserver returns
    422 — at which point we silently skip. Counts/lastChildEvent already
    landed via the prior merge-patch, so a skipped phase write is fine.

    When ``expect_phase`` is empty (initial create, before any phase has
    been written), the test would never match — fall back to a plain
    merge-patch in that case since the racy peer (sweep-controller) has
    not yet had a chance to write phase either.
    """
    import aiohttp
    import kopf
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    if not expect_phase:
        await _patch_parent_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            name=name,
            namespace=namespace,
            body={"status": {"phase": new_phase}},
        )
        return

    try:
        async with k8s_client() as api:
            custom = k8s.CustomObjectsApi(api)
            await custom.patch_namespaced_custom_object_status(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                plural="aiperfsweeps",
                namespace=namespace,
                name=name,
                body=[
                    {"op": "test", "path": "/status/phase", "value": expect_phase},
                    {"op": "replace", "path": "/status/phase", "value": new_phase},
                ],
                field_manager=ROLLUP_FIELD_MANAGER,
                _content_type="application/json-patch+json",
            )
    except ApiException as e:
        # 422 = test op failed; the parent moved to a different phase
        # between our read and our patch (typically: sweep-controller wrote
        # a terminal phase). Skip silently — that write is the source of
        # truth and we should not clobber it.
        if e.status == 422:
            return
        if e.status == 404:
            return
        raise kopf.TemporaryError(
            f"apiserver rejected phase patch ({e.status}): {e.reason}", delay=15
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during phase patch: {e}", delay=15
        ) from e
