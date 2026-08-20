# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""@kopf.on.field handler on AIPerfJob.status.phase.

When a child has an AIPerfSweep ownerReference, recompute the parent's
rollup counts. Standalone AIPerfJobs are no-ops.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aiperf.operator.handlers.sweep import _child_runs
from aiperf.operator.handlers.sweep._child_phase_buckets import (
    _api_or_new,
    _count_owned_children,
    _find_current_child,
    _is_owned_child,
)
from aiperf.operator.handlers.sweep._child_runs import (
    append_run_entry as _append_run_entry,
)

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

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

# ``_api_or_new``, ``_count_owned_children``, and ``_find_current_child``
# are imported from ``_child_phase_buckets`` above and re-exported here so
# existing test callers that
# ``monkeypatch.setattr(child_rollup, "_count_owned_children", ...)``
# continue to work without touching their import path.
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
    ownerReference, recompute the parent's rollup counts.

    Holds one ``k8s_client()`` context for the whole tick — under a 100-cell
    sweep with overlapping child terminations, this collapses 3-4 TCP/TLS
    handshakes per child phase change into one.
    """
    from aiperf.kubernetes.client import k8s_client

    parent = _find_sweep_owner(body)
    if parent is None:
        return
    sweep_name, sweep_uid = parent
    metadata = body.get("metadata") or {}
    child_uid = metadata.get("uid")
    child_run_epoch = (metadata.get("labels") or {}).get(
        "aiperf.nvidia.com/sweep-run-epoch"
    )
    if not isinstance(child_uid, str) or not isinstance(child_run_epoch, str):
        return
    if not _is_owned_child(
        body,
        sweep_uid=sweep_uid,
        sweep_name=sweep_name,
        run_epoch=child_run_epoch,
        expected_child_uid=child_uid,
    ):
        return

    async with k8s_client() as api:
        current_body = await _read_current_child(
            namespace=namespace,
            name=name,
            expected_child_uid=child_uid,
            sweep_name=sweep_name,
            sweep_uid=sweep_uid,
            run_epoch=child_run_epoch,
            api=api,
        )
        if current_body is None:
            return
        counts = await _count_owned_children(
            namespace,
            sweep_uid,
            sweep_name,
            run_epoch=child_run_epoch,
            api=api,
        )
        current_body = await _read_current_child(
            namespace=namespace,
            name=name,
            expected_child_uid=child_uid,
            sweep_name=sweep_name,
            sweep_uid=sweep_uid,
            run_epoch=child_run_epoch,
            api=api,
        )
        if current_body is None:
            return
        current_status = current_body.get("status") or {}
        body_patch = _build_rollup_status_patch(
            namespace=namespace,
            sweep_name=sweep_name,
            sweep_uid=sweep_uid,
            child_name=name,
            child_phase=current_status.get("phase", "Unknown"),
            counts=counts,
        )
        # Counts + lastChildEvent are exclusive top-level fields. The metadata
        # UID tells the helper which parent incarnation may receive them.
        #
        # Guard: skip the merge-patch when the parent sweep is already in a
        # terminal phase. The sweep-controller's aggregation_complete patch
        # writes completedRuns/failedRuns/runStates with authoritative values
        # at terminal time; a delayed rollup tick that fires after the children
        # are deleted (post-harvest) would find _count_owned_children returning
        # 0 and would overwrite those authoritative counts with zeros.
        # Fall through to _advance_parent_phase_if_complete regardless so the
        # _ingest_sweep_aggregate side-effect still fires; that helper reuses
        # this same read instead of issuing a second GET.
        parent_status_pre = await _read_parent_status(
            namespace, sweep_name, expected_uid=sweep_uid, api=api
        )
        parent_phase_pre = (parent_status_pre or {}).get("phase") or ""
        if parent_phase_pre not in PARENT_TERMINAL_PHASES:
            await _patch_parent_status(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                plural="aiperfsweeps",
                name=sweep_name,
                namespace=namespace,
                body=body_patch,
                api=api,
            )
        # If this rollup was triggered by a TERMINAL child phase, append a
        # slim summary entry to AIPerfSweep.status.runs[] (~600 B / entry).
        # Truncation safety net for huge sweeps lives in Task 12.
        if (
            current_status.get("phase") or ""
        ).lower() in _child_runs.TERMINAL_CHILD_PHASES:
            current_body = await _read_current_child(
                namespace=namespace,
                name=name,
                expected_child_uid=child_uid,
                sweep_name=sweep_name,
                sweep_uid=sweep_uid,
                run_epoch=child_run_epoch,
                api=api,
            )
            if current_body is None:
                return
            current_status = current_body.get("status") or {}
            entry = _child_runs.build_run_entry(
                body=current_body, status=current_status, name=name
            )
            await _append_run_entry(
                namespace,
                sweep_name,
                entry,
                expected_uid=sweep_uid,
                api=api,
            )
        await _advance_parent_phase_if_complete(
            namespace=namespace,
            child_name=name,
            sweep_name=sweep_name,
            sweep_uid=sweep_uid,
            run_epoch=child_run_epoch,
            expected_child_uid=child_uid,
            counts=counts,
            parent_status=parent_status_pre,
            api=api,
        )


def _build_rollup_status_patch(
    *,
    namespace: str,
    sweep_name: str,
    sweep_uid: str,
    child_name: str,
    child_phase: str,
    counts: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the operator-owned merge-patch body for AIPerfSweep.status.

    Split out of ``on_child_phase_transition`` because it is a pure mapping
    from the child-phase bucket counts to JSON — no apiserver I/O, no
    ordering constraints — which leaves the handler readable as a linear
    sequence of awaited apiserver steps and makes the patch shape (the part
    that has to stay disjoint from the sweep-controller's fields) directly
    unit-testable.
    """
    from aiperf.operator.environment import OperatorEnvironment

    body_patch: dict[str, Any] = {
        "metadata": {"uid": sweep_uid},
        "status": {
            "completedRuns": counts["completed"],
            "failedRuns": counts["failed"],
            "runStates": {
                "pending": counts.get("pending", 0),
                "running": counts.get("running", 0),
                "completed": counts["completed"],
                "failed": counts["failed"],
                "cancelled": counts.get("cancelled", 0),
            },
            "lastChildEvent": {"name": child_name, "phase": child_phase},
            # Re-stamp apiUrl every rollup tick so AIPerfSweep CRs created
            # before the URL-collapse cleanup self-heal post-upgrade. Stale
            # `:8080` values from old chart installs (no FastAPI on that
            # port) get overwritten on the next child phase change. The
            # create-handler stamps once on `handle()`, but reconciles
            # never re-touched this field — leaving in-flight CRs broken
            # until a delete+recreate. Idempotent merge-patch.
            "apiUrl": (
                f"{OperatorEnvironment.SERVICE.BASE_URL.rstrip('/')}"
                f"/api/v1/sweeps/{namespace}/{sweep_name}"
            ),
            # Pointer to the active child for `kubectl get aiperfsweep -o yaml`
            # drill-down. See `_find_current_child` for selection priority.
            "currentChildRef": _build_current_child_ref(counts),
        },
    }
    return body_patch


def _build_current_child_ref(counts: dict[str, Any]) -> dict[str, Any] | None:
    """Return the ``status.currentChildRef`` value, or None when no child is active.

    The variation index is label-derived and therefore user/CRD-writable, so a
    non-integer label degrades to ``-1`` rather than failing the whole rollup.
    """
    current = _find_current_child(counts.get("owned_children") or [])
    if current is None:
        return None
    labels = (current.get("metadata") or {}).get("labels") or {}
    try:
        idx = int(labels.get("aiperf.nvidia.com/variation-index", "-1"))
    except (TypeError, ValueError):
        idx = -1
    return {
        "name": current["metadata"]["name"],
        "index": idx,
        "label": labels.get("aiperf.nvidia.com/variation-label", ""),
    }


async def _advance_parent_phase_if_complete(
    *,
    namespace: str,
    child_name: str,
    sweep_name: str,
    sweep_uid: str,
    run_epoch: str,
    expected_child_uid: str,
    counts: dict[str, Any],
    parent_status: dict[str, Any] | None,
    api: ApiClient,
) -> None:
    """Advance the exact parent only after every expected run is accounted.

    ``parent_status`` is the caller's pre-patch read, threaded through rather
    than re-read: the intervening merge-patch never touches ``phase`` or
    ``maxTotalRuns``, and a concurrent terminal write from the sweep-controller
    is caught by the ``test`` op inside ``_conditional_phase_set`` (422 → skip),
    so the second GET bought nothing.
    """
    terminal_phase = counts.get("total_terminal_phase")
    if not terminal_phase:
        return
    current = await _read_current_child(
        namespace=namespace,
        name=child_name,
        expected_child_uid=expected_child_uid,
        sweep_name=sweep_name,
        sweep_uid=sweep_uid,
        run_epoch=run_epoch,
        api=api,
    )
    if current is None:
        return
    parent_phase = (parent_status or {}).get("phase") or ""
    if parent_phase in PARENT_TERMINAL_PHASES:
        await _ingest_sweep_aggregate(namespace, sweep_name)
        return
    max_total_runs = (parent_status or {}).get("maxTotalRuns")
    accounted = counts["completed"] + counts["failed"] + counts.get("cancelled", 0)
    if (
        isinstance(max_total_runs, int)
        and max_total_runs > 0
        and accounted < max_total_runs
    ):
        return
    await _conditional_phase_set(
        namespace=namespace,
        name=sweep_name,
        expect_phase=parent_phase,
        new_phase=terminal_phase,
        expected_uid=sweep_uid,
        api=api,
    )


def _find_sweep_owner(child_body: dict[str, Any]) -> tuple[str, str] | None:
    refs = (child_body.get("metadata") or {}).get("ownerReferences") or []
    for ref in refs:
        if (
            isinstance(ref, dict)
            and ref.get("apiVersion") == "aiperf.nvidia.com/v1alpha1"
            and ref.get("kind") == "AIPerfSweep"
            and ref.get("controller") is True
            and ref.get("name")
            and ref.get("uid")
        ):
            return ref["name"], ref["uid"]
    return None


async def _read_current_child(
    *,
    namespace: str,
    name: str,
    expected_child_uid: str,
    sweep_name: str,
    sweep_uid: str,
    run_epoch: str,
    api: ApiClient,
) -> dict[str, Any] | None:
    """Read and validate the exact child resource that triggered this tick."""
    import aiohttp
    import kopf
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    custom = k8s.CustomObjectsApi(api)
    try:
        child = await custom.get_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfjobs",
            name=name,
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise kopf.TemporaryError(
            f"apiserver rejected child identity read ({exc.status}): {exc.reason}",
            delay=15,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"apiserver unreachable during child identity read: {exc}", delay=15
        ) from exc
    if not _is_owned_child(
        child,
        sweep_uid=sweep_uid,
        sweep_name=sweep_name,
        run_epoch=run_epoch,
        expected_child_uid=expected_child_uid,
    ):
        return None
    return child


async def _ingest_sweep_aggregate(namespace: str, sweep_name: str) -> None:
    """Best-effort ingest of ``aggregate.json`` for the sweep's latest epoch.

    Imported lazily (``runs_index`` and ``results_layout`` import paths)
    to keep this handler module slim and avoid pulling the index code
    into pure-rollup unit tests that don't need it. Failures log and
    swallow so the rollup tick never fails on index-side issues.
    """
    try:
        from aiperf.operator import runs_index
        from aiperf.operator.environment import OperatorEnvironment
        from aiperf.operator.results_layout import resolve_sweep_dir
    except ImportError as exc:  # pragma: no cover - defensive
        logger.warning("runs_index unavailable for sweep aggregate ingest: %s", exc)
        return

    base = OperatorEnvironment.RESULTS.DIR
    sweep_epoch_dir = resolve_sweep_dir(base, namespace, sweep_name)
    if sweep_epoch_dir is None:
        return
    try:
        await runs_index._index_sweep_from_disk(
            namespace, sweep_name, sweep_epoch_dir.name, sweep_epoch_dir
        )
    except Exception as exc:  # noqa: BLE001 - index path must never break the rollup
        logger.warning(
            "runs_index sweep aggregate ingest failed for %s/%s: %s",
            namespace,
            sweep_name,
            exc,
        )


async def _patch_parent_status(
    *,
    group: str,
    version: str,
    plural: str,
    name: str,
    namespace: str,
    body: dict[str, Any],
    api: ApiClient | None = None,
) -> None:
    """Merge-patch operator-owned rollup fields on AIPerfSweep.status.

    Uses ``application/merge-patch+json`` with field manager
    ``aiperf-operator-rollup`` as observability metadata. The operator owns
    ``completedRuns``, ``failedRuns``, ``lastChildEvent``, and
    conditionally ``phase``; the sweep-controller writes disjoint fields
    (``currentCell``, ``aggregation``, ``aggregateRef``) under its own
    field manager. The disjoint-top-level-field invariant means merge-patch
    is safe — neither writer can clobber the other's fields. When ``body``
    carries ``metadata.uid``, the helper first verifies that immutable UID and
    includes the current resourceVersion as an optimistic concurrency guard.
    This prevents an old child's delayed event from patching a recreated,
    same-named sweep. (Server-Side Apply was tried and reverted: SSA's
    relinquishment semantics drop a manager's own previously-set fields
    between calls when the new apply body doesn't include them.)
    """
    import aiohttp
    import kopf
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    try:
        async with _api_or_new(api) as client:
            custom = k8s.CustomObjectsApi(client)
            expected_uid = (body.get("metadata") or {}).get("uid")
            patch_body = body
            if isinstance(expected_uid, str):
                current = await custom.get_namespaced_custom_object(
                    group=group,
                    version=version,
                    plural=plural,
                    namespace=namespace,
                    name=name,
                )
                metadata = current.get("metadata") or {}
                if metadata.get("uid") != expected_uid:
                    return
                resource_version = metadata.get("resourceVersion")
                if not isinstance(resource_version, str):
                    raise kopf.TemporaryError(
                        f"AIPerfSweep {namespace}/{name} has no resourceVersion",
                        delay=15,
                    )
                patch_body = {
                    **body,
                    "metadata": {"resourceVersion": resource_version},
                }
            await custom.patch_namespaced_custom_object_status(
                group=group,
                version=version,
                plural=plural,
                namespace=namespace,
                name=name,
                body=patch_body,
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


async def _read_parent_status(
    namespace: str,
    name: str,
    *,
    expected_uid: str | None = None,
    api: ApiClient | None = None,
) -> dict[str, Any] | None:
    """Return parent AIPerfSweep ``status`` dict, or None if missing/unreadable.

    The rollup needs both ``phase`` (TOCTOU guard) and ``maxTotalRuns``
    (the operator-create-handler-set total target the rollup compares
    completed+failed against before flipping phase to ``Aggregating``).
    A single read avoids two GETs against the apiserver.

    Returning ``None`` means "the CR genuinely has no status yet" (404 →
    initial create) — the caller treats that as a safe unconditional set.
    A transient read failure must NOT collapse into that same ``None`` or
    it would defeat both the TOCTOU ``test``-op guard and the
    ``maxTotalRuns`` guard, regressing a freshly-written terminal phase
    back to ``Aggregating``. So transient errors raise
    ``kopf.TemporaryError`` (mirroring ``_patch_parent_status``) and the
    tick retries instead of clobbering.
    """
    import aiohttp
    import kopf
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    try:
        async with _api_or_new(api) as client:
            custom = k8s.CustomObjectsApi(client)
            cr = await custom.get_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="aiperfsweeps",
                name=name,
            )
    except ApiException as e:
        if e.status == 404:
            return None
        raise kopf.TemporaryError(
            f"apiserver rejected status read ({e.status}): {e.reason}", delay=15
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during status read: {e}", delay=15
        ) from e
    if expected_uid is not None:
        current_uid = (cr.get("metadata") or {}).get("uid")
        if current_uid != expected_uid:
            return None
    return (cr.get("status") or {}) or None


async def _read_parent_phase(
    namespace: str, name: str, *, api: ApiClient | None = None
) -> str | None:
    """Return parent AIPerfSweep status.phase, or None if missing/unreadable.

    Thin wrapper around ``_read_parent_status`` retained for backwards
    compatibility with existing tests that patch this symbol directly.
    """
    status = await _read_parent_status(namespace, name, api=api)
    return (status or {}).get("phase") or None


async def _conditional_phase_set(
    *,
    namespace: str,
    name: str,
    expect_phase: str,
    new_phase: str,
    expected_uid: str | None = None,
    api: ApiClient | None = None,
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

    if not expect_phase:
        await _patch_parent_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            name=name,
            namespace=namespace,
            body={
                **({"metadata": {"uid": expected_uid}} if expected_uid else {}),
                "status": {"phase": new_phase},
            },
            api=api,
        )
        return

    try:
        async with _api_or_new(api) as client:
            custom = k8s.CustomObjectsApi(client)
            patch_ops: list[dict[str, Any]] = []
            if expected_uid is not None:
                patch_ops.append(
                    {
                        "op": "test",
                        "path": "/metadata/uid",
                        "value": expected_uid,
                    }
                )
            patch_ops.extend(
                [
                    {"op": "test", "path": "/status/phase", "value": expect_phase},
                    {"op": "replace", "path": "/status/phase", "value": new_phase},
                ]
            )
            await custom.patch_namespaced_custom_object_status(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                plural="aiperfsweeps",
                namespace=namespace,
                name=name,
                body=patch_ops,
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
