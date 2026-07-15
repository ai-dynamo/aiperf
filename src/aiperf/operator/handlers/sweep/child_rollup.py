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
    from aiperf.operator.environment import OperatorEnvironment

    parent = _find_sweep_owner(body)
    if parent is None:
        return
    sweep_name, sweep_uid = parent

    # Only count children from the same run-epoch as the child that just
    # transitioned. Without this filter the rollup picks up stale children
    # from prior re-applies of the sweep and writes ``completedRuns >
    # totalVariations`` (the UI then shows a nonsensical "5 / 3"). The
    # epoch label is set at child-creation time by the sweep-controller;
    # missing label → fall back to the unfiltered count for backward
    # compatibility with old children created before the label existed.
    child_labels = (body.get("metadata") or {}).get("labels") or {}
    child_run_epoch = child_labels.get("aiperf.nvidia.com/sweep-run-epoch")

    async with k8s_client() as api:
        counts = await _count_owned_children(
            namespace,
            sweep_uid,
            sweep_name,
            run_epoch=child_run_epoch,
            api=api,
        )
        body_patch: dict[str, Any] = {
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
                "lastChildEvent": {
                    "name": name,
                    "phase": status.get("phase", "Unknown"),
                },
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
            }
        }
        # Pointer to the active child for `kubectl get aiperfsweep -o yaml`
        # drill-down. See `_find_current_child` for selection priority.
        children = counts.get("owned_children") or []
        current = _find_current_child(children)
        if current is not None:
            labels = (current.get("metadata") or {}).get("labels") or {}
            try:
                idx = int(labels.get("aiperf.nvidia.com/variation-index", "-1"))
            except (TypeError, ValueError):
                idx = -1
            body_patch["status"]["currentChildRef"] = {
                "name": current["metadata"]["name"],
                "index": idx,
                "label": labels.get("aiperf.nvidia.com/variation-label", ""),
            }
        else:
            body_patch["status"]["currentChildRef"] = None
        # Counts + lastChildEvent are this writer's exclusive top-level fields,
        # so a plain merge-patch is safe with no atomicity hand-shake.
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
        if (status.get("phase") or "").lower() in _child_runs.TERMINAL_CHILD_PHASES:
            entry = _child_runs.build_run_entry(body=body, status=status, name=name)
            await _append_run_entry(namespace, sweep_name, entry, api=api)

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
        parent_cr = await _read_parent_status(namespace, sweep_name, api=api)
        parent_phase = (parent_cr.get("phase") if parent_cr else "") or ""
        if parent_phase in PARENT_TERMINAL_PHASES:
            # Sweep-controller has already written the terminal phase and
            # (presumably) the aggregate.json. Best-effort: ingest the
            # aggregate into runs_index so analytics queries hit the index
            # rather than re-reading aggregate.json on every request.
            # Underscore-internal _index_sweep_from_disk is reused here as
            # the canonical "ingest one sweep epoch" entry point rather than
            # duplicating the aggregate-walk logic.
            await _ingest_sweep_aggregate(namespace, sweep_name)
            return
        max_total_runs = (parent_cr or {}).get("maxTotalRuns")
        if isinstance(max_total_runs, int) and max_total_runs > 0:
            accounted = (
                counts["completed"] + counts["failed"] + counts.get("cancelled", 0)
            )
            if accounted < max_total_runs:
                return
        await _conditional_phase_set(
            namespace=namespace,
            name=sweep_name,
            expect_phase=parent_phase,
            new_phase=counts["total_terminal_phase"],
            api=api,
        )


def _find_sweep_owner(child_body: dict[str, Any]) -> tuple[str, str] | None:
    refs = (child_body.get("metadata") or {}).get("ownerReferences") or []
    for ref in refs:
        if ref.get("kind") == "AIPerfSweep" and ref.get("name") and ref.get("uid"):
            return ref["name"], ref["uid"]
    return None


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

    try:
        async with _api_or_new(api) as client:
            custom = k8s.CustomObjectsApi(client)
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


async def _read_parent_status(
    namespace: str, name: str, *, api: ApiClient | None = None
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
            body={"status": {"phase": new_phase}},
            api=api,
        )
        return

    try:
        async with _api_or_new(api) as client:
            custom = k8s.CustomObjectsApi(client)
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
