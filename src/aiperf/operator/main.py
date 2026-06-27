# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes Operator.

Handles AIPerfJob CRD lifecycle with:
- Spec validation and endpoint health checks
- Kubernetes event emission
- Condition tracking (ConfigValid, EndpointReachable, ResourcesCreated, etc.)
- Metrics summary extraction
- Results storage with retry logic
- Job cancellation support
- Job timeout detection
- Pod restart monitoring
- Results TTL cleanup

Run: kopf run -m aiperf.operator.main --verbose

Handler categories dispatched below, in order: startup (configure),
lifecycle (on_create / on_delete / on_cancel / on_benchmark_complete),
and timers (monitor_progress, cleanup_old_results).

All kopf decorators live here so handler modules stay decorator-free
and are independently testable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import replace
from typing import Any

import kopf
from kopf._cogs.structs.credentials import ConnectionInfo

from aiperf.kubernetes.client import APISERVER_TLS_SERVER_NAME_OVERRIDE_ENV
from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_GROUP,
    AIPERF_PLURAL,
    AIPERF_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.operator import runs_index
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers import cleanup, create, lifecycle, monitor
from aiperf.operator.handlers import jobset_terminal as jobset_terminal_handler
from aiperf.operator.handlers import pod_restarts as pod_restarts_handler
from aiperf.operator.handlers.sweep import child_rollup as sweep_rollup
from aiperf.operator.handlers.sweep import create as sweep_create
from aiperf.operator.handlers.sweep import lifecycle as sweep_lifecycle
from aiperf.operator.metrics import start_metrics_server, track_handler

AIPERF_SWEEPS_PLURAL = "aiperfsweeps"

logger = logging.getLogger(__name__)


@kopf.on.login()
async def login_for_apiserver_proxy(
    *,
    logger: logging.Logger | logging.LoggerAdapter,
    settings: kopf.OperatorSettings,
    **_: Any,
) -> ConnectionInfo | None:
    """Authenticate kopf, allowing the C15 apiserver proxy route to connect."""
    connection = await kopf.login_via_async_client(logger=logger, settings=settings)
    if connection is None:
        return None
    if not os.environ.get(APISERVER_TLS_SERVER_NAME_OVERRIDE_ENV, "").strip():
        return connection
    logger.warning(
        "Disabling kopf apiserver TLS verification because %s is set; "
        "AIPerf direct Kubernetes clients still verify using tls_server_name",
        APISERVER_TLS_SERVER_NAME_OVERRIDE_ENV,
    )
    return replace(connection, insecure=True)


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_: Any) -> None:
    """Configure operator settings."""
    settings.persistence.finalizer = f"{AIPERF_GROUP}/finalizer"
    settings.posting.level = logging.INFO
    start_metrics_server(OperatorEnvironment.METRICS_PORT)


@kopf.on.create(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL)
@track_handler("on_create")
async def on_create(
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    patch: kopf.Patch,
    **_: Any,
) -> dict[str, Any]:
    """Create ConfigMap and JobSet for the benchmark job."""
    return await create.on_create(
        body=body, spec=spec, name=name, namespace=namespace, uid=uid, patch=patch
    )


@kopf.on.delete(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL)
@track_handler("on_delete")
async def on_delete(
    name: str, namespace: str, status: dict[str, Any], **_: Any
) -> None:
    """Clean up cached ProgressClient on CR deletion."""
    await lifecycle.on_delete(name=name, namespace=namespace, status=status)


@kopf.on.update(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL, field="spec.cancel")
@track_handler("on_cancel")
async def on_cancel(
    body: dict[str, Any],
    spec: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle cancellation request via spec.cancel field."""
    await lifecycle.on_cancel(
        body=body, spec=spec, status=status, name=name, namespace=namespace, patch=patch
    )


@kopf.on.update(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    annotations={Annotations.BENCHMARK_COMPLETE: "true"},
)
@track_handler("on_benchmark_complete")
async def on_benchmark_complete(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle benchmark completion signal from controller pod."""
    await lifecycle.on_benchmark_complete(
        body=body, status=status, name=name, namespace=namespace, patch=patch
    )


@kopf.on.create(AIPERF_GROUP, AIPERF_VERSION, AIPERF_SWEEPS_PLURAL)
@track_handler("on_aiperfsweep_create")
async def on_aiperfsweep_create(
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Validate, provision RBAC, and create the sweep-controller JobSet."""
    await sweep_create.handle(
        body=body, spec=spec, name=name, namespace=namespace, patch=patch
    )


@kopf.on.update(AIPERF_GROUP, AIPERF_VERSION, AIPERF_SWEEPS_PLURAL, field="spec.cancel")
@track_handler("on_aiperfsweep_cancel")
async def on_aiperfsweep_cancel(
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Mirror spec.cancel into status.conditions[Cancelling]."""
    await sweep_lifecycle.cancel(
        body=body, spec=spec, name=name, namespace=namespace, patch=patch
    )


async def _delete_sweep_jobset(namespace: str, jobset_name: str) -> None:
    """Delete a sweep-controller's backing JobSet; tolerate a missing one.

    Idempotent across reconciles: a 404 (already deleted on a prior tick or by
    CR-TTL cascade) is not an error. Non-404 failures are logged but not
    re-raised — the harvest already succeeded, so a transient delete failure
    must not flip the handler to a retry that re-harvests pointlessly; the next
    reconcile (or CR TTL) reaps the pod.
    """
    from kubernetes_asyncio import client
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            await client.CustomObjectsApi(api).delete_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=namespace,
                name=jobset_name,
            )
        logger.info(f"Deleted sweep JobSet {namespace}/{jobset_name} after harvest")
    except ApiException as e:
        if e.status != 404:
            logger.warning(
                f"Failed to delete sweep JobSet {namespace}/{jobset_name} "
                f"after harvest: {e}"
            )


@kopf.on.field(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_SWEEPS_PLURAL,
    field="status.aggregation.phase",
    new="Complete",
)
@track_handler("on_aiperfsweep_aggregation_complete")
async def on_aiperfsweep_aggregation_complete(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """Harvest the cross-variation aggregate from sweep-controller's sidecar.

    The sweep-controller writes ``/results/{aggregate,sweeps}/...`` and the
    ``.aiperf_results_ready.json`` marker; the sidecar then serves them
    over HTTP. The pod uses ``emptyDir{}`` for ``/results`` (per the
    no-PVC-on-controller-pods constraint), so the operator MUST pull the
    artifacts to its own results PVC at
    ``<base>/<ns>/sweeps/<name>/<runEpoch>/`` before the JobSet is
    deleted. Without this, ``getSweepEpochs`` and ``getSweepCells`` find
    nothing on disk and the SweepDetail page renders as empty cells.
    """
    from aiperf.operator.environment import OperatorEnvironment
    from aiperf.operator.handlers.sweep import _aggregate_fetch

    epoch = (status or {}).get("runEpoch")
    if epoch is None:
        logger.warning(
            f"AIPerfSweep {namespace}/{name} aggregation Complete with no "
            f"status.runEpoch; skipping disk persistence"
        )
        return
    base_dir = OperatorEnvironment.RESULTS.DIR
    fetched_count = await _aggregate_fetch.fetch_sweep_aggregate_to_disk(
        sweep_name=name,
        namespace=namespace,
        epoch=str(epoch),
        base_dir=base_dir,
    )
    if fetched_count == 0:
        # A re-fire AFTER we deleted the JobSet (below) hits a dead sidecar
        # and gets 0 files — but the harvest already happened on the tick
        # that did the delete. Treat an already-populated dest-dir as success
        # so the handler does not loop forever on TemporaryError once the pod
        # is gone. Only retry when the artifacts are genuinely absent.
        sweep_epoch_dir = (
            base_dir / namespace / "sweeps" / name / str(epoch) / "aggregate.json"
        )
        if sweep_epoch_dir.exists():
            logger.info(
                f"AIPerfSweep {namespace}/{name} aggregate already on disk "
                f"(epoch={epoch}); sidecar gone after JobSet reap, treating as done"
            )
            await _delete_sweep_jobset(namespace, f"aiperf-{name}")
            return
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} aggregate sidecar returned no files; retrying",
            delay=30,
        )

    # The aggregate is now on the operator's PVC, so the sweep-controller pod
    # has nothing left to serve. Delete its JobSet to reap the pod promptly —
    # otherwise the pod lingers until the CR's `ttlSecondsAfterFinished`
    # reaper fires, because the controller container exits 0 but the
    # results-sidecar runs uvicorn forever and a Job pod only reaches
    # `Succeeded` once ALL containers terminate. Mirrors the AIPerfJob
    # harvest's `_maybe_delete_jobset_after_success` (delete only after a
    # successful fetch, so we never tear the sidecar down before harvesting).
    await _delete_sweep_jobset(namespace, f"aiperf-{name}")


@kopf.on.field(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL, field="status.phase")
@track_handler("on_aiperfjob_phase_transition")
async def on_aiperfjob_phase_transition(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """Bubble AIPerfJob phase transitions up to owning AIPerfSweep, if any."""
    await sweep_rollup.on_child_phase_transition(
        body=body, status=status, name=name, namespace=namespace
    )
    await lifecycle.record_phase_transition(
        namespace=namespace, name=name, status=status
    )


@kopf.on.event(
    "v1",
    "pods",
    labels={"jobset.sigs.k8s.io/jobset-name": kopf.PRESENT},
)
@track_handler("on_pod_container_status_change")
async def on_pod_container_status_change(
    *,
    event: dict[str, Any],
    body: dict[str, Any],
    meta: dict[str, Any],
    namespace: str,
    name: str,
    **_: Any,
) -> None:
    """React to JobSet-labeled Pod restart counts; replaces the monitor-tick poll.

    Uses ``@kopf.on.event`` rather than ``@kopf.on.field`` because field-watchers
    require kopf to write a per-resource diff-base annotation (``pods: patch``
    RBAC), which the operator does not have on benchmark namespaces. Event
    handlers don't need that — kopf stores no state on the watched Pod.
    Dedup via ``_warned_pod_restarts`` (in-process) handles "same restart count,
    don't emit twice" without help from kopf.
    """
    if event.get("type") == "DELETED":
        return
    new = ((body.get("status") or {}).get("containerStatuses")) or []
    await pod_restarts_handler.handle_pod_restart(
        old=[],
        new=new,
        body=body,
        meta=meta,
        namespace=namespace,
        name=name,
        threshold=OperatorEnvironment.POD_RESTART_THRESHOLD,
    )


@kopf.on.field(
    JOBSET_GROUP,
    JOBSET_VERSION,
    JOBSET_PLURAL,
    field="status.conditions",
)
@track_handler("on_jobset_conditions")
async def on_jobset_conditions(
    *,
    old: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
    namespace: str,
    name: str,
    body: dict[str, Any],
    **_: Any,
) -> None:
    """Detect a JobSet terminal-success flip immediately rather than waiting for the next monitor tick.

    Failure-condition recovery stays on the monitor-timer path; this handler only
    annotates the parent AIPerfJob on success, which makes kopf dispatch the
    existing ``on_benchmark_complete`` handler.
    """
    await jobset_terminal_handler.handle_jobset_conditions(
        old=old, new=new, namespace=namespace, jobset_name=name, jobset_body=body
    )


@kopf.on.delete(AIPERF_GROUP, AIPERF_VERSION, AIPERF_SWEEPS_PLURAL)
@track_handler("on_aiperfsweep_delete")
async def on_aiperfsweep_delete(name: str, namespace: str, **_: Any) -> None:
    """On AIPerfSweep deletion, request cooperative cancellation of any running children.

    OwnerReferences will cascade-GC child AIPerfJobs and the sweep-controller
    JobSet, but cooperative cancel lets in-flight benchmarks shut down
    cleanly (write-out partial results, signal workers) before the cascade
    SIGKILLs them.
    """
    await sweep_lifecycle.on_delete(name=name, namespace=namespace)


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_SWEEPS_PLURAL,
    interval=86400.0,
    initial_delay=3600.0,
    idle=3600.0,
)
@track_handler("cleanup_old_sweeps")
async def cleanup_old_sweeps(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """Delete AIPerfSweep CRs whose ttlSecondsAfterFinished has elapsed."""
    await sweep_lifecycle.maybe_reap_finished(
        body=body, status=status, name=name, namespace=namespace
    )


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    interval=OperatorEnvironment.MONITOR.INTERVAL,
    initial_delay=OperatorEnvironment.MONITOR.INITIAL_DELAY,
)
@track_handler("monitor_progress")
async def monitor_progress(
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Monitor job progress and update status."""
    await monitor.monitor_progress(
        body=body, status=status, spec=spec, name=name, namespace=namespace, patch=patch
    )


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    interval=86400.0,
    initial_delay=3600.0,
    idle=3600.0,
)
@track_handler("cleanup_old_results")
async def cleanup_old_results(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    **_: Any,
) -> None:
    """Clean up old results based on TTL."""
    await cleanup.cleanup_old_results(body=body, status=status, name=name)


@kopf.on.startup()
async def open_runs_index(**_: Any) -> None:
    """Open the runs_index SQLite DB and schedule a background bootstrap.

    On corruption, rename the file to ``.broken-<unix>`` for forensics and
    reopen a fresh DB. ``bootstrap`` runs as a background task so operator
    readiness is not gated on a full PVC scan.
    """
    base = OperatorEnvironment.RESULTS.DIR
    db_path = base / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    if not await runs_index.integrity_check():
        logger.warning("runs_index corrupt; renaming and rebuilding")
        broken = base / f".aiperf_index.sqlite.broken-{int(time.time())}"
        db_path.rename(broken)
        await runs_index.close()
        await runs_index.open(db_path)

    # Fire-and-forget bootstrap with a done-callback so any unhandled
    # exception lands in the operator's log instead of asyncio's "Task
    # exception was never retrieved" GC warning. Per-iteration try/except
    # in runs_index.bootstrap covers per-run failures; this catches the
    # outer-loop edge cases (PVC unmount mid-startup, EACCES on iterdir,
    # sqlite-level errors that escape the inner guards).
    bootstrap_task = asyncio.create_task(runs_index.bootstrap(base))

    def _log_bootstrap_exception(t: asyncio.Task[Any]) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.exception(
                "runs_index bootstrap task crashed (operator continues without "
                "rebuilt index; trigger `aiperf kube index rebuild` to recover): %s",
                exc,
            )

    bootstrap_task.add_done_callback(_log_bootstrap_exception)


@kopf.on.cleanup()
async def close_runs_index(**_: Any) -> None:
    """Close the runs_index SQLite connection on operator shutdown."""
    await runs_index.close()
