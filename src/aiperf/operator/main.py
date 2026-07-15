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
from pathlib import Path
from typing import Any

import kopf
import orjson
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


def _sweep_aggregate_on_disk(aggregate_path: Path) -> bool:
    """True iff a parseable ``aggregate.json`` is present at ``aggregate_path``.

    An ``exists()`` check alone is not enough: a fetch interrupted mid-stream
    or an operator crash mid-write can leave a truncated ``aggregate.json``
    that passes ``exists()`` but is unreadable — and once the sweep JobSet is
    deleted, the only other copy (the controller pod's emptyDir) is gone.
    Parse failure is therefore treated exactly like absence, keeping the
    caller on the re-fetch path.
    """
    try:
        orjson.loads(aggregate_path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return False
    return True


SWEEP_HARVEST_SENTINEL_NAME = ".aiperf_sweep_harvest_complete.json"
"""Dotfile written next to ``aggregate.json`` after a FULL harvest.

Follows the ``.aiperf_*`` marker convention (``.aiperf_results_ready.json``,
``.aiperf_results_processing.json``). Its presence is the positive evidence
that every file the sidecar advertised for this epoch landed on the
operator's PVC — a parseable ``aggregate.json`` alone also matches a PARTIAL
harvest (aggregate landed, sibling artifacts did not), which must never be
treated as done while the controller pod's emptyDir still holds the only
other copy of the missing files.
"""


def _sweep_harvest_sentinel_path(aggregate_path: Path) -> Path:
    """Return the harvest-complete sentinel path for an epoch's aggregate."""
    return aggregate_path.parent / SWEEP_HARVEST_SENTINEL_NAME


def _write_sweep_harvest_sentinel(
    aggregate_path: Path, *, downloaded: int, listed: int
) -> None:
    """Record that a full ``downloaded == listed`` harvest reached the PVC.

    Written ONLY from the full-success path, immediately before the JobSet
    delete, so an operator that crashes between the two still converges on
    the next tick via the sentinel. A write failure is advisory: the harvest
    itself already succeeded, and the no-sentinel fallback (JobSet-existence
    check) still resolves later zero-download re-fires correctly.
    """
    sentinel = _sweep_harvest_sentinel_path(aggregate_path)
    try:
        sentinel.write_bytes(
            orjson.dumps(
                {"harvestComplete": True, "downloaded": downloaded, "listed": listed}
            )
        )
    except OSError as exc:
        logger.warning(
            f"Failed to write sweep harvest sentinel {sentinel} "
            f"({type(exc).__name__}: {exc}); harvest already succeeded so this "
            f"is advisory"
        )


async def _sweep_jobset_exists(namespace: str, jobset_name: str) -> bool:
    """Whether the sweep-controller's JobSet is still on the apiserver.

    Returns False only on a confirmed 404. Any other outcome (found, apiserver
    error, transient connectivity failure) counts as "exists": callers use
    absence as the licence to stop re-harvesting the controller pod's
    emptyDir, so ambiguity must fail toward retrying — never toward giving up
    on files that may still be recoverable.
    """
    from kubernetes_asyncio import client
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            await client.CustomObjectsApi(api).get_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=namespace,
                name=jobset_name,
            )
    except ApiException as e:
        return e.status != 404
    except Exception:  # noqa: BLE001 - existence gate is fail-safe; unknown counts as exists
        return True
    return True


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
    fetched = await _aggregate_fetch.fetch_sweep_aggregate_to_disk(
        sweep_name=name,
        namespace=namespace,
        epoch=str(epoch),
        base_dir=base_dir,
    )
    aggregate_marker = (
        base_dir / namespace / "sweeps" / name / str(epoch) / "aggregate.json"
    )
    jobset_name = f"aiperf-{name}"
    if fetched.downloaded == 0:
        if fetched.listed > 0:
            # The sidecar is alive and advertising files, yet none landed on
            # the PVC — a failed download, never an already-finished harvest.
            # Same reasoning as the partial branch below: the emptyDir holds
            # the only other copy, so keep the JobSet and re-harvest.
            raise kopf.TemporaryError(
                f"AIPerfSweep {namespace}/{name} sidecar listed {fetched.listed} "
                f"file(s) but none downloaded; retrying",
                delay=30,
            )
        # listed == 0: the sidecar is unreachable, pre-marker, or gone after a
        # prior tick's JobSet reap. Treating an already-harvested epoch as done
        # keeps the handler from looping forever once the pod is gone — but a
        # parseable aggregate.json alone is NOT proof of a full harvest: a
        # partial harvest (5/6 files, aggregate landed) followed by transient
        # sidecar unreachability produces the exact same on-disk shape, and
        # deleting the JobSet then destroys the emptyDir-only copy of the
        # missing files. Require the harvest-complete sentinel as positive
        # evidence before the shortcut.
        if _sweep_aggregate_on_disk(aggregate_marker):
            if _sweep_harvest_sentinel_path(aggregate_marker).is_file():
                # Sentinel present: a prior tick completed a downloaded ==
                # listed harvest of this epoch's frozen file set. Covers the
                # operator that crashed after the full harvest but before the
                # delete below — converge by deleting now (idempotent on 404).
                logger.info(
                    f"AIPerfSweep {namespace}/{name} aggregate + harvest "
                    f"sentinel already on disk (epoch={epoch}); treating as done"
                )
                await _delete_sweep_jobset(namespace, jobset_name)
                return
            # Back-compat: PVCs harvested by pre-sentinel operator versions
            # have aggregate.json but no sentinel — indistinguishable from the
            # partial-then-unreachable data-loss shape above. The JobSet's
            # existence is the deciding signal: while it exists the emptyDir
            # may still hold files we never pulled, so retry instead of
            # deleting; once it is confirmed gone (404) there is nothing left
            # to recover and the on-disk aggregate is the best remaining copy.
            if not await _sweep_jobset_exists(namespace, jobset_name):
                logger.info(
                    f"AIPerfSweep {namespace}/{name} aggregate on disk without "
                    f"harvest sentinel (epoch={epoch}) and JobSet {jobset_name} "
                    f"is gone; treating pre-sentinel harvest as done"
                )
                return
            raise kopf.TemporaryError(
                f"AIPerfSweep {namespace}/{name} aggregate on disk without "
                f"harvest sentinel and JobSet {jobset_name} still exists; "
                f"retrying harvest instead of deleting",
                delay=30,
            )
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} aggregate sidecar returned no files; retrying",
            delay=30,
        )

    if fetched.is_partial:
        # Some advertised sibling artifacts (children.json, sweep_aggregate/
        # exports, ...) failed to download even though others landed. The only
        # other copy lives on the controller pod's emptyDir, so deleting the
        # JobSet now would destroy the failed files permanently. Keep the
        # JobSet (retry or CR TTL reaps it) and re-harvest on the next tick.
        logger.error(
            f"AIPerfSweep {namespace}/{name} harvest downloaded "
            f"{fetched.downloaded}/{fetched.listed} advertised file(s); keeping "
            f"JobSet aiperf-{name} alive for re-harvest"
        )
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} aggregate harvest partial "
            f"({fetched.downloaded}/{fetched.listed} files downloaded); retrying",
            delay=30,
        )

    if not _sweep_aggregate_on_disk(aggregate_marker):
        # A download reported success without landing a usable aggregate.json
        # (sidecar dying mid-stream, PVC write failure, crash-truncated file).
        # Same reasoning as the partial branch: keep the JobSet alive.
        logger.error(
            f"AIPerfSweep {namespace}/{name} harvest fetched {fetched.downloaded} "
            f"file(s) but {aggregate_marker} is missing or unparsable; keeping "
            f"JobSet aiperf-{name} alive for re-harvest"
        )
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} aggregate harvest incomplete "
            f"(aggregate.json not on disk after fetch); retrying",
            delay=30,
        )

    # Full success: every advertised file landed and the aggregate parses.
    # Record the sentinel BEFORE the delete so a crash between the two still
    # converges on the next tick (zero-download branch above sees it).
    _write_sweep_harvest_sentinel(
        aggregate_marker, downloaded=fetched.downloaded, listed=fetched.listed
    )

    # The aggregate is now on the operator's PVC, so the sweep-controller pod
    # has nothing left to serve. Delete its JobSet to reap the pod promptly —
    # otherwise the pod lingers until the CR's `ttlSecondsAfterFinished`
    # reaper fires, because the controller container exits 0 but the
    # results-sidecar runs uvicorn forever and a Job pod only reaches
    # `Succeeded` once ALL containers terminate. Mirrors the AIPerfJob
    # harvest's `_maybe_delete_jobset_after_success` (delete only after a
    # successful fetch, so we never tear the sidecar down before harvesting).
    await _delete_sweep_jobset(namespace, jobset_name)


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
    # Self-heal a corrupt on-disk index BEFORE open(): open() runs the schema
    # script, which raises "file is not a database" / "disk image is malformed"
    # on a corrupt file and would crash operator startup. integrity_check()
    # opens its own throwaway connection and never raises, so it is the only
    # safe probe here. Guarded on exists() so a first boot (no file yet) skips
    # straight to open(), which creates the parent dir + a fresh DB.
    if db_path.exists() and not await runs_index.integrity_check(db_path):
        logger.warning(
            "runs_index corrupt at %s; renaming aside and rebuilding", db_path
        )
        broken = base / f".aiperf_index.sqlite.broken-{int(time.time())}"
        db_path.rename(broken)
        # Orphan the corrupt DB's WAL/SHM sidecars too: a stale -wal would be
        # replayed against the fresh DB and re-corrupt it.
        for suffix in ("-wal", "-shm"):
            sidecar = db_path.with_name(db_path.name + suffix)
            sidecar.unlink(missing_ok=True)
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
            # Do NOT suggest `aiperf kube index rebuild` here: that CLI hits
            # the results-server container, whose /admin/index/rebuild is
            # mounted with allow_rebuild=False and always returns 503. Only a
            # fresh bootstrap in this (writer) process can rebuild the index.
            logger.exception(
                "runs_index bootstrap task crashed (operator continues without "
                "rebuilt index; restart the operator pod to re-run bootstrap): %s",
                exc,
            )

    bootstrap_task.add_done_callback(_log_bootstrap_exception)


@kopf.on.cleanup()
async def close_runs_index(**_: Any) -> None:
    """Close the runs_index SQLite connection on operator shutdown."""
    await runs_index.close()
