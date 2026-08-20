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
resource watches, and bounded timers (heartbeat_watchdog, cleanup_old_results).

All kopf decorators live here so handler modules stay decorator-free
and are independently testable.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import kopf
import orjson
from kopf._cogs.structs.credentials import ConnectionInfo

from aiperf.kubernetes.client import APISERVER_TLS_SERVER_NAME_OVERRIDE_ENV
from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_GROUP,
    AIPERF_PLURAL,
    AIPERF_SWEEP_API_VERSION,
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

_sweep_results_retention_task: asyncio.Task[None] | None = None


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
    name: str,
    namespace: str,
    uid: str,
    status: dict[str, Any],
    **_: Any,
) -> None:
    """Clean up cached ProgressClient on CR deletion."""
    await lifecycle.on_delete(name=name, namespace=namespace, uid=uid, status=status)


@kopf.on.update(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL, field="spec.cancel")
@track_handler("on_cancel")
async def on_cancel(
    body: dict[str, Any],
    spec: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle cancellation request via spec.cancel field."""
    await lifecycle.on_cancel(
        body=body,
        spec=spec,
        status=status,
        name=name,
        namespace=namespace,
        expected_parent_uid=uid,
        patch=patch,
    )


@kopf.on.update(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    field="spec.timeoutSeconds",
)
@track_handler("on_timeout_update")
async def on_timeout_update(
    body: dict[str, Any],
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Acknowledge a mutable timeout consumed by the monitor timer."""
    await lifecycle.acknowledge_timeout_update(body=body, patch=patch)


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
    uid: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle benchmark completion signal from controller pod."""
    await lifecycle.on_benchmark_complete(
        body=body,
        status=status,
        name=name,
        namespace=namespace,
        expected_parent_uid=uid,
        patch=patch,
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


@kopf.on.update(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_SWEEPS_PLURAL,
    field="spec.ttlSecondsAfterFinished",
)
@track_handler("on_aiperfsweep_ttl_update")
async def on_aiperfsweep_ttl_update(
    body: dict[str, Any],
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Acknowledge a mutable parent TTL consumed by the sweep reaper."""
    await sweep_lifecycle.acknowledge_ttl_update(body=body, patch=patch)


class _StaleSweepCallback(Exception):
    """Stop a delayed callback whose resource name now belongs to another UID."""


async def _sweep_parent_is_current(
    namespace: str,
    name: str,
    sweep_uid: str,
) -> bool:
    """Return whether the live same-name AIPerfSweep still has ``sweep_uid``."""
    import aiohttp
    from kubernetes_asyncio import client
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            current = await client.CustomObjectsApi(api).get_namespaced_custom_object(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                plural=AIPERF_SWEEPS_PLURAL,
                namespace=namespace,
                name=name,
            )
    except ApiException as exc:
        if exc.status == 404:
            return False
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name}: identity read failed "
            f"({exc.status}): {exc.reason}; retrying",
            delay=30,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name}: identity read failed: {exc}; retrying",
            delay=30,
        ) from exc

    current_uid = (current.get("metadata") or {}).get("uid")
    if current_uid is None:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name}: identity read returned no metadata.uid; "
            "retrying",
            delay=30,
        )
    return str(current_uid) == sweep_uid


async def _owned_sweep_jobset_uid(
    namespace: str,
    jobset_name: str,
    *,
    sweep_name: str,
    sweep_uid: str,
) -> str | None:
    """Return the exact owned JobSet UID, ``None`` on 404, or reject replacement."""
    import aiohttp
    from kubernetes_asyncio import client
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            jobset = await client.CustomObjectsApi(api).get_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=namespace,
                name=jobset_name,
            )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise kopf.TemporaryError(
            f"sweep JobSet {namespace}/{jobset_name}: identity read failed "
            f"({exc.status}): {exc.reason}; retrying",
            delay=30,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"sweep JobSet {namespace}/{jobset_name}: identity read failed: "
            f"{exc}; retrying",
            delay=30,
        ) from exc

    metadata = jobset.get("metadata") or {}
    owner_references = metadata.get("ownerReferences") or []
    is_exact_owner = any(
        isinstance(ref, dict)
        and ref.get("apiVersion") == AIPERF_SWEEP_API_VERSION
        and ref.get("kind") == "AIPerfSweep"
        and ref.get("name") == sweep_name
        and ref.get("uid") == sweep_uid
        and ref.get("controller") is True
        for ref in owner_references
    )
    if not is_exact_owner:
        raise _StaleSweepCallback(
            f"JobSet {namespace}/{jobset_name} is not owned by "
            f"AIPerfSweep {sweep_name} uid={sweep_uid}"
        )
    jobset_uid = metadata.get("uid")
    if jobset_uid is None:
        raise kopf.TemporaryError(
            f"sweep JobSet {namespace}/{jobset_name}: identity read returned no "
            "metadata.uid; retrying",
            delay=30,
        )
    return str(jobset_uid)


async def _delete_sweep_jobset(
    namespace: str,
    jobset_name: str,
    *,
    sweep_name: str,
    sweep_uid: str,
) -> None:
    """Delete only the exact JobSet owned by the harvesting sweep identity."""
    import aiohttp
    from kubernetes_asyncio import client
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        jobset_uid = await _owned_sweep_jobset_uid(
            namespace,
            jobset_name,
            sweep_name=sweep_name,
            sweep_uid=sweep_uid,
        )
    except _StaleSweepCallback as exc:
        logger.info(f"Skipping stale sweep JobSet delete: {exc}")
        return
    if jobset_uid is None:
        return

    try:
        async with k8s_client() as api:
            await client.CustomObjectsApi(api).delete_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=namespace,
                name=jobset_name,
                body=client.V1DeleteOptions(
                    preconditions=client.V1Preconditions(uid=jobset_uid)
                ),
            )
        logger.info(f"Deleted sweep JobSet {namespace}/{jobset_name} after harvest")
    except ApiException as exc:
        if exc.status in (404, 409):
            return
        raise kopf.TemporaryError(
            f"sweep JobSet {namespace}/{jobset_name}: delete failed "
            f"({exc.status}): {exc.reason}; retrying",
            delay=30,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"sweep JobSet {namespace}/{jobset_name}: delete failed: {exc}; retrying",
            delay=30,
        ) from exc


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

    Written ONLY from the full-success path, immediately before ``latest.txt``
    and the JobSet delete, so an operator that crashes between them converges
    on the next tick. A write failure aborts the commit and preserves the
    controller pod's emptyDir for a retry.
    """
    sentinel = _sweep_harvest_sentinel_path(aggregate_path)
    tmp = sentinel.with_suffix(".tmp")
    try:
        tmp.write_bytes(
            orjson.dumps(
                {"harvestComplete": True, "downloaded": downloaded, "listed": listed}
            )
        )
        os.replace(tmp, sentinel)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _finalize_sweep_archive(
    *,
    base_dir: Path,
    namespace: str,
    name: str,
    epoch: str,
    status: dict[str, Any],
) -> None:
    """Atomically add the operator-owned status metadata to a harvested archive."""
    from aiperf.sweep_controller.aggregator import write_sweep_aggregate

    epoch_dir = base_dir / namespace / "sweeps" / name / epoch
    aggregate_path = epoch_dir / "aggregate.json"
    doc = orjson.loads(aggregate_path.read_bytes())
    for key in (
        "startedAt",
        "completedAt",
        "completionTime",
        "apiUrl",
        "aggregateRef",
        "maxTotalRuns",
        "observedGeneration",
        "resultsAvailable",
    ):
        value = status.get(key)
        if value is not None:
            doc[key] = value
    if not isinstance(doc.get("totalVariations"), int):
        doc["totalVariations"] = int(status.get("totalVariations") or 0)
    conditions = status.get("conditions")
    write_sweep_aggregate(
        base_dir=base_dir,
        namespace=namespace,
        sweep_name=name,
        sweep_run_epoch=epoch,
        doc=doc,
        conditions=list(conditions) if isinstance(conditions, list) else [],
        update_latest=False,
    )


async def _commit_sweep_archive(
    *,
    base_dir: Path,
    namespace: str,
    name: str,
    epoch: str,
    durable_ref: dict[str, Any] | None = None,
    sweep_uid: str | None = None,
) -> bool:
    """Materialize lineage, fence publication, then advance archive discovery."""
    from aiperf.operator.handlers.sweep import _aggregate_fetch

    try:
        _materialize_sweep_child_lineage(
            base_dir=base_dir,
            namespace=namespace,
            name=name,
            epoch=epoch,
        )
    except (OSError, orjson.JSONDecodeError, TypeError, ValueError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} epoch={epoch}: child-lineage "
            f"commit failed ({type(exc).__name__}: {exc}); retrying before "
            "the sweep-controller JobSet is deleted",
            delay=30,
        ) from exc

    if durable_ref is not None or sweep_uid is not None:
        if durable_ref is None or sweep_uid is None:
            raise ValueError("durable_ref and sweep_uid must be provided together")
        published = await _publish_durable_sweep_aggregate_ref(
            namespace,
            name,
            durable_ref,
            sweep_uid,
        )
        if not published:
            return False

    try:
        _aggregate_fetch._write_sweep_latest_pointer(base_dir, namespace, name, epoch)
    except (OSError, ValueError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} epoch={epoch}: latest-pointer "
            f"commit failed ({type(exc).__name__}: {exc}); retrying before "
            "the sweep-controller JobSet is deleted",
            delay=30,
        ) from exc

    epoch_dir = base_dir / namespace / "sweeps" / name / epoch
    try:
        await runs_index._index_sweep_from_disk(namespace, name, epoch, epoch_dir)
    except Exception as exc:  # noqa: BLE001 - the index is a rebuildable cache
        logger.warning(
            f"AIPerfSweep {namespace}/{name} epoch={epoch}: runs-index ingest "
            f"failed ({type(exc).__name__}: {exc}); durable files remain authoritative"
        )
    return True


def _materialize_sweep_child_lineage(
    *,
    base_dir: Path,
    namespace: str,
    name: str,
    epoch: str,
) -> int:
    """Recreate child ``sweep.json`` markers from the durable parent manifest."""
    from aiperf.sweep_controller.k8s_executor import write_child_sweep_marker

    children_path = base_dir / namespace / "sweeps" / name / epoch / "children.json"
    try:
        doc = orjson.loads(children_path.read_bytes())
    except FileNotFoundError as exc:
        raise ValueError(f"missing sweep children manifest: {children_path}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("children"), list):
        raise ValueError(f"invalid sweep children manifest: {children_path}")

    count = 0
    for child in doc["children"]:
        if not isinstance(child, dict):
            raise ValueError(f"invalid child entry in sweep manifest: {children_path}")
        child_name = child.get("name")
        if not isinstance(child_name, str) or not child_name:
            raise ValueError(
                f"child entry has no name in sweep manifest: {children_path}"
            )
        child_namespace = child.get("namespace") or namespace
        variation_index = int(child["variation_index"])
        trial_value = child.get("trial_index")
        trial_index = int(trial_value) if trial_value is not None else None
        write_child_sweep_marker(
            base_dir=base_dir,
            namespace=str(child_namespace),
            child_name=child_name,
            sweep_name=name,
            variation_index=variation_index,
            variation_label=str(child.get("variation_label") or ""),
            trial_index=trial_index,
            sweep_run_epoch=epoch,
            child_run_epoch=str(child.get("child_run_epoch") or epoch),
        )
        count += 1
    return count


def _durable_sweep_aggregate_ref(
    namespace: str,
    name: str,
    epoch: str,
) -> dict[str, Any]:
    """Build an aggregate reference served by the durable operator API."""
    from aiperf.operator.environment import OperatorEnvironment

    base_url = OperatorEnvironment.SERVICE.BASE_URL.rstrip("/")
    parsed = urlsplit(base_url)
    host = parsed.hostname or parsed.path.split(":", 1)[0]
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    api_path = (
        f"/api/v1/sweeps/{namespace}/{name}/epochs/{epoch}/artifacts/aggregate.json"
    )
    return {
        "resultsServerHost": host,
        "port": port,
        "apiPath": api_path,
        "url": f"{base_url}{api_path}",
    }


async def _publish_durable_sweep_aggregate_ref(
    namespace: str,
    name: str,
    aggregate_ref: dict[str, Any],
    sweep_uid: str,
) -> bool:
    """UID-fenced publication of the operator-backed aggregate reference."""
    import aiohttp
    from kubernetes_asyncio import client
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            await client.CustomObjectsApi(api).patch_namespaced_custom_object_status(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                plural=AIPERF_SWEEPS_PLURAL,
                namespace=namespace,
                name=name,
                body=[
                    {
                        "op": "test",
                        "path": "/metadata/uid",
                        "value": sweep_uid,
                    },
                    {
                        "op": "add",
                        "path": "/status/aggregateRef",
                        "value": aggregate_ref,
                    },
                    {
                        "op": "add",
                        "path": "/status/resultsAvailable",
                        "value": True,
                    },
                ],
                field_manager="aiperf-operator",
                _content_type="application/json-patch+json",
            )
    except ApiException as exc:
        if exc.status == 404:
            logger.info(
                f"Skipping stale aggregate publication for AIPerfSweep "
                f"{namespace}/{name} uid={sweep_uid}: {exc.status} {exc.reason}"
            )
            return False
        if exc.status == 422 and not await _sweep_parent_is_current(
            namespace,
            name,
            sweep_uid,
        ):
            logger.info(
                f"Skipping stale aggregate publication for AIPerfSweep "
                f"{namespace}/{name} uid={sweep_uid}: UID test rejected"
            )
            return False
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name}: durable aggregate reference "
            f"publication failed ({exc.status}: {exc.reason}); retrying",
            delay=30,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name}: durable aggregate reference "
            f"publication failed ({type(exc).__name__}: {exc}); retrying",
            delay=30,
        ) from exc
    return True


async def _sweep_jobset_exists(
    namespace: str,
    jobset_name: str,
    *,
    sweep_name: str,
    sweep_uid: str,
) -> bool:
    """Whether the exact sweep-owned JobSet is still on the apiserver.

    A same-name JobSet owned by another AIPerfSweep UID raises the internal
    stale-callback signal. Transient reads retry rather than being mistaken
    for proof that the source emptyDir is gone.
    """
    return (
        await _owned_sweep_jobset_uid(
            namespace,
            jobset_name,
            sweep_name=sweep_name,
            sweep_uid=sweep_uid,
        )
        is not None
    )


async def _commit_existing_sweep_archive(
    *,
    base_dir: Path,
    namespace: str,
    name: str,
    epoch: str,
    durable_ref: dict[str, Any],
    jobset_name: str,
    sweep_uid: str,
    delete_jobset: bool,
) -> bool:
    """Commit an existing sweep archive, publish its ref, and optionally reap."""
    committed = await _commit_sweep_archive(
        base_dir=base_dir,
        namespace=namespace,
        name=name,
        epoch=epoch,
        durable_ref=durable_ref,
        sweep_uid=sweep_uid,
    )
    if not committed:
        return False
    if delete_jobset:
        await _delete_sweep_jobset(
            namespace,
            jobset_name,
            sweep_name=name,
            sweep_uid=sweep_uid,
        )
    return True


async def _recover_pre_sentinel_sweep_archive(
    *,
    base_dir: Path,
    namespace: str,
    name: str,
    epoch: str,
    status: dict[str, Any],
    durable_ref: dict[str, Any],
    aggregate_marker: Path,
    jobset_name: str,
    sweep_uid: str,
) -> None:
    """Finalize an archive written before harvest sentinels were introduced."""
    try:
        _finalize_sweep_archive(
            base_dir=base_dir,
            namespace=namespace,
            name=name,
            epoch=epoch,
            status={
                **status,
                "aggregateRef": durable_ref,
                "resultsAvailable": True,
            },
        )
        _write_sweep_harvest_sentinel(
            aggregate_marker,
            downloaded=0,
            listed=0,
        )
    except (OSError, orjson.JSONDecodeError, TypeError, ValueError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} pre-sentinel archive "
            f"commit failed ({type(exc).__name__}: {exc}); retrying",
            delay=30,
        ) from exc
    await _commit_existing_sweep_archive(
        base_dir=base_dir,
        namespace=namespace,
        name=name,
        epoch=epoch,
        durable_ref=durable_ref,
        jobset_name=jobset_name,
        sweep_uid=sweep_uid,
        delete_jobset=False,
    )


async def _handle_zero_download_sweep_harvest(
    *,
    listed: int,
    base_dir: Path,
    namespace: str,
    name: str,
    epoch: str,
    status: dict[str, Any],
    durable_ref: dict[str, Any],
    aggregate_marker: Path,
    jobset_name: str,
    sweep_uid: str,
) -> None:
    """Resolve a zero-file harvest as retry, prior success, or legacy archive."""
    if listed > 0:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} sidecar listed {listed} "
            f"file(s) but none downloaded; retrying",
            delay=30,
        )
    if not _sweep_aggregate_on_disk(aggregate_marker):
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} aggregate sidecar returned no files; retrying",
            delay=30,
        )
    if _sweep_harvest_sentinel_path(aggregate_marker).is_file():
        logger.info(
            f"AIPerfSweep {namespace}/{name} aggregate + harvest sentinel "
            f"already on disk (epoch={epoch}); treating as done"
        )
        await _commit_existing_sweep_archive(
            base_dir=base_dir,
            namespace=namespace,
            name=name,
            epoch=epoch,
            durable_ref=durable_ref,
            jobset_name=jobset_name,
            sweep_uid=sweep_uid,
            delete_jobset=True,
        )
        return
    try:
        jobset_exists = await _sweep_jobset_exists(
            namespace,
            jobset_name,
            sweep_name=name,
            sweep_uid=sweep_uid,
        )
    except _StaleSweepCallback as exc:
        logger.info(f"Skipping stale sweep harvest callback: {exc}")
        return
    if jobset_exists:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} aggregate on disk without "
            f"harvest sentinel and JobSet {jobset_name} still exists; "
            f"retrying harvest instead of deleting",
            delay=30,
        )
    logger.info(
        f"AIPerfSweep {namespace}/{name} aggregate on disk without harvest "
        f"sentinel (epoch={epoch}) and JobSet {jobset_name} is gone; treating "
        "pre-sentinel harvest as done"
    )
    await _recover_pre_sentinel_sweep_archive(
        base_dir=base_dir,
        namespace=namespace,
        name=name,
        epoch=epoch,
        status=status,
        durable_ref=durable_ref,
        aggregate_marker=aggregate_marker,
        jobset_name=jobset_name,
        sweep_uid=sweep_uid,
    )


async def _resolve_sweep_harvest_identity(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
) -> tuple[str, str] | None:
    """Resolve and verify the immutable parent and source-JobSet identities."""
    aggregation = status.get("aggregation") or {}
    if aggregation.get("phase") != "Complete":
        return None
    epoch = status.get("runEpoch")
    if epoch is None:
        logger.warning(
            f"AIPerfSweep {namespace}/{name} aggregation Complete with no "
            f"status.runEpoch; skipping disk persistence"
        )
        return None
    sweep_uid = str((body.get("metadata") or {}).get("uid") or "")
    if not sweep_uid:
        logger.warning(
            f"AIPerfSweep {namespace}/{name} aggregation callback has no "
            "metadata.uid; skipping unsafe delayed harvest"
        )
        return None
    if not await _sweep_parent_is_current(namespace, name, sweep_uid):
        logger.info(
            f"Skipping stale aggregate harvest for AIPerfSweep {namespace}/{name} "
            f"uid={sweep_uid}"
        )
        return None
    try:
        await _owned_sweep_jobset_uid(
            namespace,
            f"aiperf-{name}",
            sweep_name=name,
            sweep_uid=sweep_uid,
        )
    except _StaleSweepCallback as exc:
        logger.info(f"Skipping stale sweep harvest callback: {exc}")
        return None
    return str(epoch), sweep_uid


@kopf.on.resume(AIPERF_GROUP, AIPERF_VERSION, AIPERF_SWEEPS_PLURAL)
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

    The sweep-controller writes the canonical
    ``/results/<namespace>/sweeps/<name>/<epoch>/...`` bundle and the
    ``.aiperf_results_ready.json`` marker; the sidecar then serves it
    over HTTP. The pod uses ``emptyDir{}`` for ``/results`` (per the
    no-PVC-on-controller-pods constraint), so the operator MUST pull the
    artifacts to its own results PVC at
    ``<base>/<ns>/sweeps/<name>/<runEpoch>/`` before the JobSet is
    deleted. Without this, ``getSweepEpochs`` and ``getSweepCells`` find
    nothing on disk and the SweepDetail page renders as empty cells.
    """
    from aiperf.operator.environment import OperatorEnvironment
    from aiperf.operator.handlers.sweep import _aggregate_fetch

    status = status or {}
    identity = await _resolve_sweep_harvest_identity(
        body=body,
        status=status,
        name=name,
        namespace=namespace,
    )
    if identity is None:
        return
    epoch, sweep_uid = identity
    base_dir = OperatorEnvironment.RESULTS.DIR
    durable_ref = _durable_sweep_aggregate_ref(namespace, name, epoch)
    aggregate_marker = base_dir / namespace / "sweeps" / name / epoch / "aggregate.json"
    jobset_name = f"aiperf-{name}"
    aggregate_ref = status.get("aggregateRef") or {}
    if (
        status.get("resultsAvailable") is True
        and isinstance(aggregate_ref, dict)
        and bool(aggregate_ref.get("url"))
        and _sweep_aggregate_on_disk(aggregate_marker)
    ):
        # Resume after status publication but before latest/index commit or
        # source JobSet reap. The UID-fenced commit path is idempotent.
        await _commit_existing_sweep_archive(
            base_dir=base_dir,
            namespace=namespace,
            name=name,
            epoch=epoch,
            durable_ref=durable_ref,
            jobset_name=jobset_name,
            sweep_uid=sweep_uid,
            delete_jobset=True,
        )
        return
    fetched = await _aggregate_fetch.fetch_sweep_aggregate_to_disk(
        sweep_name=name,
        namespace=namespace,
        epoch=epoch,
        base_dir=base_dir,
    )
    if fetched.downloaded == 0:
        await _handle_zero_download_sweep_harvest(
            listed=fetched.listed,
            base_dir=base_dir,
            namespace=namespace,
            name=name,
            epoch=epoch,
            status=status or {},
            durable_ref=durable_ref,
            aggregate_marker=aggregate_marker,
            jobset_name=jobset_name,
            sweep_uid=sweep_uid,
        )
        return

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
    # Complete the operator-owned archive metadata before creating the commit
    # sentinel. A failure retains the JobSet so the next reconcile can retry
    # while the controller's emptyDir still exists.
    try:
        _finalize_sweep_archive(
            base_dir=base_dir,
            namespace=namespace,
            name=name,
            epoch=epoch,
            status={
                **(status or {}),
                "aggregateRef": durable_ref,
                "resultsAvailable": True,
            },
        )
        _write_sweep_harvest_sentinel(
            aggregate_marker, downloaded=fetched.downloaded, listed=fetched.listed
        )
    except (OSError, orjson.JSONDecodeError, TypeError, ValueError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{name} archive commit failed "
            f"({type(exc).__name__}: {exc}); retrying",
            delay=30,
        ) from exc

    committed = await _commit_sweep_archive(
        base_dir=base_dir,
        namespace=namespace,
        name=name,
        epoch=epoch,
        durable_ref=durable_ref,
        sweep_uid=sweep_uid,
    )
    if not committed:
        return

    # The aggregate is now on the operator's PVC, so the sweep-controller pod
    # has nothing left to serve. Delete its JobSet to reap the pod promptly —
    # otherwise the pod lingers until the CR's `ttlSecondsAfterFinished`
    # reaper fires, because the controller container exits 0 but the
    # results-sidecar runs uvicorn forever and a Job pod only reaches
    # `Succeeded` once ALL containers terminate. Mirrors the AIPerfJob
    # harvest's `_maybe_delete_jobset_after_success` (delete only after a
    # successful fetch, so we never tear the sidecar down before harvesting).
    await _delete_sweep_jobset(
        namespace,
        jobset_name,
        sweep_name=name,
        sweep_uid=sweep_uid,
    )


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


@kopf.on.field(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL, field="status.subPhase")
@track_handler("on_controller_subphase")
async def on_controller_subphase(
    *,
    new: str | None,
    body: dict[str, Any],
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """Project controller lifecycle pushes onto the coarse operator phase."""
    await monitor.handle_controller_subphase_event(
        body=body,
        new=new,
        namespace=namespace,
        name=name,
    )


@kopf.on.field(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    field="status.controllerFailure",
)
@track_handler("on_controller_failure")
async def on_controller_failure(
    *,
    body: dict[str, Any],
    new: str | None,
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """Fail a job from its controller's direct status push."""
    await monitor.handle_controller_failure_event(
        body=body,
        new=new,
        namespace=namespace,
        name=name,
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
    await monitor.handle_pod_recovery_event(
        body=body,
        meta=meta,
        namespace=namespace,
        name=name,
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
    """Handle JobSet terminal conditions without bypassing result readiness.

    AIPerfJob completion remains controller-driven, because a JobSet's terminal
    condition does not prove durable artifact export. Failed conditions dispatch
    to identity-fenced AIPerfJob or AIPerfSweep recovery.
    """
    await jobset_terminal_handler.handle_jobset_conditions(
        old=old, new=new, namespace=namespace, jobset_name=name, jobset_body=body
    )


@kopf.on.field(
    JOBSET_GROUP,
    JOBSET_VERSION,
    JOBSET_PLURAL,
    field="status.replicatedJobsStatus",
)
@track_handler("on_jobset_replicated_jobs_status")
async def on_jobset_replicated_jobs_status(
    *,
    namespace: str,
    name: str,
    body: dict[str, Any],
    **_: Any,
) -> None:
    """Project exact JobSet worker readiness without a broad monitor tick."""
    await jobset_terminal_handler.handle_jobset_progress(
        namespace=namespace,
        jobset_name=name,
        jobset_body=body,
    )


@kopf.on.delete(AIPERF_GROUP, AIPERF_VERSION, AIPERF_SWEEPS_PLURAL)
@track_handler("on_aiperfsweep_delete")
async def on_aiperfsweep_delete(
    *,
    body: dict[str, Any],
    uid: str,
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """On AIPerfSweep deletion, request cooperative cancellation of any running children.

    OwnerReferences will cascade-GC child AIPerfJobs and the sweep-controller
    JobSet, but cooperative cancel lets in-flight benchmarks shut down
    cleanly (write-out partial results, signal workers) before the cascade
    SIGKILLs them.
    """
    await sweep_lifecycle.on_delete(
        body=body,
        uid=uid,
        name=name,
        namespace=namespace,
    )


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_SWEEPS_PLURAL,
    interval=OperatorEnvironment.MONITOR.INTERVAL,
    initial_delay=OperatorEnvironment.MONITOR.INITIAL_DELAY,
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
@track_handler("startup_issue_deadline")
async def startup_issue_deadline(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Age only an already-observed startup blocker from cached CR status."""
    await monitor.startup_issue_deadline(
        body=body,
        status=status,
        name=name,
        namespace=namespace,
        patch=patch,
    )


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    interval=OperatorEnvironment.MONITOR.INTERVAL,
    initial_delay=OperatorEnvironment.MONITOR.INITIAL_DELAY,
)
@track_handler("heartbeat_watchdog")
async def heartbeat_watchdog(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Run recovery only when controller heartbeat expires or timeout is due."""
    await monitor.heartbeat_watchdog(
        body=body, status=status, spec=spec, name=name, namespace=namespace, patch=patch
    )


async def monitor_progress(
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Undecorated compatibility entry point for the broad recovery engine."""
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
            # The results-server mounts /admin/index/rebuild with
            # allow_rebuild=False, so only a fresh bootstrap in this (writer)
            # process can rebuild the index -- hence the restart hint below.
            logger.exception(
                "runs_index bootstrap task crashed (operator continues without "
                "rebuilt index; restart the operator pod to re-run bootstrap): %s",
                exc,
            )

    bootstrap_task.add_done_callback(_log_bootstrap_exception)

    global _sweep_results_retention_task
    if _sweep_results_retention_task is None or _sweep_results_retention_task.done():
        _sweep_results_retention_task = asyncio.create_task(
            _run_sweep_results_retention(base, bootstrap_task)
        )


async def _run_sweep_results_retention(
    base_dir: Path,
    bootstrap_task: asyncio.Task[Any],
) -> None:
    """Reconcile durable sweep TTLs after bootstrap and once per day."""
    with contextlib.suppress(Exception):
        await bootstrap_task

    while True:
        try:
            await cleanup.reconcile_sweep_results(base_dir=base_dir)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - retention must survive one bad PVC entry
            logger.exception("sweep results retention pass failed; retrying next cycle")
        try:
            await asyncio.wait_for(
                asyncio.Event().wait(),
                timeout=cleanup.SWEEP_RESULTS_CLEANUP_INTERVAL_SECONDS,
            )
        except TimeoutError:
            continue


@kopf.on.cleanup()
async def close_runs_index(**_: Any) -> None:
    """Close the runs_index SQLite connection on operator shutdown."""
    global _sweep_results_retention_task
    if _sweep_results_retention_task is not None:
        _sweep_results_retention_task.cancel()
        await asyncio.gather(_sweep_results_retention_task, return_exceptions=True)
        _sweep_results_retention_task = None
    await runs_index.close()
