# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API router for live Kubernetes job and cluster state."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import orjson
from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Depends as DependsParam
from fastapi.responses import Response
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.results_markers import EPOCH_RE
from aiperf.kubernetes.client import (
    cancel_aiperf_job,
    cluster_version,
    get_pods,
    get_raw_aiperfjob,
    get_raw_aiperfjob_status,
    list_events_for_object,
    list_nodes,
    list_pods_all_namespaces,
)
from aiperf.operator import runs_index
from aiperf.operator.job_union import (
    _read_summary,
    _summary_path,
    find_any_job,
    list_all_jobs,
    synthesize_status_from_summary,
)
from aiperf.operator.results_layout import (
    RunEntry,
    job_dir,
    list_runs_async,
    resolve_run_dir,
)
from aiperf.operator.routers._etag import etag_response
from aiperf.operator.routers._path_params import validate_results_path_params
from aiperf.operator.routers.jobs_logs import get_pod_logs_impl
from aiperf.operator.routers.jobs_models import (
    ActiveJobListResponse,
    CancelResponse,
    ClusterResponse,
    CreateJobRequest,
    CreateJobResponse,
    EventEntry,
    EventInvolvedObject,
    EventSource,
    JobDetailResponse,
    JobEpochsResponse,
    JobEpochSummary,
    JobEventsResponse,
    JobPodSummary,
)
from aiperf.operator.runs_index_models import RunIndexRow

if TYPE_CHECKING:
    from kubernetes_asyncio.client.models import V1Node, V1Pod

logger = logging.getLogger("aiperf.operator.ui")


def _pod_summary(pod: V1Pod) -> JobPodSummary:
    """Extract pod name, phase, readiness, and restart count for the UI."""
    meta = pod.metadata
    status = pod.status
    spec = getattr(pod, "spec", None)
    container_statuses = (status.container_statuses or []) if status else []
    spec_containers = (spec.containers or []) if spec else []
    return JobPodSummary(
        name=(meta.name if meta else "") or "",
        phase=(status.phase if status else None) or "Unknown",
        ready=any(bool(c.ready) for c in container_statuses),
        restarts=sum(int(c.restart_count or 0) for c in container_statuses),
        containers=[c.name for c in spec_containers if getattr(c, "name", None)],
    )


def _node_gpu_count(node: V1Node) -> int:
    """Return the number of nvidia.com/gpu resources allocatable on a node."""
    alloc = (node.status.allocatable or {}) if node.status else {}
    try:
        return int(alloc.get("nvidia.com/gpu", 0))
    except (TypeError, ValueError):
        return 0


async def _fetch_k8s_version(api: ApiClient) -> str:
    """Return the cluster gitVersion, or 'unknown' if the call fails."""
    try:
        version_info = await cluster_version(api)
    except Exception:  # noqa: BLE001 - best-effort; UI tolerates 'unknown'
        return "unknown"
    return version_info.get("gitVersion", "unknown")


def _parse_kubeadm_cluster_name(cluster_yaml: str) -> str | None:
    """Extract ``clusterName`` from a kubeadm ``ClusterConfiguration`` doc.

    Cheap line-scan for `clusterName: <value>` — avoids pulling a
    YAML dep just for this single field. The kubeadm doc keeps it
    at top level so a simple prefix match is correct.
    """
    for line in cluster_yaml.splitlines():
        stripped = line.strip()
        if stripped.startswith("clusterName:"):
            name = stripped.split(":", 1)[1].strip().strip("'\"")
            if name and name not in {"kubernetes", ""}:
                return name
    return None


def _parse_kubeconfig_cluster_name(kubeconfig_blob: str) -> str | None:
    """Extract the first cluster ``name:`` from an embedded kubeconfig blob.

    Same strategy as :func:`_parse_kubeadm_cluster_name` — line-scan for
    the first `name:` under `clusters:`. Good enough for the simple
    kubeconfig format the cluster-info CM typically embeds.
    """
    in_clusters_section = False
    for line in kubeconfig_blob.splitlines():
        stripped = line.strip()
        if stripped == "clusters:":
            in_clusters_section = True
            continue
        if in_clusters_section and stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip().strip("'\"")
            if name and name not in {"kubernetes", ""}:
                return name
        if (
            in_clusters_section
            and stripped
            and not stripped.startswith("-")
            and not stripped.startswith(
                ("name:", "cluster:", "server:", "certificate", "insecure")
            )
        ):
            in_clusters_section = False
    return None


async def _cluster_name_from_kubeadm_config(api: ApiClient) -> str | None:
    """Read ``kube-system/kubeadm-config`` and extract ``clusterName``, or None."""
    try:
        core = client.CoreV1Api(api)
        cm = await core.read_namespaced_config_map(
            name="kubeadm-config", namespace="kube-system"
        )
    except Exception:  # noqa: BLE001 - best-effort; try the next source
        return None
    data = getattr(cm, "data", None) or {}
    cluster_yaml = data.get("ClusterConfiguration") or data.get("ClusterStatus")
    if isinstance(cluster_yaml, str):
        return _parse_kubeadm_cluster_name(cluster_yaml)
    return None


async def _cluster_name_from_cluster_info(api: ApiClient) -> str | None:
    """Read ``kube-system/cluster-info`` and extract a cluster label, or None."""
    try:
        core = client.CoreV1Api(api)
        cm = await core.read_namespaced_config_map(
            name="cluster-info", namespace="kube-system"
        )
    except Exception:  # noqa: BLE001 - best-effort
        return None
    data = getattr(cm, "data", None) or {}
    return _parse_kubeconfig_cluster_name(data.get("kubeconfig") or "")


async def _fetch_cluster_name(api: ApiClient) -> str | None:
    """Auto-detect the cluster name from the apiserver, or None.

    Kubernetes core has no canonical "cluster name" field, but two
    well-known sources cover most installs:

    * ``kube-system/kubeadm-config`` ConfigMap — written by kubeadm
      (and tools that wrap it: kind, kops, kubespray, most on-prem and
      DGX-style clusters). The ``ClusterConfiguration`` doc embedded in
      its ``data`` block has a ``clusterName`` field.
    * ``kube-system/cluster-info`` ConfigMap — public bootstrap info on
      some installs; the embedded kubeconfig blob's ``clusters[0].name``
      gives a usable label.

    Best-effort: any RBAC denial / parse failure falls back to ``None``,
    which lets the UI show the Kubernetes version instead. Operators
    can override via ``AIPERF_OPERATOR_CLUSTER_NAME`` for clusters that
    don't expose either of these (managed GKE/EKS, custom installers).
    """
    name = await _cluster_name_from_kubeadm_config(api)
    if name is not None:
        return name
    return await _cluster_name_from_cluster_info(api)


def _pod_gpu_request(pod: V1Pod) -> int:
    """Sum nvidia.com/gpu requests across every container in a pod.

    Mirrors the gpu-report.sh accounting: only ``requests`` are summed
    (not ``limits``), and any non-integer / missing value is treated as 0.
    """
    spec = pod.spec
    if spec is None:
        return 0
    total = 0
    for container in spec.containers or []:
        resources = getattr(container, "resources", None)
        requests = getattr(resources, "requests", None) or {}
        raw = requests.get("nvidia.com/gpu", 0)
        try:
            total += int(raw)
        except (TypeError, ValueError):
            continue
    return total


async def _list_nodes_safe(api: ApiClient) -> list[V1Node]:
    """List cluster nodes, absorbing and logging every Kubernetes error."""
    try:
        return await list_nodes(api)
    except ApiException as e:
        if (e.status or 0) == 403:
            logger.error(
                "Cluster node listing forbidden (403) — check that the "
                "operator ClusterRole grants `nodes get/list`: %s",
                e,
            )
        else:
            logger.warning("Failed to query nodes (apiserver %s): %s", e.status, e)
        return []
    except Exception as e:  # noqa: BLE001 - UI tolerates missing cluster-wide query
        logger.warning(f"Failed to query nodes: {e}")
        return []


async def _list_pods_safe(api: ApiClient) -> list[V1Pod]:
    """List pods in all namespaces, absorbing and logging every Kubernetes error."""
    try:
        return await list_pods_all_namespaces(api)
    except ApiException as e:
        if (e.status or 0) == 403:
            logger.error(
                "Cluster pod listing forbidden (403) — check that the "
                "operator ClusterRole grants `pods get/list`: %s",
                e,
            )
        else:
            logger.warning("Failed to query pods (apiserver %s): %s", e.status, e)
        return []
    except Exception as e:  # noqa: BLE001 - best-effort
        logger.warning(f"Failed to query pods cluster-wide: {e}")
        return []


def _gpu_capacity_by_node(nodes: list[V1Node]) -> dict[str, int]:
    """Map node-name -> allocatable GPUs (only nodes that actually have GPUs)."""
    node_capacity: dict[str, int] = {}
    for node in nodes:
        name = node.metadata.name if node.metadata else None
        if not name:
            continue
        gpus = _node_gpu_count(node)
        if gpus > 0:
            node_capacity[name] = gpus
    return node_capacity


def _gpu_usage_by_node(
    pods: list[V1Pod], node_capacity: dict[str, int]
) -> dict[str, int]:
    """Sum GPU requests per GPU node across Running/Pending pods."""
    used_per_node: dict[str, int] = {}
    for pod in pods:
        phase = (pod.status.phase if pod.status else None) or ""
        if phase not in ("Running", "Pending"):
            continue
        node_name = pod.spec.node_name if pod.spec else None
        if not node_name or node_name not in node_capacity:
            continue
        req = _pod_gpu_request(pod)
        if req > 0:
            used_per_node[node_name] = used_per_node.get(node_name, 0) + req
    return used_per_node


async def _fetch_cluster_gpu_stats(api: ApiClient) -> dict[str, Any]:
    """Compute cluster-wide GPU capacity, usage, and node-state breakdown.

    Combines :func:`list_nodes` (allocatable totals per node) with
    :func:`list_pods_all_namespaces` (pod-level GPU requests) to produce
    the same headline numbers as ``~/gpu-report.sh``: total/used/free
    GPUs, utilization %, and the count of nodes that are completely
    free / partially used / fully used.

    All Kubernetes errors are absorbed and logged — this endpoint is
    supplementary UI context, not a critical path. The response always
    has every key present so the JS side can rely on the schema.
    """
    nodes = await _list_nodes_safe(api)
    node_capacity = _gpu_capacity_by_node(nodes)

    used_per_node: dict[str, int] = {}
    if node_capacity:
        pods = await _list_pods_safe(api)
        used_per_node = _gpu_usage_by_node(pods, node_capacity)

    total_gpus = sum(node_capacity.values())
    total_used = sum(
        min(used_per_node.get(n, 0), cap) for n, cap in node_capacity.items()
    )
    total_free = max(total_gpus - total_used, 0)
    utilization = round(100.0 * total_used / total_gpus, 1) if total_gpus > 0 else 0.0

    nodes_free = sum(
        1 for n, cap in node_capacity.items() if used_per_node.get(n, 0) == 0
    )
    nodes_full = sum(
        1 for n, cap in node_capacity.items() if used_per_node.get(n, 0) >= cap
    )
    nodes_partial = len(node_capacity) - nodes_free - nodes_full

    return {
        "node_count": len(nodes),
        "gpu_nodes": len(node_capacity),
        "gpus": total_gpus,
        "gpus_used": total_used,
        "gpus_free": total_free,
        "utilization_percent": utilization,
        "nodes_free": nodes_free,
        "nodes_partial": nodes_partial,
        "nodes_full": nodes_full,
    }


async def _list_jobs_impl(api: ApiClient, results_dir: Path) -> ActiveJobListResponse:
    """Body of GET /api/v1/jobs: union of active CRs + archived PVC directories.

    Returns the unified view from :func:`aiperf.operator.job_union.list_all_jobs`:
    live CRs (``source="live"``), PVC-only historical runs (``source="archived"``),
    and CRs that also have a persisted summary (``source="both"``). Keyed by
    ``(namespace, name)``; overlap entries prefer CR values on live fields and
    backfill from PVC on historical-only fields.

    Raises:
        HTTPException: Any non-404 ``kubernetes_asyncio.client.ApiException``
            status code from the CR half is surfaced verbatim (e.g. 401/403 on
            RBAC denial). The PVC half is tolerant and falls back to an empty
            list on filesystem errors.
    """
    jobs = await list_all_jobs(api, results_dir, all_namespaces=True)
    return ActiveJobListResponse(jobs=[j.model_dump(by_alias=True) for j in jobs])


async def _get_job_impl(
    api: ApiClient,
    results_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> JobDetailResponse:
    """Body of GET /api/v1/jobs/{namespace}/{name}: fetch a CR plus its pod roster.

    Returns three things joined into one response: (1) the AIPerfJob summary
    (same shape as ``list_jobs``), (2) the raw CR ``.status`` subresource
    (phase, conditions, liveMetrics), and (3) the current pod list filtered by
    the ``aiperf.nvidia.com/job-id=<name>`` label selector.

    Archived (PVC-only) jobs have no cluster CR, so the response returns an
    empty ``status`` dict and empty ``pods`` list alongside the archived job
    summary.

    When ``epoch`` is supplied, the archived half is pinned to that historical
    run directory rather than ``latest.txt``; ``find_any_job`` likewise refuses
    to merge the live CR onto a historical epoch (see Task 6).

    Args:
        api: The kubernetes_asyncio ApiClient.
        results_dir: Base directory on the results PVC.
        namespace: Kubernetes namespace containing the AIPerfJob CR or PVC dir.
        name: Name of the AIPerfJob CR (also the label value matched when
            listing pods, and the PVC subdirectory name).
        epoch: Optional decimal-seconds epoch (matching :data:`EPOCH_RE`)
            selecting a historical run directory. Caller must validate the
            shape; this helper assumes ``epoch`` is well-formed.

    Raises:
        HTTPException: 404 if neither a live CR nor a PVC directory exists.
        HTTPException: Other ``kubernetes_asyncio.client.ApiException`` status
            codes propagate (e.g. 401/403 on RBAC denial).
    """
    job = await find_any_job(api, results_dir, namespace, name, epoch=epoch)
    if job is None:
        raise HTTPException(404, f"Job {namespace}/{name} not found")

    if job.source == "archived":
        job_dir = resolve_run_dir(results_dir, namespace, name, epoch=epoch)
        if job_dir is None:
            raise HTTPException(404, f"No persisted run for {namespace}/{name}")
        # ``_summary_path`` handles the .zst-then-raw fallback used elsewhere
        # in the codebase (results_db.py:76, runs_index.py:907) — without it
        # archived-job detail pages on a deployment with the default
        # AIPERF_RESULTS_COMPRESS_ON_DISK=true silently render empty Final KPIs.
        summary_file = _summary_path(job_dir)
        summary = (_read_summary(summary_file) or {}) if summary_file else {}
        conditions: list[dict[str, Any]] | None = None
        conditions_path = job_dir / "conditions.json"
        if conditions_path.is_file():
            try:
                raw = orjson.loads(conditions_path.read_bytes())
            except (OSError, orjson.JSONDecodeError) as e:
                logger.warning(
                    f"Failed to read archived conditions {conditions_path}: {e}"
                )
            else:
                if isinstance(raw, list):
                    conditions = raw
                elif isinstance(raw, dict) and isinstance(raw.get("conditions"), list):
                    conditions = raw["conditions"]
        return JobDetailResponse(
            job=job.model_dump(by_alias=True),
            status=synthesize_status_from_summary(
                namespace,
                name,
                summary,
                conditions,
                phase=job.phase,
            ),
            pods=[],
        )

    raw_status = await get_raw_aiperfjob_status(api, name, namespace)
    pods_raw = await get_pods(api, namespace, f"aiperf.nvidia.com/job-id={name}")
    return JobDetailResponse(
        job=job.model_dump(by_alias=True),
        status=raw_status or {},
        pods=[_pod_summary(p) for p in pods_raw],
    )


_SUCCEEDED_PHASES = frozenset({"succeeded", "completed"})
_FAILED_PHASES = frozenset({"failed"})
_CANCELLED_PHASES = frozenset({"cancelled", "canceled"})


def derive_run_status(
    row: RunIndexRow,
    *,
    live_running_epoch: str | None,
) -> Literal["running", "succeeded", "failed", "cancelled", "unknown"]:
    """Reconcile a runs-index row with the live CR into a single status enum.

    The live in-flight epoch always reports ``"running"`` even if the index
    row's ``phase`` lags behind (the index is updated on completion; the CR
    is the truth-of-the-moment for "is this epoch alive right now?"). For
    every other row, ``error`` overrides phase (a row that finished with an
    error is failed, regardless of the phase column), and unknown phases
    fall through to ``"unknown"`` rather than guessing.

    Example:
        For a row with phase="Succeeded" and no live in-flight epoch,
        ``derive_run_status(row, live_running_epoch=None)`` returns ``"succeeded"``.
        For the same row when ``live_running_epoch`` matches ``row.epoch``,
        it returns ``"running"`` regardless of phase.
    """
    if live_running_epoch is not None and row.epoch == live_running_epoch:
        return "running"
    if row.error:
        return "failed"
    phase = (row.phase or "").lower()
    if phase in _SUCCEEDED_PHASES:
        return "succeeded"
    if phase in _FAILED_PHASES:
        return "failed"
    if phase in _CANCELLED_PHASES:
        return "cancelled"
    return "unknown"


def _disk_epoch_summary(
    run: RunEntry, *, live_running_epoch: str | None
) -> JobEpochSummary:
    """Convert a disk-discovered run into a status-light epoch summary."""
    return JobEpochSummary(
        epoch=run.epoch,
        is_latest=run.is_latest,
        mtime_epoch=run.mtime_epoch,
        file_count=run.file_count,
        status=(
            "running"
            if live_running_epoch is not None and run.epoch == live_running_epoch
            else "unknown"
        ),
        started_at=None,
        ended_at=None,
    )


def _indexed_epoch_summary(
    row: RunIndexRow,
    existing: JobEpochSummary | None,
    *,
    live_running_epoch: str | None,
) -> JobEpochSummary:
    """Overlay indexed status fields onto a disk summary when available."""
    return JobEpochSummary(
        epoch=row.epoch,
        is_latest=existing.is_latest if existing is not None else bool(row.is_latest),
        mtime_epoch=existing.mtime_epoch
        if existing is not None
        else int(row.mtime_epoch or 0),
        file_count=existing.file_count if existing is not None else row.file_count,
        status=derive_run_status(row, live_running_epoch=live_running_epoch),
        started_at=_iso_to_unix(row.start_time),
        ended_at=_iso_to_unix(row.end_time),
    )


async def _list_job_epochs_impl(
    api: ApiClient | None,
    base_dir: Path,
    namespace: str,
    name: str,
) -> JobEpochsResponse:
    """Body of GET /api/v1/jobs/{namespace}/{name}/epochs.

    Reads rich rows from the runs SQLite index and reconciles each row's
    ``phase`` / ``error`` against the live CR's ``status.runEpoch`` to
    produce a single normalized ``status`` enum per epoch. Merges those rows
    with a disk walk (``list_runs_async``) so disk-only epochs report
    ``status='unknown'`` and ``started_at`` / ``ended_at`` of ``None``.

    Order is ascending by ``mtime_epoch`` so the latest entry sits at the
    tail; this matches the prior contract.

    Returns an empty list when neither the index nor the disk has rows
    (job has never been persisted, or PVC directory was reaped).
    """
    # Resolve the live in-flight epoch from the CR (None if not running).
    live_running_epoch: str | None = None
    if api is not None:
        try:
            cr = await get_raw_aiperfjob(api, namespace, name)
        except Exception:  # noqa: BLE001 — UI surface, never block on CR errors
            cr = None
        if isinstance(cr, dict):
            cr_status = cr.get("status") or {}
            if isinstance(cr_status, dict) and cr_status.get("phase") == "Running":
                run_epoch = cr_status.get("runEpoch")
                if run_epoch is not None:
                    live_running_epoch = str(run_epoch)

    # Index-first read, then reconcile with disk so a stale index never hides
    # a newer persisted run.
    rich_rows: list[RunIndexRow] = []
    try:
        rich_rows = await runs_index.list_runs_for_job(namespace, name)
    except Exception:  # noqa: BLE001 — index unavailable degrades to disk
        rich_rows = []

    runs = await list_runs_async(base_dir, namespace, name)
    by_epoch = {
        run.epoch: _disk_epoch_summary(run, live_running_epoch=live_running_epoch)
        for run in runs
    }

    # Mirror the disk guard from list_runs_async: a runs-index row whose epoch
    # dir was reaped from disk (retention rmtree) but whose index DELETE lagged
    # (fire-and-forget create_task, or a stale read on the read-only
    # results-server) must NOT surface as a phantom run — a follow-up file
    # fetch for it would 404. The live in-flight epoch is exempt because its
    # dir may not be on disk yet.
    parent = job_dir(base_dir, namespace, name)
    for row in rich_rows:
        if (
            row.epoch not in by_epoch
            and row.epoch != live_running_epoch
            and not (parent / row.epoch).is_dir()
        ):
            continue
        by_epoch[row.epoch] = _indexed_epoch_summary(
            row,
            by_epoch.get(row.epoch),
            live_running_epoch=live_running_epoch,
        )

    return JobEpochsResponse(
        epochs=sorted(
            by_epoch.values(), key=lambda entry: (entry.mtime_epoch, entry.epoch)
        )
    )


def _iso_to_unix(ts: str | None) -> int | None:
    """Parse a ``2026-05-01T00:00:00+00:00`` style timestamp to unix seconds; None on miss or non-string input."""
    if not isinstance(ts, str):
        return None
    try:
        # Accept both 'Z' suffix and explicit offsets.
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None


async def _create_job_impl(
    api: ApiClient,
    manifest: dict[str, Any],
) -> CreateJobResponse:
    """Body of POST /api/v1/jobs: create an AIPerfJob CR from a manifest dict.

    Fills in ``apiVersion`` and ``kind`` when omitted, resolves the target
    namespace (default: ``default``), and submits to the CustomObjectsApi.
    Returns the namespace/name/uid so the UI can deep-link to the new run's
    workbench page immediately.

    Args:
        api: The kubernetes_asyncio ApiClient.
        manifest: Full AIPerfJob manifest shaped like ``kubectl apply -f`` input.

    Raises:
        HTTPException: 400 when the manifest is missing ``metadata.name`` or
            is otherwise malformed in a way the client should fix.
        HTTPException: Other ``kubernetes_asyncio.client.ApiException`` status
            codes propagate (e.g. 401/403 on RBAC denial, 409 if a CR with
            the same name already exists, 422 on schema validation errors).
    """
    if not isinstance(manifest, dict):
        raise HTTPException(400, "Manifest must be a JSON/YAML object.")

    manifest = dict(manifest)
    manifest.setdefault("apiVersion", "aiperf.nvidia.com/v1alpha1")
    manifest.setdefault("kind", "AIPerfJob")
    metadata = manifest.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise HTTPException(400, "metadata must be an object.")
    name = metadata.get("name")
    if not name:
        raise HTTPException(400, "metadata.name is required.")
    namespace = metadata.get("namespace") or "default"
    metadata["namespace"] = namespace
    manifest["metadata"] = metadata

    co = client.CustomObjectsApi(api)
    try:
        created = await co.create_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfjobs",
            body=manifest,
        )
    except ApiException as e:
        detail = e.body or e.reason or "Kubernetes API error"
        raise HTTPException(e.status or 500, detail) from e

    uid = (created.get("metadata") or {}).get("uid")
    return CreateJobResponse(namespace=namespace, name=name, uid=uid)


async def _cancel_job_impl(
    api: ApiClient,
    results_dir: Path,
    namespace: str,
    name: str,
) -> CancelResponse:
    """Body of POST /api/v1/jobs/{namespace}/{name}/cancel: set ``spec.cancel=true``.

    This endpoint is *asynchronous*: it patches the AIPerfJob CR's
    ``spec.cancel`` field to ``true`` and returns immediately. The kopf
    operator's reconciler observes the change and drives the benchmark to a
    stopped state (cancelling workers, tearing down pods, finalising results).
    The endpoint does NOT wait for that reconciliation - callers that need to
    observe the terminal phase should poll ``get_job`` until ``status.phase``
    becomes ``Cancelled``/``Failed``/``Succeeded``.

    Archived (PVC-only) jobs cannot be cancelled — their Kubernetes resource no
    longer exists — so the endpoint returns 400 instead of attempting the patch.

    Args:
        api: The kubernetes_asyncio ApiClient.
        results_dir: Base directory on the results PVC (used to detect
            archived-only jobs that have no CR to cancel).
        namespace: Kubernetes namespace containing the AIPerfJob CR.
        name: Name of the AIPerfJob CR to cancel.

    Raises:
        HTTPException: 404 if neither a live CR nor a PVC directory exists.
        HTTPException: 400 if the job is archived-only (no CR on the cluster).
        HTTPException: Other ``kubernetes_asyncio.client.ApiException`` status
            codes propagate (e.g. 401/403 on RBAC denial, 409 on
            concurrent-modification conflicts).
    """
    job = await find_any_job(api, results_dir, namespace, name)
    if job is None:
        raise HTTPException(404, f"Job {namespace}/{name} not found")
    if job.source == "archived":
        raise HTTPException(
            400,
            f"Cannot cancel archived job {namespace}/{name}: "
            "the Kubernetes resource no longer exists.",
        )
    await cancel_aiperf_job(api, name, namespace)
    return CancelResponse(cancelled=True)


MAX_EVENTS_RETURNED = 200

# GKE-managed ValidatingAdmissionPolicies whose CEL expressions error out on
# unrelated objects (e.g. accessing ``request.userInfo.username`` without a
# ``has(...)`` guard). The resulting PolicyViolation events surface against
# AIPerfJob pods even though the policy targets node-level kubelet/P4SA flows
# we don't participate in. They are pure control-plane noise; drop them so
# the events pane stays focused on workload-relevant signal.
_NOISE_ADMISSION_POLICIES: tuple[str, ...] = ("validating-node-p4sa-audience",)


def _is_noise_event(raw: Any) -> bool:
    """Return True for known-noisy GKE admission-policy events to drop."""
    if getattr(raw, "reason", None) != "PolicyViolation":
        return False
    msg = getattr(raw, "message", "") or ""
    return any(p in msg for p in _NOISE_ADMISSION_POLICIES)


def _event_to_entry(raw: Any) -> EventEntry:
    """Map a ``V1Event`` to the UI-facing :class:`EventEntry`.

    Timestamps are ISO-8601 strings (``.isoformat()``) so the UI does not need
    to know the ``kubernetes_asyncio`` datetime type. Both ``firstTimestamp``
    and ``lastTimestamp`` can be None for events emitted via the newer
    ``events.k8s.io/v1`` API — we fall back to ``eventTime`` if present.
    """
    involved = raw.involved_object
    src = raw.source
    # event_time is the newer ``events.k8s.io/v1`` timestamp; older Events
    # populate first/last timestamp but not event_time.
    event_time = getattr(raw, "event_time", None)
    first_ts = raw.first_timestamp or event_time
    last_ts = raw.last_timestamp or event_time
    return EventEntry(
        type=raw.type,
        reason=raw.reason,
        message=raw.message,
        source=EventSource(
            component=getattr(src, "component", None) if src else None,
            host=getattr(src, "host", None) if src else None,
        ),
        involved_object=EventInvolvedObject(
            kind=getattr(involved, "kind", None) if involved else None,
            name=getattr(involved, "name", None) if involved else None,
            namespace=getattr(involved, "namespace", None) if involved else None,
        ),
        first_timestamp=first_ts.isoformat() if first_ts is not None else None,
        last_timestamp=last_ts.isoformat() if last_ts is not None else None,
        count=raw.count,
    )


async def _list_events_impl(
    api: ApiClient,
    namespace: str,
    name: str,
) -> JobEventsResponse:
    """Body of GET /api/v1/jobs/{namespace}/{name}/events.

    Collects events for (1) the AIPerfJob CR itself and (2) every pod labelled
    ``aiperf.nvidia.com/job-id=<name>``. Owned intermediate resources (k8s Jobs,
    JobSets, ConfigMaps, Services) are intentionally omitted — their event
    streams are low-signal for the UI log and the pod events already surface
    the interesting failures (ImagePull, FailedScheduling, OOMKilled, ...).

    The result is sorted by ``lastTimestamp`` descending and capped at
    :data:`MAX_EVENTS_RETURNED` entries. Events with no timestamp sort last.

    Archived (PVC-only) runs whose AIPerfJob CR is gone return an empty
    event list — there are no live objects left to source events from,
    and the run view still renders for archived jobs so 404 here would
    surface as a noisy ``console.error`` on the browser side for a
    legitimate UI state.

    Raises:
        ApiException: Non-404 ``kubernetes_asyncio.client.ApiException``
            errors (e.g. 401/403) propagate via the app-level handler
            registered in ``results_server._register_k8s_exception_handler``.
    """
    # suppress_api_errors=False keeps the 404-vs-500 contract distinct: a gone
    # CR (404) is an archived run -> empty events; an apiserver failure (500,
    # etcd unavailable, 401/403) must surface, not masquerade as "no events".
    cr = await get_raw_aiperfjob(api, namespace, name, suppress_api_errors=False)
    if cr is None:
        return JobEventsResponse(events=[])

    cr_events = await list_events_for_object(api, namespace, name)

    pods = await get_pods(api, namespace, f"aiperf.nvidia.com/job-id={name}")
    pod_names = [p.metadata.name for p in pods if p.metadata and p.metadata.name]

    pod_event_lists: list[list[Any]] = []
    for pod_name in pod_names:
        try:
            pod_event_lists.append(
                await list_events_for_object(api, namespace, pod_name)
            )
        except ApiException as e:
            logger.warning(
                "Failed to list events for pod %s/%s (apiserver %s): %s",
                namespace,
                pod_name,
                e.status,
                e,
            )

    raw_events: list[Any] = [*cr_events]
    for lst in pod_event_lists:
        raw_events.extend(lst)

    raw_events = [e for e in raw_events if not _is_noise_event(e)]

    entries = [_event_to_entry(e) for e in raw_events]
    # Sort by last_timestamp desc; push None (no timestamp) to the end.
    entries.sort(key=lambda e: e.last_timestamp or "", reverse=True)
    return JobEventsResponse(events=entries[:MAX_EVENTS_RETURNED])


async def _cluster_info_impl(api: ApiClient) -> ClusterResponse:
    """Body of GET /api/v1/cluster: best-effort cluster-wide capacity, GPU
    utilization, and Kubernetes server version.

    Calls the core ``/version`` endpoint for the gitVersion plus
    :func:`_fetch_cluster_gpu_stats` for nodes/pods accounting. All
    sub-calls are best-effort: failures fall back to ``"unknown"`` /
    zero counts rather than surfacing errors, because the UI displays
    this as supplementary context and callers with limited RBAC should
    not see the page fail.
    """
    k8s_version = await _fetch_k8s_version(api)
    stats = await _fetch_cluster_gpu_stats(api)
    from aiperf.operator.environment import OperatorEnvironment

    # Env var wins (operator can override on managed clusters where the
    # apiserver lookup is impossible); otherwise probe well-known
    # ConfigMaps for the cluster name.
    cluster_name = OperatorEnvironment.CLUSTER_NAME or await _fetch_cluster_name(api)
    return ClusterResponse(
        nodes=stats["node_count"],
        gpus=stats["gpus"],
        gpus_used=stats["gpus_used"],
        gpus_free=stats["gpus_free"],
        utilization_percent=stats["utilization_percent"],
        gpu_nodes=stats["gpu_nodes"],
        nodes_free=stats["nodes_free"],
        nodes_partial=stats["nodes_partial"],
        nodes_full=stats["nodes_full"],
        kubernetes_version=k8s_version,
        cluster_name=cluster_name,
    )


def _register_job_collection_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
    results_dir: Path,
    mutating_dependencies: Sequence[DependsParam],
) -> None:
    """Register the ``GET /jobs`` (list) and ``POST /jobs`` (create) endpoints."""

    @router.get("/jobs", response_model=ActiveJobListResponse)
    async def list_jobs(request: Request) -> Response:
        result = await _list_jobs_impl(require_api(), results_dir)
        return etag_response(request, result.model_dump(mode="json", by_alias=True))

    @router.post(
        "/jobs",
        response_model=CreateJobResponse,
        status_code=201,
        dependencies=list(mutating_dependencies),
    )
    async def create_job(body: CreateJobRequest) -> CreateJobResponse:
        return await _create_job_impl(require_api(), body.manifest)


def _register_job_detail_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
    optional_api: Callable[[], ApiClient | None],
    results_dir: Path,
) -> None:
    """Register the per-job read endpoints (detail + epoch listing)."""

    @router.get("/jobs/{namespace}/{name}", response_model=JobDetailResponse)
    async def get_job(
        request: Request, namespace: str, name: str, epoch: str | None = None
    ) -> Response:
        validate_results_path_params(namespace, name)
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        result = await _get_job_impl(
            require_api(), results_dir, namespace, name, epoch=epoch
        )
        return etag_response(request, result.model_dump(mode="json", by_alias=True))

    @router.get(
        "/jobs/{namespace}/{name}/epochs",
        response_model=JobEpochsResponse,
        response_model_by_alias=True,
    )
    async def list_job_epochs(namespace: str, name: str) -> JobEpochsResponse:
        validate_results_path_params(namespace, name)
        return await _list_job_epochs_impl(optional_api(), results_dir, namespace, name)


def _register_job_action_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
    results_dir: Path,
    mutating_dependencies: Sequence[DependsParam],
) -> None:
    """Register the per-job cancel, events, and pod-log endpoints."""

    @router.post(
        "/jobs/{namespace}/{name}/cancel",
        response_model=CancelResponse,
        dependencies=list(mutating_dependencies),
    )
    async def cancel_job(namespace: str, name: str) -> CancelResponse:
        validate_results_path_params(namespace, name)
        return await _cancel_job_impl(require_api(), results_dir, namespace, name)

    @router.get("/jobs/{namespace}/{name}/events", response_model=JobEventsResponse)
    async def list_job_events(namespace: str, name: str) -> JobEventsResponse:
        return await _list_events_impl(require_api(), namespace, name)

    @router.get("/jobs/{namespace}/{name}/logs")
    async def get_pod_logs(
        namespace: str,
        name: str,
        *,
        pod: str,
        follow: int = 0,
        tail_lines: int = 200,
        container: str | None = None,
    ) -> Response:
        return await get_pod_logs_impl(
            require_api(),
            namespace,
            name,
            pod=pod,
            follow=bool(follow),
            tail_lines=tail_lines,
            container=container,
        )


def _register_cluster_routes(
    router: APIRouter, require_api: Callable[[], ApiClient]
) -> None:
    """Register the cluster-wide capacity/version endpoint."""

    @router.get("/cluster", response_model=ClusterResponse)
    async def cluster_info() -> ClusterResponse:
        return await _cluster_info_impl(require_api())


def create_jobs_router(
    api_holder: list[ApiClient | None] | None = None,
    results_dir: Path | None = None,
    mutating_dependencies: Sequence[DependsParam] = (),
) -> APIRouter:
    """Create the jobs/cluster API router.

    All endpoints return 503 if the Kubernetes ApiClient has not been
    initialised (set during FastAPI lifespan startup). See the ``_*_impl``
    helpers above for per-endpoint behaviour and error semantics.

    Args:
        api_holder: Mutable single-element list holding the kubernetes_asyncio
            ApiClient. The client is set during app lifespan startup. If the
            list is empty or contains None, endpoints return 503.
        results_dir: Base directory on the results PVC; passed to the union
            helpers so ``GET /jobs`` and ``GET /jobs/{ns}/{name}`` can surface
            archived (CR-deleted) runs alongside live ones.
    """
    _holder = api_holder if api_holder is not None else [None]
    _results_dir = results_dir if results_dir is not None else Path("/data")
    router = APIRouter(prefix="/api/v1", tags=["jobs"])

    def _require_api() -> ApiClient:
        api = _holder[0] if _holder else None
        if api is None:
            raise HTTPException(
                503,
                "Kubernetes API client not yet initialized by FastAPI lifespan; "
                "retry in a few seconds or check /healthz",
            )
        return api

    def _optional_api() -> ApiClient | None:
        """Return the live ApiClient, or None if lifespan hasn't initialized it.

        Used by endpoints that can degrade gracefully when the cluster is not
        reachable (e.g. epoch listing falls back to the runs index + disk).
        """
        return _holder[0] if _holder else None

    _register_job_collection_routes(
        router, _require_api, _results_dir, mutating_dependencies
    )
    _register_job_detail_routes(router, _require_api, _optional_api, _results_dir)
    _register_job_action_routes(
        router, _require_api, _results_dir, mutating_dependencies
    )
    _register_cluster_routes(router, _require_api)
    return router
