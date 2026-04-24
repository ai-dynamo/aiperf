# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API router for live Kubernetes job and cluster state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import (
    cancel_aiperf_job,
    cluster_version,
    get_pods,
    get_raw_aiperfjob,
    get_raw_aiperfjob_status,
)
from aiperf.operator.job_union import (
    find_any_job,
    list_all_jobs,
    synthesize_status_from_summary,
)
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
    JobEventsResponse,
    JobPodSummary,
)

if TYPE_CHECKING:
    from kubernetes_asyncio.client.models import V1Node, V1Pod

logger = logging.getLogger("aiperf.operator.ui")


def _pod_summary(pod: V1Pod) -> JobPodSummary:
    """Extract pod name, phase, readiness, and restart count for the UI."""
    meta = pod.metadata
    status = pod.status
    container_statuses = (status.container_statuses or []) if status else []
    return JobPodSummary(
        name=(meta.name if meta else "") or "",
        phase=(status.phase if status else None) or "Unknown",
        ready=any(bool(c.ready) for c in container_statuses),
        restarts=sum(int(c.restart_count or 0) for c in container_statuses),
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


async def _fetch_node_gpu_totals(api: ApiClient) -> tuple[int, int]:
    """Return (node_count, total_nvidia_gpus). Returns (0, 0) on failure."""
    try:
        node_list = await client.CoreV1Api(api).list_node()
    except ApiException as e:
        # 403 here is almost always the operator ClusterRole missing
        # `nodes get/list` — log at ERROR so it surfaces in the usual
        # RBAC-misconfig triage instead of masquerading as "0 nodes".
        if (e.status or 0) == 403:
            logger.error(
                "Cluster node listing forbidden (403) — check that the "
                "operator ClusterRole grants `nodes get/list`: %s",
                e,
            )
        else:
            logger.warning("Failed to query nodes (apiserver %s): %s", e.status, e)
        return 0, 0
    except Exception as e:  # noqa: BLE001 - UI tolerates missing cluster-wide query
        logger.warning(f"Failed to query nodes: {e}")
        return 0, 0
    nodes = node_list.items
    return len(nodes), sum(_node_gpu_count(n) for n in nodes)


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
) -> JobDetailResponse:
    """Body of GET /api/v1/jobs/{namespace}/{name}: fetch a CR plus its pod roster.

    Returns three things joined into one response: (1) the AIPerfJob summary
    (same shape as ``list_jobs``), (2) the raw CR ``.status`` subresource
    (phase, conditions, liveMetrics), and (3) the current pod list filtered by
    the ``aiperf.nvidia.com/job-id=<name>`` label selector.

    Archived (PVC-only) jobs have no cluster CR, so the response returns an
    empty ``status`` dict and empty ``pods`` list alongside the archived job
    summary.

    Args:
        api: The kubernetes_asyncio ApiClient.
        results_dir: Base directory on the results PVC.
        namespace: Kubernetes namespace containing the AIPerfJob CR or PVC dir.
        name: Name of the AIPerfJob CR (also the label value matched when
            listing pods, and the PVC subdirectory name).

    Raises:
        HTTPException: 404 if neither a live CR nor a PVC directory exists.
        HTTPException: Other ``kubernetes_asyncio.client.ApiException`` status
            codes propagate (e.g. 401/403 on RBAC denial).
    """
    job = await find_any_job(api, results_dir, namespace, name)
    if job is None:
        raise HTTPException(404, f"Job {namespace}/{name} not found")

    if job.source == "archived":
        job_dir = results_dir / namespace / name
        summary_path = job_dir / "profile_export_aiperf.json"
        try:
            summary = orjson.loads(summary_path.read_bytes())
        except (OSError, orjson.JSONDecodeError) as e:
            logger.warning(f"Failed to read archived summary {summary_path}: {e}")
            summary = {}
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
            status=synthesize_status_from_summary(namespace, name, summary, conditions),
            pods=[],
        )

    raw_status = await get_raw_aiperfjob_status(api, name, namespace)
    pods_raw = await get_pods(api, namespace, f"aiperf.nvidia.com/job-id={name}")
    return JobDetailResponse(
        job=job.model_dump(by_alias=True),
        status=raw_status or {},
        pods=[_pod_summary(p) for p in pods_raw],
    )


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


async def _events_for_object(
    core: client.CoreV1Api,
    namespace: str,
    object_name: str,
) -> list[Any]:
    """Return raw ``V1Event`` objects whose ``involvedObject.name`` matches.

    Uses a field selector so the apiserver filters server-side — this is cheap
    even in busy namespaces. ``involvedObject.name`` is not globally unique
    (two kinds can share a name), so callers may need to further filter by
    ``involvedObject.kind`` if disambiguation matters; the jobs endpoint does
    not, because AIPerfJob CRs and their pods always have distinct names.
    """
    resp = await core.list_namespaced_event(
        namespace=namespace,
        field_selector=f"involvedObject.name={object_name}",
    )
    return list(resp.items or [])


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

    Raises:
        HTTPException: 404 if the AIPerfJob CR does not exist in ``namespace``.
            Non-404 ``ApiException`` errors propagate via the app-level handler
            registered in ``results_server._register_k8s_exception_handler``.
    """
    cr = await get_raw_aiperfjob(api, namespace, name)
    if cr is None:
        raise HTTPException(404, f"AIPerfJob {namespace}/{name} not found")

    core = client.CoreV1Api(api)
    cr_events = await _events_for_object(core, namespace, name)

    pods = await get_pods(api, namespace, f"aiperf.nvidia.com/job-id={name}")
    pod_names = [p.metadata.name for p in pods if p.metadata and p.metadata.name]

    pod_event_lists: list[list[Any]] = []
    for pod_name in pod_names:
        pod_event_lists.append(await _events_for_object(core, namespace, pod_name))

    raw_events: list[Any] = [*cr_events]
    for lst in pod_event_lists:
        raw_events.extend(lst)

    entries = [_event_to_entry(e) for e in raw_events]
    # Sort by last_timestamp desc; push None (no timestamp) to the end.
    entries.sort(key=lambda e: e.last_timestamp or "", reverse=True)
    return JobEventsResponse(events=entries[:MAX_EVENTS_RETURNED])


async def _cluster_info_impl(api: ApiClient) -> ClusterResponse:
    """Body of GET /api/v1/cluster: best-effort cluster-wide node and GPU totals.

    Calls the core ``/version`` endpoint for the server gitVersion and
    ``list_node`` for node count + ``nvidia.com/gpu`` allocatable totals. Both
    calls are best-effort: failures fall back to ``"unknown"`` / ``(0, 0)``
    rather than surfacing errors, because the UI displays this as supplementary
    context and callers with limited RBAC should not see the page fail.
    """
    k8s_version = await _fetch_k8s_version(api)
    node_count, gpu_count = await _fetch_node_gpu_totals(api)
    return ClusterResponse(
        nodes=node_count,
        gpus=gpu_count,
        kubernetes_version=k8s_version,
    )


def create_jobs_router(
    api_holder: list[ApiClient | None] | None = None,
    results_dir: Path | None = None,
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

    @router.get("/jobs", response_model=ActiveJobListResponse)
    async def list_jobs() -> ActiveJobListResponse:
        return await _list_jobs_impl(_require_api(), _results_dir)

    @router.post("/jobs", response_model=CreateJobResponse, status_code=201)
    async def create_job(body: CreateJobRequest) -> CreateJobResponse:
        return await _create_job_impl(_require_api(), body.manifest)

    @router.get("/jobs/{namespace}/{name}", response_model=JobDetailResponse)
    async def get_job(namespace: str, name: str) -> JobDetailResponse:
        return await _get_job_impl(_require_api(), _results_dir, namespace, name)

    @router.post("/jobs/{namespace}/{name}/cancel", response_model=CancelResponse)
    async def cancel_job(namespace: str, name: str) -> CancelResponse:
        return await _cancel_job_impl(_require_api(), _results_dir, namespace, name)

    @router.get("/jobs/{namespace}/{name}/events", response_model=JobEventsResponse)
    async def list_job_events(namespace: str, name: str) -> JobEventsResponse:
        return await _list_events_impl(_require_api(), namespace, name)

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
            _require_api(),
            namespace,
            name,
            pod=pod,
            follow=bool(follow),
            tail_lines=tail_lines,
            container=container,
        )

    @router.get("/cluster", response_model=ClusterResponse)
    async def cluster_info() -> ClusterResponse:
        return await _cluster_info_impl(_require_api())

    return router
