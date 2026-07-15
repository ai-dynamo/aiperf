# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI router for /api/v1/sweeps* — read-only AIPerfSweep view.

Dual-backed via :mod:`aiperf.operator.sweep_union`: every endpoint
returns the same shape regardless of whether the parent CR exists or
the data is reconstructed from the archived ``aggregate.json``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, HTTPException
from kubernetes_asyncio.client import ApiClient

from aiperf.kubernetes.models import AIPerfJobInfo
from aiperf.operator.job_union import list_all_jobs
from aiperf.operator.results_layout import (
    EPOCH_RE,
    list_sweep_epochs_async,
    resolve_sweep_dir,
)
from aiperf.operator.routers._path_params import validate_results_path_params
from aiperf.operator.routers._sweeps_artifacts import register_sweep_artifact_routes
from aiperf.operator.routers._sweeps_diagnostics import (
    fetch_sweep_pod_summaries,
    register_diagnostics_routes,
)
from aiperf.operator.routers._sweeps_live import children_manifest_from_live_aiperfjobs
from aiperf.operator.routers._sweeps_spec import (
    spec_summary_from_record,
)
from aiperf.operator.routers.sweeps_models import (
    CellAggregatesResponse,
    CellEntry,
    ChildJobRef,
    ChildrenManifestEntry,
    ChildrenManifestResponse,
    SweepDetailResponse,
    SweepEpochsResponse,
    SweepEpochSummary,
    SweepListResponse,
    SweepSummary,
)
from aiperf.operator.sweep_union import (
    SweepRecord,
    find_any_sweep,
    list_all_sweeps,
    sanitize_current_child_ref,
    sanitize_run_states,
    synthesize_sweep_status_from_aggregate,
)

logger = logging.getLogger("aiperf.operator.ui")


def _summary(rec: SweepRecord) -> SweepSummary:
    return SweepSummary(
        namespace=rec.namespace,
        name=rec.name,
        source=rec.source,  # type: ignore[arg-type]
        phase=rec.phase,
        total_variations=rec.total_variations,
        completed_runs=rec.completed_runs,
        failed_runs=rec.failed_runs,
        cancelled_runs=rec.cancelled_runs,
        age_seconds=rec.age_seconds,
        model=rec.model,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        api_url=rec.api_url,
        results_available=rec.results_available,
        current_child_ref=sanitize_current_child_ref(rec.current_child_ref),
        run_states=sanitize_run_states(rec.run_states),
    )


def _read_conditions(sweep_dir_path: str | None) -> list[dict[str, Any]]:
    if not sweep_dir_path:
        return []
    p = Path(sweep_dir_path).parent / "conditions.json"
    if not p.is_file():
        return []
    try:
        raw = orjson.loads(p.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning("conditions.json unreadable at %s: %s", p, e)
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("conditions"), list):
        return raw["conditions"]
    return []


async def _list_sweeps_impl(api: ApiClient, base_dir: Path) -> SweepListResponse:
    records = await list_all_sweeps(api, base_dir, all_namespaces=True)
    return SweepListResponse(sweeps=[_summary(r) for r in records])


async def _get_sweep_impl(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> SweepDetailResponse:
    rec = await find_any_sweep(api, base_dir, namespace, name, epoch=epoch)
    if rec is None:
        raise HTTPException(404, f"Sweep {namespace}/{name} not found")

    if rec.source == "archived" and rec.aggregate_doc is not None:
        status = synthesize_sweep_status_from_aggregate(
            namespace, name, rec.aggregate_doc, _read_conditions(rec.aggregate_path)
        )
    elif rec.source == "archived":
        status = {"phase": "Unknown", "conditions": []}
    else:
        status = rec.raw_status or {}

    spec_summary = spec_summary_from_record(rec)

    children_records = await list_all_jobs(
        api, base_dir, all_namespaces=False, namespace=namespace
    )
    children = [
        j.model_dump(by_alias=True)
        for j in children_records
        if getattr(j, "sweep_name", None) == name and j.namespace == namespace
    ]

    pods = await fetch_sweep_pod_summaries(api, namespace, name, rec.source)

    return SweepDetailResponse(
        sweep=_summary(rec),
        status=status,
        spec_summary=spec_summary,
        children=children,
        pods=pods,
    )


def _cells_from_aggregate(doc: dict[str, Any]) -> list[CellEntry]:
    raw_cells = doc.get("per_cell_aggregates") or []
    out: list[CellEntry] = []
    for c in raw_cells:
        if not isinstance(c, dict):
            continue
        children_raw = c.get("children") or []
        children = [
            ChildJobRef(
                namespace=child.get("namespace") or "",
                name=child.get("name") or "",
                trial_index=child.get("trial_index"),
                phase=child.get("phase"),
            )
            for child in children_raw
            if isinstance(child, dict)
        ]
        out.append(
            CellEntry(
                variation_index=int(c.get("variation_index") or 0),
                variation_label=str(c.get("variation_label") or ""),
                values=dict(c.get("values") or {}),
                trials_completed=int(c.get("trials_completed") or 0),
                trials_failed=int(c.get("trials_failed") or 0),
                metrics=dict(c.get("metrics") or {}),
                children=children,
            )
        )
    return sorted(out, key=lambda x: x.variation_index)


def _fold_child_into_bucket(bucket: dict[str, Any], job: AIPerfJobInfo) -> None:
    """Fold one child job into its variation bucket (counts, metrics, refs).

    Status mapping: only count terminal children towards aggregates.
    """
    phase = (job.phase or "").lower()
    if phase in {"succeeded", "completed"}:
        bucket["trials_completed"] += 1
        if job.throughput_rps is not None:
            bucket["throughputs"].append(float(job.throughput_rps))
        if job.latency_p99_ms is not None:
            bucket["p99_latencies"].append(float(job.latency_p99_ms))
    elif phase in {"failed", "cancelled", "partiallyfailed"}:
        bucket["trials_failed"] += 1
    bucket["children"].append(
        ChildJobRef(
            namespace=job.namespace,
            name=job.name,
            trial_index=None,
            phase=job.phase,
        )
    )


def _avg(xs: list[float]) -> float | None:
    """Arithmetic mean of ``xs``, or None when the list is empty."""
    return (sum(xs) / len(xs)) if xs else None


def _cell_entry_from_bucket(idx: int, bucket: dict[str, Any]) -> CellEntry:
    """Materialize the response-facing CellEntry for one variation bucket."""
    metrics: dict[str, dict[str, float]] = {}
    thr_avg = _avg(bucket["throughputs"])
    if thr_avg is not None:
        metrics["request_throughput"] = {"avg": thr_avg}
    lat_avg = _avg(bucket["p99_latencies"])
    if lat_avg is not None:
        metrics["request_latency_p99"] = {"avg": lat_avg}
    return CellEntry(
        variation_index=idx,
        variation_label=bucket["variation_label"],
        values={},  # structured values come from spec; live path leaves empty
        trials_completed=bucket["trials_completed"],
        trials_failed=bucket["trials_failed"],
        metrics=metrics,
        children=bucket["children"],
    )


async def _cells_from_live_children(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
) -> list[CellEntry]:
    """Compute per-cell aggregates by grouping children by variation_index.

    Used when the sweep is live and has no aggregate.json yet (mid-run).
    Reads each child's profile_export_aiperf.json from the PVC if present.
    Returns an empty list if no terminal children are persisted yet.
    """
    children_records = await list_all_jobs(
        api, base_dir, all_namespaces=False, namespace=namespace
    )
    matched = [
        j
        for j in children_records
        if getattr(j, "sweep_name", None) == sweep_name and j.namespace == namespace
    ]
    by_cell: dict[int, dict[str, Any]] = {}
    for j in matched:
        idx = getattr(j, "variation_index", None)
        if idx is None:
            continue
        bucket = by_cell.setdefault(
            int(idx),
            {
                "variation_label": getattr(j, "variation_label", "") or "",
                "trials_completed": 0,
                "trials_failed": 0,
                "throughputs": [],
                "p99_latencies": [],
                "children": [],
            },
        )
        _fold_child_into_bucket(bucket, j)

    return [_cell_entry_from_bucket(idx, b) for idx, b in sorted(by_cell.items())]


async def _get_cells_impl(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> CellAggregatesResponse:
    rec = await find_any_sweep(api, base_dir, namespace, name, epoch=epoch)
    if rec is None:
        raise HTTPException(404, f"Sweep {namespace}/{name} not found")
    spec_summary = spec_summary_from_record(rec)
    if rec.aggregate_doc is not None:
        cells = _cells_from_aggregate(rec.aggregate_doc)
        source = rec.source
    else:
        cells = await _cells_from_live_children(api, base_dir, namespace, name)
        source = "live"
    return CellAggregatesResponse(
        dimensions=spec_summary.dimensions,
        cells=cells,
        source=source,  # type: ignore[arg-type]
    )


async def _list_sweep_epochs_impl(
    base_dir: Path, namespace: str, name: str
) -> SweepEpochsResponse:
    runs = await list_sweep_epochs_async(base_dir, namespace, name)
    return SweepEpochsResponse(
        epochs=[
            SweepEpochSummary(
                epoch=r.epoch,
                is_latest=r.is_latest,
                mtime_epoch=r.mtime_epoch,
                file_count=r.file_count,
            )
            for r in runs
        ]
    )


def _children_manifest_from_doc(
    doc: dict[str, Any], epoch: str | None
) -> ChildrenManifestResponse:
    """Build a ChildrenManifestResponse from a ``children.json``-shaped dict.

    Accepts the disk envelope ``{"sweep_run_epoch": "...", "children": [...]}``
    that is also embedded verbatim into ``status.aggregate.children`` on the
    live CR — both the live (CR) and archived (PVC) read paths converge on
    this shape.
    """
    return ChildrenManifestResponse(
        sweep_run_epoch=str(doc.get("sweep_run_epoch") or epoch or ""),
        children=[
            ChildrenManifestEntry(
                namespace=c.get("namespace", ""),
                name=c.get("name", ""),
                variation_index=int(c.get("variation_index") or 0),
                variation_label=c.get("variation_label") or "",
                trial_index=c.get("trial_index"),
                child_run_epoch=str(c.get("child_run_epoch") or ""),
            )
            for c in (doc.get("children") or [])
            if isinstance(c, dict)
        ],
    )


async def _get_children_impl(
    api: ApiClient,
    base_dir: Path,
    *,
    namespace: str,
    name: str,
    epoch: str | None,
) -> ChildrenManifestResponse:
    """Resolve the per-epoch children manifest, preferring the live CR.

    The sweep-controller writes ``children.json`` to its own pod-local PVC
    *and* embeds the same envelope at ``status.aggregate.children`` on the
    parent AIPerfSweep CR. The operator pod reading this route does NOT
    share a PVC with the controller pod, so the on-disk file is invisible
    here for live sweeps — the CR is the only source the operator can
    actually observe. Read the CR first; fall back to disk only for
    archived (post-TTL) sweeps where the CR is gone but the controller's
    PVC was promoted to a shared archive.

    Returns 404 only when neither half has data — the prior disk-only
    implementation 404'd every live sweep regardless of CR state.
    """
    if epoch is None:
        rec = await find_any_sweep(api, base_dir, namespace, name)
        if rec is not None and rec.raw_status:
            aggregate = rec.raw_status.get("aggregate")
            if isinstance(aggregate, dict):
                children_doc = aggregate.get("children")
                if isinstance(children_doc, dict) and isinstance(
                    children_doc.get("children"), list
                ):
                    return _children_manifest_from_doc(children_doc, epoch=epoch)

        if rec is not None:
            live = await children_manifest_from_live_aiperfjobs(api, namespace, name)
            if live is not None:
                return live

    sweep_dir = resolve_sweep_dir(base_dir, namespace, name, epoch=epoch)
    if sweep_dir is None:
        raise HTTPException(
            404, f"Sweep epoch not found: {namespace}/{name} epoch={epoch}"
        )
    p = sweep_dir / "children.json"
    if not p.is_file():
        raise HTTPException(
            404, f"children.json missing for {namespace}/{name} epoch={epoch}"
        )
    try:
        doc = orjson.loads(p.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        raise HTTPException(503, f"children.json unreadable: {e}") from e
    return _children_manifest_from_doc(doc, epoch=epoch)


def _register_sweep_read_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
    base_dir: Path,
) -> None:
    """Register the sweep list, detail, and epoch-listing endpoints."""

    @router.get("/sweeps", response_model=SweepListResponse)
    async def list_sweeps() -> SweepListResponse:
        return await _list_sweeps_impl(require_api(), base_dir)

    @router.get("/sweeps/{namespace}/{name}", response_model=SweepDetailResponse)
    async def get_sweep(
        namespace: str, name: str, epoch: str | None = None
    ) -> SweepDetailResponse:
        validate_results_path_params(namespace, name)
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        return await _get_sweep_impl(
            require_api(), base_dir, namespace, name, epoch=epoch
        )

    @router.get(
        "/sweeps/{namespace}/{name}/epochs",
        response_model=SweepEpochsResponse,
        response_model_by_alias=True,
    )
    async def list_sweep_epochs_endpoint(
        namespace: str, name: str
    ) -> SweepEpochsResponse:
        validate_results_path_params(namespace, name)
        return await _list_sweep_epochs_impl(base_dir, namespace, name)


def _register_sweep_cell_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
    base_dir: Path,
) -> None:
    """Register the per-cell aggregate and children-manifest endpoints."""

    @router.get(
        "/sweeps/{namespace}/{name}/cells",
        response_model=CellAggregatesResponse,
    )
    async def get_sweep_cells(
        namespace: str, name: str, epoch: str | None = None
    ) -> CellAggregatesResponse:
        validate_results_path_params(namespace, name)
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        return await _get_cells_impl(
            require_api(), base_dir, namespace, name, epoch=epoch
        )

    @router.get(
        "/sweeps/{namespace}/{name}/children",
        response_model=ChildrenManifestResponse,
        response_model_by_alias=True,
    )
    async def get_sweep_children(
        namespace: str, name: str, epoch: str | None = None
    ) -> ChildrenManifestResponse:
        validate_results_path_params(namespace, name)
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        return await _get_children_impl(
            require_api(), base_dir, namespace=namespace, name=name, epoch=epoch
        )


def create_sweeps_router(
    api_holder: list[ApiClient | None] | None = None,
    results_dir: Path | None = None,
) -> APIRouter:
    """Build the sweeps router. Mirrors :func:`create_jobs_router`'s shape."""
    _holder = api_holder if api_holder is not None else [None]
    _base_dir = results_dir if results_dir is not None else Path("/data")
    router = APIRouter(prefix="/api/v1", tags=["sweeps"])

    def _require_api() -> ApiClient:
        api = _holder[0] if _holder else None
        if api is None:
            raise HTTPException(
                503,
                "Kubernetes API client not yet initialized by FastAPI lifespan; "
                "retry in a few seconds or check /healthz",
            )
        return api

    _register_sweep_read_routes(router, _require_api, _base_dir)
    register_sweep_artifact_routes(router, _base_dir)
    _register_sweep_cell_routes(router, _require_api, _base_dir)
    register_diagnostics_routes(router, _require_api)

    return router
