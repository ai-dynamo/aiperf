# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI router for /api/v1/sweeps* — read-only AIPerfSweep view.

Dual-backed via :mod:`aiperf.operator.sweep_union`: every endpoint
returns the same shape regardless of whether the parent CR exists or
the data is reconstructed from the archived ``aggregate.json``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, HTTPException
from kubernetes_asyncio.client import ApiClient

from aiperf.operator.job_union import list_all_jobs
from aiperf.operator.routers.sweeps_models import (
    DimensionInfo,
    SpecSummary,
    SweepDetailResponse,
    SweepListResponse,
    SweepSummary,
)
from aiperf.operator.sweep_union import (
    SweepRecord,
    find_any_sweep,
    list_all_sweeps,
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
        age_seconds=rec.age_seconds,
        model=rec.model,
    )


def _dimensions_from_live_spec(spec: dict[str, Any]) -> list[DimensionInfo]:
    sweep = spec.get("sweep") or {}
    axes = sweep.get("axes") or sweep.get("dimensions") or []
    out: list[DimensionInfo] = []
    for axis in axes:
        if not isinstance(axis, dict):
            continue
        nm = axis.get("name")
        vals = axis.get("values") or []
        if isinstance(nm, str):
            out.append(DimensionInfo(name=nm, values=list(vals)))
    return out


def _spec_summary_from_record(rec: SweepRecord) -> SpecSummary:
    """Build a SpecSummary from whichever side of the union is available."""
    if rec.raw_spec:
        sweep = rec.raw_spec.get("sweep") or {}
        return SpecSummary(
            sweep_type=str(sweep.get("type") or "grid"),  # type: ignore[arg-type]
            dimensions=_dimensions_from_live_spec(rec.raw_spec),
            multi_run=rec.raw_spec.get("multiRun"),
            convergence=rec.raw_spec.get("convergence"),
        )
    if rec.aggregate_doc is not None:
        snap = rec.aggregate_doc.get("spec_snapshot") or {}
        dims_raw = snap.get("dimensions") or []
        dims = [
            DimensionInfo(name=d["name"], values=list(d.get("values") or []))
            for d in dims_raw
            if isinstance(d, dict) and isinstance(d.get("name"), str)
        ]
        return SpecSummary(
            sweep_type=str(snap.get("sweep_type") or "grid"),  # type: ignore[arg-type]
            dimensions=dims,
            multi_run=snap.get("multi_run"),
            convergence=snap.get("convergence"),
        )
    return SpecSummary(
        sweep_type="grid", dimensions=[], multi_run=None, convergence=None
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
    api: ApiClient, base_dir: Path, namespace: str, name: str
) -> SweepDetailResponse:
    rec = await find_any_sweep(api, base_dir, namespace, name)
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

    spec_summary = _spec_summary_from_record(rec)

    children_records = await list_all_jobs(api, base_dir, all_namespaces=False)
    children = [
        j.model_dump(by_alias=True)
        for j in children_records
        if getattr(j, "sweep_name", None) == name and j.namespace == namespace
    ]

    return SweepDetailResponse(
        sweep=_summary(rec),
        status=status,
        spec_summary=spec_summary,
        children=children,
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

    @router.get("/sweeps", response_model=SweepListResponse)
    async def list_sweeps() -> SweepListResponse:
        return await _list_sweeps_impl(_require_api(), _base_dir)

    @router.get("/sweeps/{namespace}/{name}", response_model=SweepDetailResponse)
    async def get_sweep(namespace: str, name: str) -> SweepDetailResponse:
        return await _get_sweep_impl(_require_api(), _base_dir, namespace, name)

    return router
