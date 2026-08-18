# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator admin endpoints — index stats and manual rebuild."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Depends as DependsParam
from pydantic import BaseModel, Field

from aiperf.common.finite import FiniteFloat
from aiperf.operator import runs_index


class IndexStatsResponse(BaseModel):
    runs_count: int = Field(ge=0, description="Total rows in the runs table.")
    sweep_variations_count: int = Field(
        ge=0, description="Total rows in sweep_variations."
    )
    db_bytes: int = Field(ge=0, description="On-disk size of the SQLite file.")
    last_bootstrap_unix: int | None = Field(
        ge=0,
        description="Unix epoch of the last bootstrap completion, or null if never run.",
    )
    schema_version: int = Field(ge=0, description="Compiled-in schema version.")


class IndexRebuildResponse(BaseModel):
    runs_indexed: int = Field(ge=0, description="Runs ingested by the rebuild walk.")
    sweep_variations_indexed: int = Field(
        ge=0, description="Sweep variations ingested."
    )
    duration_seconds: FiniteFloat = Field(
        ge=0.0, description="Wall-clock duration of the rebuild."
    )


class IndexRunRowResponse(BaseModel):
    """Narrow-column projection of a single ``runs`` row, used by the
    K8s-vs-local audit suite's ``index_consistency`` check.

    Exposes the six ``DEFAULT_COMPARE_METRICS`` flat columns plus identity
    fields so the audit can confirm that what the index stored matches what
    landed on disk. Intentionally narrow — the full metrics blob is already
    available via ``/api/v1/analytics/summary``.
    """

    namespace: str = Field(description="K8s namespace owning the AIPerfJob.")
    job_id: str = Field(description="AIPerfJob CR name.")
    epoch: str = Field(description="Run epoch (timestamp directory name).")
    phase: str = Field(description="Last-known run phase (Succeeded/Failed/...).")
    request_throughput_avg: FiniteFloat | None = Field(
        default=None, description="Average request throughput for the indexed run."
    )
    request_throughput_p50: FiniteFloat | None = Field(
        default=None, description="Median request throughput for the indexed run."
    )
    request_throughput_p99: FiniteFloat | None = Field(
        default=None, description="P99 request throughput for the indexed run."
    )
    request_latency_avg: FiniteFloat | None = Field(
        default=None, description="Average request latency for the indexed run."
    )
    request_latency_p50: FiniteFloat | None = Field(
        default=None, description="Median request latency for the indexed run."
    )
    request_latency_p99: FiniteFloat | None = Field(
        default=None, description="P99 request latency for the indexed run."
    )
    time_to_first_token_avg: FiniteFloat | None = Field(
        default=None,
        description="Average time-to-first-token latency for the indexed run.",
    )
    time_to_first_token_p50: FiniteFloat | None = Field(
        default=None,
        description="Median time-to-first-token latency for the indexed run.",
    )
    time_to_first_token_p99: FiniteFloat | None = Field(
        default=None, description="P99 time-to-first-token latency for the indexed run."
    )
    output_token_throughput_avg: FiniteFloat | None = Field(
        default=None, description="Average output-token throughput for the indexed run."
    )
    output_token_throughput_p50: FiniteFloat | None = Field(
        default=None, description="Median output-token throughput for the indexed run."
    )
    output_token_throughput_p99: FiniteFloat | None = Field(
        default=None, description="P99 output-token throughput for the indexed run."
    )
    output_token_throughput_per_user_avg: FiniteFloat | None = Field(
        default=None,
        description="Average per-user output-token throughput for the indexed run.",
    )
    output_token_throughput_per_user_p50: FiniteFloat | None = Field(
        default=None,
        description="Median per-user output-token throughput for the indexed run.",
    )
    output_token_throughput_per_user_p99: FiniteFloat | None = Field(
        default=None,
        description="P99 per-user output-token throughput for the indexed run.",
    )
    inter_token_latency_avg: FiniteFloat | None = Field(
        default=None, description="Average inter-token latency for the indexed run."
    )
    inter_token_latency_p50: FiniteFloat | None = Field(
        default=None, description="Median inter-token latency for the indexed run."
    )
    inter_token_latency_p99: FiniteFloat | None = Field(
        default=None, description="P99 inter-token latency for the indexed run."
    )


async def _reject_request_body(request: Request) -> None:
    body = await request.body()
    if body:
        raise HTTPException(422, "Request body is not allowed for this endpoint.")


def create_admin_router(
    base_dir: Path,
    db_path: Path,
    mutating_dependencies: Sequence[DependsParam] = (),
    *,
    allow_rebuild: bool = True,
) -> APIRouter:
    """Build the /admin/index router bound to ``base_dir`` and ``db_path``.

    The router exposes:
    - ``GET /admin/index/stats`` — current row counts, DB size, schema version,
      and last-bootstrap epoch. Useful for confirming the index is populated
      after operator startup.
    - ``POST /admin/index/rebuild`` — drop and rewalk the PVC, re-ingesting
      every run + sweep variation. Used as the manual recovery hatch when the
      DB falls out of sync with disk (e.g. after an external file restore).
    """
    router = APIRouter(prefix="/admin/index", tags=["admin"])
    rebuild_lock = asyncio.Lock()

    @router.get("/stats", response_model=IndexStatsResponse)
    async def stats(request: Request) -> IndexStatsResponse:
        await _reject_request_body(request)
        if not runs_index.is_open():
            raise HTTPException(503, "Runs index not available yet")
        s = await runs_index.stats(db_path)
        return IndexStatsResponse(**s)

    @router.post(
        "/rebuild",
        response_model=IndexRebuildResponse,
        dependencies=list(mutating_dependencies),
    )
    async def rebuild(request: Request) -> IndexRebuildResponse:
        await _reject_request_body(request)
        if not allow_rebuild:
            raise HTTPException(
                503,
                "Index rebuild is disabled in the read-only results-server sidecar; "
                "run it from the operator writer process.",
            )
        if rebuild_lock.locked():
            raise HTTPException(409, "Index rebuild already in progress")

        await rebuild_lock.acquire()
        try:
            result = await runs_index.bootstrap(base_dir, force=True)
        finally:
            rebuild_lock.release()
        return IndexRebuildResponse(
            runs_indexed=result.runs_indexed,
            sweep_variations_indexed=result.sweep_variations_indexed,
            duration_seconds=result.duration_seconds,
        )

    @router.get("/run/{namespace}/{job_id}", response_model=IndexRunRowResponse)
    async def run_row(
        namespace: str, job_id: str, request: Request
    ) -> IndexRunRowResponse:
        """Return the narrow-column projection of the latest ``runs`` row.

        Used by the audit suite to verify the index's stored flat columns
        match the on-disk ``profile_export_aiperf.json``. 404 if no row
        exists for ``(namespace, job_id)`` yet.
        """
        await _reject_request_body(request)
        narrow = await runs_index.get_run_narrow_metrics(namespace, job_id)
        if narrow is None:
            raise HTTPException(404, f"No index row for {namespace}/{job_id}")
        return IndexRunRowResponse(namespace=namespace, job_id=job_id, **narrow)

    return router
