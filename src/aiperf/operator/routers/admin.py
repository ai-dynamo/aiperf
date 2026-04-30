# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator admin endpoints — index stats and manual rebuild."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from aiperf.operator import runs_index


class IndexStatsResponse(BaseModel):
    runs_count: int = Field(description="Total rows in the runs table.")
    sweep_variations_count: int = Field(description="Total rows in sweep_variations.")
    db_bytes: int = Field(description="On-disk size of the SQLite file.")
    last_bootstrap_unix: int | None = Field(
        description="Unix epoch of the last bootstrap completion, or null if never run."
    )
    schema_version: int = Field(description="Compiled-in schema version.")


class IndexRebuildResponse(BaseModel):
    runs_indexed: int = Field(description="Runs ingested by the rebuild walk.")
    sweep_variations_indexed: int = Field(description="Sweep variations ingested.")
    duration_seconds: float = Field(description="Wall-clock duration of the rebuild.")


def create_admin_router(base_dir: Path, db_path: Path) -> APIRouter:
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

    @router.get("/stats", response_model=IndexStatsResponse)
    async def stats() -> IndexStatsResponse:
        s = await runs_index.stats(db_path)
        return IndexStatsResponse(**s)

    @router.post("/rebuild", response_model=IndexRebuildResponse)
    async def rebuild() -> IndexRebuildResponse:
        result = await runs_index.bootstrap(base_dir, force=True)
        return IndexRebuildResponse(
            runs_indexed=result.runs_indexed,
            sweep_variations_indexed=result.sweep_variations_indexed,
            duration_seconds=result.duration_seconds,
        )

    return router
