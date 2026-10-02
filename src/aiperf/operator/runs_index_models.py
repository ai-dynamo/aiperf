# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row dataclasses for the runs/sweep_variations SQLite index.

Plain ``@dataclass(slots=True)`` rather than Pydantic — these are constructed
from raw ``sqlite3.Row`` tuples on every read, and the Pydantic overhead would
dominate the query cost we're trying to eliminate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RunIndexRow:
    """One row from the ``runs`` table, hydrated for read API consumers."""

    namespace: str
    job_id: str
    epoch: str
    phase: str
    is_latest: bool
    start_time: str | None
    end_time: str | None
    created_unix: int
    mtime_epoch: int | None
    error: str | None
    model: str | None
    endpoint: str | None
    gpu_count: int
    gpu_name: str | None
    file_count: int
    total_size_bytes: int
    sweep_namespace: str | None
    sweep_name: str | None
    sweep_epoch: str | None
    sweep_variation_idx: int | None


@dataclass(slots=True)
class SweepVariationRow:
    """One row from the ``sweep_variations`` table."""

    namespace: str
    sweep_name: str
    sweep_epoch: str
    variation_idx: int
    mode: str
    phase: str | None
    pareto_rank: int | None
    is_best: bool
    child_namespace: str | None
    child_job_id: str | None
    child_epoch: str | None


@dataclass(slots=True)
class BootstrapStats:
    """Returned from ``runs_index.bootstrap()`` — used by the rebuild CLI."""

    runs_indexed: int
    sweep_variations_indexed: int
    duration_seconds: float
