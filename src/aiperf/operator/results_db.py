# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytics facade for stored benchmark results, backed by runs_index.

This module is now a thin compatibility wrapper around runs_index — the
DuckDB JSON-glob path has been removed in favour of indexed flat-column
SELECTs. The wrapper exists so the FastAPI routers in
``routers/results_analytics.py`` can keep their existing dependency-injected
``get_db()`` factory without rewiring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson

from aiperf.operator import runs_index
from aiperf.operator.results_layout import resolve_run_dir

logger = logging.getLogger(__name__)

DEFAULT_COMPARE_METRICS = list(runs_index._NARROW_METRICS)


class ResultsDB:
    """Thin facade over runs_index. Stateless — the DB is module-global."""

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir

    def close(self) -> None:
        # runs_index lifecycle is managed by the operator startup hook.
        pass

    async def leaderboard(self, *args, **kwargs):
        return await runs_index.leaderboard(*args, **kwargs)

    async def history(self, *args, **kwargs):
        return await runs_index.history(*args, **kwargs)

    async def compare(self, *args, **kwargs):
        return await runs_index.compare(*args, **kwargs)

    async def summary(
        self,
        namespace: str,
        job_id: str,
        *,
        epoch: str | None = None,
    ) -> dict[str, Any] | None:
        # epoch=None means "latest" — pull from is_latest column
        if epoch is None:
            row = await runs_index.get_latest_run(namespace, job_id)
            if row is None:
                return await self._summary_from_disk(namespace, job_id, None)
            epoch = row.epoch

        blob = await runs_index.get_summary_blob(namespace, job_id, epoch)
        if blob:
            return orjson.loads(runs_index.zstd_decompress(blob))
        return await self._summary_from_disk(namespace, job_id, epoch)

    async def _summary_from_disk(
        self,
        namespace: str,
        job_id: str,
        epoch: str | None,
    ) -> dict[str, Any] | None:
        """Fallback when metrics_json is null (mid-completion race)."""
        run_dir = resolve_run_dir(self._results_dir, namespace, job_id, epoch)
        if run_dir is None:
            return None
        zst = run_dir / "profile_export_aiperf.json.zst"
        raw = run_dir / "profile_export_aiperf.json"
        if zst.exists():
            return orjson.loads(runs_index.zstd_decompress(zst.read_bytes()))
        if raw.exists():
            return orjson.loads(raw.read_bytes())
        return None
