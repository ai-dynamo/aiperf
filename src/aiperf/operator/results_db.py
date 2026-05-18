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
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import zstandard

from aiperf.operator import runs_index
from aiperf.operator.results_layout import (
    list_run_epochs,
    resolve_latest,
    resolve_run_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_COMPARE_METRICS = list(runs_index._NARROW_METRICS)


class ResultsDB:
    """Thin facade over runs_index. Stateless — the DB is module-global."""

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir

    def close(self) -> None:
        # runs_index lifecycle is managed by the operator startup hook.
        pass

    async def _ensure_readonly_index(self) -> bool:
        if runs_index.is_open():
            return True
        try:
            await runs_index.open_readonly(self._results_dir / ".aiperf_index.sqlite")
        except (RuntimeError, OSError, sqlite3.Error) as exc:
            logger.debug("runs_index read-only open unavailable: %s", exc)
            return False
        return True

    async def leaderboard(self, *args, **kwargs):
        if not await self._ensure_readonly_index():
            return self._leaderboard_from_disk(*args, **kwargs)
        rows = await runs_index.leaderboard(*args, **kwargs)
        return rows or self._leaderboard_from_disk(*args, **kwargs)

    async def history(self, *args, **kwargs):
        if not await self._ensure_readonly_index():
            return self._history_from_disk(*args, **kwargs)
        rows = await runs_index.history(*args, **kwargs)
        return rows or self._history_from_disk(*args, **kwargs)

    async def compare(self, *args, **kwargs):
        if not await self._ensure_readonly_index():
            return self._compare_from_disk(*args, **kwargs)
        rows = await runs_index.compare(*args, **kwargs)
        return rows or self._compare_from_disk(*args, **kwargs)

    async def summary(
        self,
        namespace: str,
        job_id: str,
        *,
        epoch: str | None = None,
    ) -> dict[str, Any] | None:
        # epoch=None means "latest" — pull from is_latest column
        if not await self._ensure_readonly_index():
            return await self._summary_from_disk(namespace, job_id, epoch)

        if epoch is None:
            row = await runs_index.get_latest_run(namespace, job_id)
            if row is None:
                return await self._summary_from_disk(namespace, job_id, None)
            epoch = row.epoch

        blob = await runs_index.get_summary_blob(namespace, job_id, epoch)
        if blob:
            return orjson.loads(runs_index.zstd_decompress(blob))
        return await self._summary_from_disk(namespace, job_id, epoch)

    def _leaderboard_from_disk(
        self,
        metric: str = "request_throughput",
        stat: str = "avg",
        order: str = "desc",
        limit: int = 20,
        *,
        epoch: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            runs_index._validate_identifier(metric)
            runs_index._validate_identifier(stat)
        except ValueError:
            return []

        rows: list[dict[str, Any]] = []
        for namespace, job_id, run_epoch, summary in self._iter_disk_summaries(epoch):
            value, unit = self._metric_stat(summary, metric, stat)
            if value is None:
                continue
            model, endpoint = runs_index._extract_model_endpoint(
                {"benchmark": summary.get("input_config", {}) or {}}
            )
            rows.append(
                {
                    "namespace": namespace,
                    "job_id": job_id,
                    "epoch": run_epoch,
                    "value": value,
                    "unit": unit,
                    "start_time": summary.get("start_time"),
                    "end_time": summary.get("end_time"),
                    "model": model,
                    "endpoint": endpoint,
                }
            )
        rows.sort(
            key=lambda row: row["value"],
            reverse=(order.lower() == "desc"),
        )
        return rows[:limit]

    def _history_from_disk(
        self,
        *,
        model: str | None = None,
        endpoint: str | None = None,
        metric: str = "request_throughput",
        stat: str = "avg",
        limit: int = 100,
        epoch: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            runs_index._validate_identifier(metric)
            runs_index._validate_identifier(stat)
        except ValueError:
            return []

        rows: list[dict[str, Any]] = []
        for namespace, job_id, run_epoch, summary in self._iter_disk_summaries(epoch):
            value, unit = self._metric_stat(summary, metric, stat)
            if value is None:
                continue
            row_model, row_endpoint = runs_index._extract_model_endpoint(
                {"benchmark": summary.get("input_config", {}) or {}}
            )
            if model and (row_model is None or model not in row_model):
                continue
            if endpoint and (row_endpoint is None or endpoint not in row_endpoint):
                continue
            rows.append(
                {
                    "namespace": namespace,
                    "job_id": job_id,
                    "epoch": run_epoch,
                    "value": value,
                    "unit": unit,
                    "start_time": summary.get("start_time"),
                    "model": row_model,
                    "endpoint": row_endpoint,
                }
            )
        rows.sort(key=lambda row: row.get("start_time") or "")
        return rows[:limit]

    def _compare_from_disk(
        self,
        job_ids: list[str],
        metrics: list[str] | None = None,
        *,
        epoch: str | None = None,
    ) -> list[dict[str, Any]]:
        if not job_ids:
            return []
        if metrics is None:
            metrics = list(DEFAULT_COMPARE_METRICS)
        try:
            for metric in metrics:
                runs_index._validate_identifier(metric)
        except ValueError:
            return []

        bare_job_ids, qualified_refs = runs_index._split_compare_job_ids(job_ids)
        qualified = set(qualified_refs)
        rows: list[dict[str, Any]] = []
        for namespace, job_id, run_epoch, summary in self._iter_disk_summaries(epoch):
            if job_id not in bare_job_ids and (namespace, job_id) not in qualified:
                continue
            row_model, row_endpoint = runs_index._extract_model_endpoint(
                {"benchmark": summary.get("input_config", {}) or {}}
            )
            gpu_count, gpu_name = runs_index._summarize_telemetry(
                summary.get("telemetry_data")
            )
            row: dict[str, Any] = {
                "namespace": namespace,
                "job_id": job_id,
                "epoch": run_epoch,
                "start_time": summary.get("start_time"),
                "model": row_model,
                "endpoint": row_endpoint,
                "gpu_count": gpu_count,
                "gpu_name": gpu_name,
            }
            for metric in metrics:
                metric_data = summary.get(metric) or {}
                for metric_stat in ("avg", "p50", "p99"):
                    row[f"{metric}_{metric_stat}"] = metric_data.get(metric_stat)
                row[f"{metric}_unit"] = metric_data.get("unit")
            rows.append(row)
        return rows

    def _iter_disk_summaries(
        self, epoch: str | None
    ) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
        if not self._results_dir.is_dir():
            return
        for namespace_dir in self._results_dir.iterdir():
            if not namespace_dir.is_dir():
                continue
            for job_dir in namespace_dir.iterdir():
                if not job_dir.is_dir() or job_dir.name == "sweeps":
                    continue
                epochs = (
                    [epoch]
                    if epoch is not None
                    else [
                        resolve_latest(
                            self._results_dir, namespace_dir.name, job_dir.name
                        )
                    ]
                )
                for run_epoch in epochs:
                    if run_epoch is None:
                        continue
                    if run_epoch not in list_run_epochs(
                        self._results_dir, namespace_dir.name, job_dir.name
                    ):
                        continue
                    summary = self._read_summary_file(job_dir / run_epoch)
                    if summary is not None:
                        yield namespace_dir.name, job_dir.name, run_epoch, summary

    def _read_summary_file(self, run_dir: Path) -> dict[str, Any] | None:
        zst = run_dir / "profile_export_aiperf.json.zst"
        raw = run_dir / "profile_export_aiperf.json"
        try:
            if zst.exists():
                return orjson.loads(runs_index.zstd_decompress(zst.read_bytes()))
            if raw.exists():
                return orjson.loads(raw.read_bytes())
        except (OSError, orjson.JSONDecodeError, zstandard.ZstdError) as exc:
            logger.warning("cannot read summary at %s: %s", run_dir, exc)
        return None

    def _metric_stat(
        self, summary: dict[str, Any], metric: str, stat: str
    ) -> tuple[float | None, str | None]:
        metric_data = summary.get(metric)
        if not isinstance(metric_data, dict):
            return None, None
        return metric_data.get(stat), metric_data.get("unit")

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
