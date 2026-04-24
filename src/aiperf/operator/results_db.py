# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DuckDB-backed analytics for stored benchmark results.

Queries result files (JSON, JSONL, Parquet, CSV) directly from the PVC
without requiring a separate ingest step. DuckDB reads zstd-compressed
files natively.

All queries run in a thread pool to avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_COMPARE_METRICS = [
    "request_throughput",
    "request_latency",
    "time_to_first_token",
    "output_token_throughput",
    "inter_token_latency",
]


class ResultsDB:
    """DuckDB query engine for benchmark results stored on the PVC.

    Reads result files directly — no ETL or schema migration needed.
    Thread-safe: each query gets its own cursor from a shared connection.
    """

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir
        self._conn = duckdb.connect(":memory:", read_only=False)
        self._conn.execute("SET enable_progress_bar = false")

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def _find_summary_files(self, glob: str = "profile_export_aiperf.json") -> str:
        """Build a glob pattern matching summary files across all jobs.

        Handles both compressed (.zst) and uncompressed variants.
        Returns a DuckDB-compatible glob string.
        """
        base = str(self._results_dir)
        # DuckDB's read_json supports glob patterns
        zst = f"{base}/*/*/{glob}.zst"
        raw = f"{base}/*/*/{glob}"

        zst_exists = any(self._results_dir.glob(f"*/*/{glob}.zst"))
        raw_exists = any(self._results_dir.glob(f"*/*/{glob}"))

        if zst_exists and raw_exists:
            return f"['{zst}', '{raw}']"
        if zst_exists:
            return f"'{zst}'"
        if raw_exists:
            return f"'{raw}'"
        return "''"

    def _extract_job_path_parts(self) -> str:
        """SQL expression to extract namespace and job_id from the filename path."""
        return (
            "string_split(filename, '/')[-3] AS namespace, "
            "string_split(filename, '/')[-2] AS job_id"
        )

    async def leaderboard(
        self,
        metric: str = "request_throughput",
        stat: str = "avg",
        order: str = "desc",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Rank all benchmark runs by a metric.

        Args:
            metric: Metric name (e.g. request_throughput, request_latency, time_to_first_token).
            stat: Statistic to rank by (avg, p50, p99, min, max).
            order: Sort order (asc or desc).
            limit: Maximum results to return.
        """
        files = self._find_summary_files()
        if files == "''":
            return []

        _validate_identifier(metric)
        _validate_identifier(stat)
        order_dir = "DESC" if order.lower() == "desc" else "ASC"

        sql = f"""
            SELECT
                {self._extract_job_path_parts()},
                t.{metric}.{stat}::DOUBLE AS value,
                t.{metric}.unit AS unit,
                t.start_time::VARCHAR AS start_time,
                t.end_time::VARCHAR AS end_time,
                t.input_config.models.items[1].name AS model,
                t.input_config.endpoint.urls[1] AS endpoint
            FROM (
                SELECT *, filename
                FROM read_json({files},
                    compression='auto_detect',
                    union_by_name=true)
            ) t
            WHERE t.{metric}.{stat} IS NOT NULL
            ORDER BY value {order_dir}
            LIMIT {int(limit)}
        """  # noqa: S608

        return await self._query(sql)

    async def history(
        self,
        *,
        model: str | None = None,
        endpoint: str | None = None,
        metric: str = "request_throughput",
        stat: str = "avg",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get metric values over time, optionally filtered by model or endpoint.

        Args:
            model: Filter by model name (substring match).
            endpoint: Filter by endpoint URL (substring match).
            metric: Metric to track.
            stat: Statistic to return.
            limit: Maximum results.
        """
        files = self._find_summary_files()
        if files == "''":
            return []

        _validate_identifier(metric)
        _validate_identifier(stat)

        where_clauses = [f"t.{metric}.{stat} IS NOT NULL"]
        if model:
            where_clauses.append(
                f"t.input_config.models.items[1].name ILIKE '%{_escape_like(model)}%' ESCAPE '\\'"
            )
        if endpoint:
            where_clauses.append(
                f"t.input_config.endpoint.urls[1] ILIKE '%{_escape_like(endpoint)}%' ESCAPE '\\'"
            )

        where = " AND ".join(where_clauses)

        sql = f"""
            SELECT
                {self._extract_job_path_parts()},
                t.{metric}.{stat}::DOUBLE AS value,
                t.{metric}.unit AS unit,
                t.start_time::VARCHAR AS start_time,
                t.input_config.models.items[1].name AS model,
                t.input_config.endpoint.urls[1] AS endpoint
            FROM (
                SELECT *, filename
                FROM read_json({files},
                    compression='auto_detect',
                    union_by_name=true)
            ) t
            WHERE {where}
            ORDER BY start_time ASC
            LIMIT {int(limit)}
        """  # noqa: S608

        return await self._query(sql)

    async def compare(
        self,
        job_ids: list[str],
        metrics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Compare specific jobs side-by-side across multiple metrics.

        Args:
            job_ids: List of job IDs to compare.
            metrics: Metrics to include (default: key performance metrics).
        """
        if not job_ids:
            return []

        files = self._find_summary_files()
        if files == "''":
            return []

        if metrics is None:
            metrics = list(DEFAULT_COMPARE_METRICS)

        for m in metrics:
            _validate_identifier(m)

        # Build job ID filter. IN uses literal equality, so only escape quotes
        # (LIKE-style %/_ escaping would corrupt job_ids containing underscores).
        job_id_list = ", ".join("'" + j.replace("'", "''") + "'" for j in job_ids)

        base_rows = await self._query(self._compare_base_sql(files, job_id_list))
        if not base_rows:
            return []

        # Key by (namespace, job_id) — two jobs with the same job_id in
        # different namespaces must not overwrite each other.
        job_data: dict[tuple[str, str], dict[str, Any]] = {
            (row.get("namespace", ""), row.get("job_id", "")): row for row in base_rows
        }

        # Query each metric separately to handle missing columns gracefully
        # (e.g., non-streaming has no TTFT).
        for m in metrics:
            metric_rows = await self._query(
                self._compare_metric_sql(files, job_id_list, m)
            )
            for row in metric_rows:
                key = (row.get("namespace", ""), row.get("job_id", ""))
                if key in job_data:
                    job_data[key].update(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("job_id", "namespace")
                        }
                    )

        return list(job_data.values())

    def _compare_base_sql(self, files: str, job_id_list: str) -> str:
        """Base SELECT for compare(): job identity and run metadata only."""
        return f"""
            SELECT
                {self._extract_job_path_parts()},
                t.start_time::VARCHAR AS start_time,
                t.input_config.models.items[1].name AS model,
                t.input_config.endpoint.urls[1] AS endpoint
            FROM (
                SELECT *, filename
                FROM read_json({files},
                    compression='auto_detect',
                    union_by_name=true)
            ) t
            WHERE string_split(filename, '/')[-2] IN ({job_id_list})
        """  # noqa: S608

    def _compare_metric_sql(self, files: str, job_id_list: str, metric: str) -> str:
        """Per-metric SELECT for compare(): extracts avg/p50/p99/unit via JSON."""
        _validate_identifier(metric)
        json_expr = f"to_json(t.{metric})"
        return f"""
            SELECT
                string_split(filename, '/')[-3] AS namespace,
                string_split(filename, '/')[-2] AS job_id,
                TRY_CAST({json_expr}->>'avg' AS DOUBLE) AS {metric}_avg,
                TRY_CAST({json_expr}->>'p50' AS DOUBLE) AS {metric}_p50,
                TRY_CAST({json_expr}->>'p99' AS DOUBLE) AS {metric}_p99,
                {json_expr}->>'unit' AS {metric}_unit
            FROM (
                SELECT *, filename
                FROM read_json({files},
                    compression='auto_detect',
                    union_by_name=true)
            ) t
            WHERE string_split(filename, '/')[-2] IN ({job_id_list})
        """  # noqa: S608

    async def summary(self, namespace: str, job_id: str) -> dict[str, Any] | None:
        """Get the full aggregated summary for a single job.

        Args:
            namespace: Kubernetes namespace.
            job_id: Job identifier.
        """
        job_dir = self._results_dir / namespace / job_id
        if not job_dir.is_dir():
            return None

        zst = job_dir / "profile_export_aiperf.json.zst"
        raw = job_dir / "profile_export_aiperf.json"

        if zst.exists():
            path = str(zst)
        elif raw.exists():
            path = str(raw)
        else:
            return None

        sql = f"""
            SELECT *
            FROM read_json('{path}',
                compression='auto_detect')
        """

        rows = await self._query(sql)
        return rows[0] if rows else None

    async def _query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a SQL query in a thread and return results as dicts."""

        def _run() -> list[dict[str, Any]]:
            # duckdb connections are not safe for concurrent execute() from
            # multiple threads — take an independent cursor per query.
            cur = self._conn.cursor()
            try:
                result = cur.execute(sql)
                columns = [desc[0] for desc in result.description]
                return [
                    dict(zip(columns, row, strict=True)) for row in result.fetchall()
                ]
            except duckdb.Error as e:
                logger.warning(f"DuckDB query failed: {e}")
                return []
            finally:
                cur.close()

        return await asyncio.to_thread(_run)


# --- Input validation ---

_VALID_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz_0123456789")


def _validate_identifier(name: str) -> None:
    """Validate that a name is a safe SQL identifier (lowercase alpha + underscore)."""
    if not name or not all(c in _VALID_IDENTIFIER_CHARS for c in name.lower()):
        raise ValueError(
            f"Invalid SQL identifier {name!r}: must be non-empty and contain only "
            "lowercase a-z, 0-9, or underscore"
        )


def _escape_like(value: str) -> str:
    """Escape special characters for SQL LIKE/ILIKE patterns."""
    return value.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
