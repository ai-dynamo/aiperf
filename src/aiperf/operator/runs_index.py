# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed index of runs and sweep variations.

Single-writer model: only the operator's kopf-owning process writes; readers
(operator FastAPI workers, results-server sidecar) open the DB read-only.
The single-writer assumption matches the operator's existing single-replica
deployment and the completion-claim mechanic in ``client_cache.py``. If the
operator is ever scaled up, only the kopf-owning process must call write APIs.

The DB lives at ``<RESULTS.DIR>/.aiperf_index.sqlite`` in WAL mode. WAL mode
gives us non-blocking readers across processes; ``BEGIN IMMEDIATE`` plus
``busy_timeout=5000`` serializes writes without explicit locks.

The index is a cache, never a source of truth. Every read site falls back to
a filesystem scan on miss and lazy-backfills the row in the background, so a
corrupt or stale index degrades to slower, never wrong.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite
import orjson
import zstandard

from aiperf.operator.runs_index_models import RunIndexRow, SweepVariationRow

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_DB: aiosqlite.Connection | None = None
_DB_PATH: Path | None = None

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS runs (
    namespace             TEXT    NOT NULL,
    job_id                TEXT    NOT NULL,
    epoch                 TEXT    NOT NULL,
    phase                 TEXT    NOT NULL,
    is_latest             INTEGER NOT NULL DEFAULT 0,
    start_time            TEXT,
    end_time              TEXT,
    created_unix          INTEGER NOT NULL,
    mtime_epoch           INTEGER,
    error                 TEXT,
    model                 TEXT,
    endpoint              TEXT,
    gpu_count             INTEGER NOT NULL DEFAULT 0,
    gpu_name              TEXT,
    file_count            INTEGER NOT NULL DEFAULT 0,
    total_size_bytes      INTEGER NOT NULL DEFAULT 0,
    spec_json             BLOB,
    request_throughput_avg                       REAL,
    request_throughput_p50                       REAL,
    request_throughput_p99                       REAL,
    request_throughput_unit                      TEXT,
    request_latency_avg                          REAL,
    request_latency_p50                          REAL,
    request_latency_p99                          REAL,
    request_latency_unit                         TEXT,
    time_to_first_token_avg                      REAL,
    time_to_first_token_p50                      REAL,
    time_to_first_token_p99                      REAL,
    time_to_first_token_unit                     TEXT,
    output_token_throughput_avg                  REAL,
    output_token_throughput_p50                  REAL,
    output_token_throughput_p99                  REAL,
    output_token_throughput_unit                 TEXT,
    output_token_throughput_per_user_avg         REAL,
    output_token_throughput_per_user_p50         REAL,
    output_token_throughput_per_user_p99         REAL,
    output_token_throughput_per_user_unit        TEXT,
    inter_token_latency_avg                      REAL,
    inter_token_latency_p50                      REAL,
    inter_token_latency_p99                      REAL,
    inter_token_latency_unit                     TEXT,
    metrics_json          BLOB,
    sweep_namespace       TEXT,
    sweep_name            TEXT,
    sweep_epoch           TEXT,
    sweep_variation_idx   INTEGER,
    PRIMARY KEY (namespace, job_id, epoch)
);

CREATE UNIQUE INDEX IF NOT EXISTS runs_one_latest
    ON runs(namespace, job_id) WHERE is_latest = 1;
CREATE INDEX IF NOT EXISTS runs_model        ON runs(model);
CREATE INDEX IF NOT EXISTS runs_start_time   ON runs(start_time);
CREATE INDEX IF NOT EXISTS runs_sweep_link   ON runs(sweep_namespace, sweep_name, sweep_epoch);

CREATE VIEW IF NOT EXISTS runs_latest AS
    SELECT * FROM runs WHERE is_latest = 1;

CREATE TABLE IF NOT EXISTS sweep_variations (
    namespace             TEXT    NOT NULL,
    sweep_name            TEXT    NOT NULL,
    sweep_epoch           TEXT    NOT NULL,
    variation_idx         INTEGER NOT NULL,
    variation_values_json BLOB    NOT NULL,
    mode                  TEXT    NOT NULL,
    phase                 TEXT,
    pareto_rank           INTEGER,
    is_best               INTEGER NOT NULL DEFAULT 0,
    child_namespace       TEXT,
    child_job_id          TEXT,
    child_epoch           TEXT,
    request_throughput_avg                       REAL,
    request_throughput_p50                       REAL,
    request_throughput_p99                       REAL,
    request_throughput_unit                      TEXT,
    request_latency_avg                          REAL,
    request_latency_p50                          REAL,
    request_latency_p99                          REAL,
    request_latency_unit                         TEXT,
    time_to_first_token_avg                      REAL,
    time_to_first_token_p50                      REAL,
    time_to_first_token_p99                      REAL,
    time_to_first_token_unit                     TEXT,
    output_token_throughput_avg                  REAL,
    output_token_throughput_p50                  REAL,
    output_token_throughput_p99                  REAL,
    output_token_throughput_unit                 TEXT,
    output_token_throughput_per_user_avg         REAL,
    output_token_throughput_per_user_p50         REAL,
    output_token_throughput_per_user_p99         REAL,
    output_token_throughput_per_user_unit        TEXT,
    inter_token_latency_avg                      REAL,
    inter_token_latency_p50                      REAL,
    inter_token_latency_p99                      REAL,
    inter_token_latency_unit                     TEXT,
    metrics_json          BLOB,
    PRIMARY KEY (namespace, sweep_name, sweep_epoch, variation_idx)
);

CREATE INDEX IF NOT EXISTS sweep_variations_best   ON sweep_variations(sweep_name, is_best);
CREATE INDEX IF NOT EXISTS sweep_variations_pareto ON sweep_variations(pareto_rank);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


async def open(path: Path) -> None:
    """Open the DB at ``path``, creating + migrating schema as needed.

    Idempotent — calling twice is safe and does not duplicate state.
    """
    global _DB, _DB_PATH

    if _DB is not None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path), isolation_level=None)
    await db.execute("PRAGMA journal_mode = WAL")
    await db.execute("PRAGMA synchronous = NORMAL")
    await db.execute("PRAGMA busy_timeout = 5000")
    await db.executescript(_SCHEMA_V1)

    cur = await db.execute("SELECT value FROM meta WHERE key = 'schema_version'")
    row = await cur.fetchone()
    await cur.close()
    if row is None:
        await db.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
    else:
        # Forward-only migrations live here when SCHEMA_VERSION bumps.
        # Today: only v1, no migration needed.
        existing = int(row[0])
        if existing > SCHEMA_VERSION:
            raise RuntimeError(
                f"runs_index DB at {path} has schema_version={existing} but this "
                f"build only knows up to {SCHEMA_VERSION}. Refusing to open."
            )

    _DB = db
    _DB_PATH = path
    logger.info("runs_index opened at %s (schema_version=%d)", path, SCHEMA_VERSION)


async def close() -> None:
    """Close the DB. Safe to call when never opened."""
    global _DB, _DB_PATH
    if _DB is not None:
        await _DB.close()
    _DB = None
    _DB_PATH = None


def _conn() -> aiosqlite.Connection:
    if _DB is None:
        raise RuntimeError("runs_index.open() has not been called")
    return _DB


async def get_meta(key: str) -> str | None:
    """Read a single ``meta`` row by key."""
    cur = await _conn().execute("SELECT value FROM meta WHERE key = ?", (key,))
    row = await cur.fetchone()
    await cur.close()
    return row[0] if row else None


async def set_meta(key: str, value: str) -> None:
    """Upsert a single ``meta`` row."""
    await _conn().execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


async def integrity_check(path: Path | None = None) -> bool:
    """Run ``PRAGMA integrity_check`` against ``path`` (or the open DB).

    Returns False on any failure mode (file unreadable, not a SQLite DB,
    PRAGMA returns anything other than ``ok``). Used at startup to drive
    corruption recovery — never raises.
    """
    target = path or _DB_PATH
    if target is None:
        return False
    try:
        async with aiosqlite.connect(str(target)) as db:
            cur = await db.execute("PRAGMA integrity_check")
            rows = await cur.fetchall()
            await cur.close()
        return rows == [("ok",)]
    except (aiosqlite.Error, OSError) as exc:
        logger.warning("integrity_check failed for %s: %s", target, exc)
        return False


_NARROW_METRICS = (
    "request_throughput",
    "request_latency",
    "time_to_first_token",
    "output_token_throughput",
    "output_token_throughput_per_user",
    "inter_token_latency",
)


def _summarize_telemetry(telemetry: Any) -> tuple[int, str | None]:
    """Extract (gpu_count, representative_gpu_name) from a telemetry payload.

    Equivalent to the legacy ``_summarize_telemetry`` in results_db.py — moved
    to the write side so analytics never parse telemetry per request.
    """
    if not telemetry:
        return 0, None
    endpoints = telemetry.get("endpoints") or {}
    if not isinstance(endpoints, dict):
        return 0, None
    count = 0
    name: str | None = None
    for ep in endpoints.values():
        gpus = (ep or {}).get("gpus") or {}
        if not isinstance(gpus, dict):
            continue
        count += len(gpus)
        if name is None:
            for gpu in gpus.values():
                if gpu and gpu.get("gpu_name"):
                    name = gpu["gpu_name"]
                    break
    return count, name


def _extract_model_endpoint(spec: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull (model_name, endpoint_url) out of a CR spec, tolerant of shape variance."""
    benchmark = spec.get("benchmark", spec)
    endpoint_cfg = benchmark.get("endpoint", {}) or {}
    models_cfg = benchmark.get("models", {}) or {}
    if isinstance(models_cfg, list):
        items = models_cfg
    else:
        items = models_cfg.get("items", models_cfg.get("modelNames", [])) or []
    model: str | None = None
    if isinstance(items, list) and items:
        first = items[0]
        model = first.get("name", first) if isinstance(first, dict) else str(first)
    urls = endpoint_cfg.get("urls", endpoint_cfg.get("url", []))
    endpoint = (
        urls[0]
        if isinstance(urls, list) and urls
        else (urls if isinstance(urls, str) else None)
    )
    return model, endpoint


def _zstd_compress(payload: dict[str, Any]) -> bytes:
    return zstandard.ZstdCompressor().compress(orjson.dumps(payload))


def _narrow_metric_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten the six DEFAULT_COMPARE_METRICS into the 24 flat-column dict."""
    out: dict[str, Any] = {}
    for name in _NARROW_METRICS:
        m = metrics.get(name) or {}
        out[f"{name}_avg"] = m.get("avg")
        out[f"{name}_p50"] = m.get("p50")
        out[f"{name}_p99"] = m.get("p99")
        out[f"{name}_unit"] = m.get("unit")
    return out


async def upsert_run_created(
    namespace: str, job_id: str, epoch: str, *, spec: dict[str, Any]
) -> None:
    """Insert (or refresh-on-conflict) the row for a newly-observed AIPerfJob.

    Sets ``phase='Pending'`` and ``created_unix=now``. Pre-existing fields
    populated by a previous completion (e.g. on operator restart) are preserved
    via ``COALESCE`` so an out-of-order create event after completion does not
    erase metrics.
    """
    model, endpoint = _extract_model_endpoint(spec)
    spec_blob = _zstd_compress(spec)
    now = int(time.time())
    await _conn().execute(
        """
        INSERT INTO runs (
            namespace, job_id, epoch, phase, is_latest, created_unix,
            model, endpoint, spec_json
        )
        VALUES (?, ?, ?, 'Pending', 0, ?, ?, ?, ?)
        ON CONFLICT(namespace, job_id, epoch) DO UPDATE SET
            model      = COALESCE(runs.model, excluded.model),
            endpoint   = COALESCE(runs.endpoint, excluded.endpoint),
            spec_json  = COALESCE(runs.spec_json, excluded.spec_json)
        """,
        (namespace, job_id, epoch, now, model, endpoint, spec_blob),
    )


async def upsert_run_phase(
    namespace: str, job_id: str, epoch: str, *, phase: str
) -> None:
    """Update phase only — no metric or completion-time mutation.

    Inserts a stub row if the create event was missed (e.g. controller saw
    the job before the operator did).
    """
    now = int(time.time())
    await _conn().execute(
        """
        INSERT INTO runs (namespace, job_id, epoch, phase, created_unix)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(namespace, job_id, epoch) DO UPDATE SET phase = excluded.phase
        """,
        (namespace, job_id, epoch, phase, now),
    )


async def upsert_run_completed(
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    summary_blob: bytes,
    metrics: dict[str, Any],
    files: list[str],
    mtime_epoch: int,
    end_time: str | None = None,
    start_time: str | None = None,
    total_size_bytes: int = 0,
    phase: str = "Succeeded",
) -> None:
    """Record the post-run state: phase, metrics, blob, file inventory."""
    gpu_count, gpu_name = _summarize_telemetry(metrics.get("telemetry_data"))
    narrow = _narrow_metric_columns(metrics)

    cols = [
        "namespace",
        "job_id",
        "epoch",
        "phase",
        "created_unix",
        "start_time",
        "end_time",
        "mtime_epoch",
        "gpu_count",
        "gpu_name",
        "file_count",
        "total_size_bytes",
        "metrics_json",
    ]
    vals: list[Any] = [
        namespace,
        job_id,
        epoch,
        phase,
        int(time.time()),
        start_time,
        end_time,
        mtime_epoch,
        gpu_count,
        gpu_name,
        len(files),
        total_size_bytes,
        summary_blob,
    ]
    for k, v in narrow.items():
        cols.append(k)
        vals.append(v)

    placeholders = ", ".join("?" * len(cols))
    update_assignments = ", ".join(
        f"{c} = excluded.{c}"
        for c in cols
        if c not in ("namespace", "job_id", "epoch", "created_unix")
    )

    sql = (
        f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(namespace, job_id, epoch) DO UPDATE SET {update_assignments}"
    )
    await _conn().execute(sql, vals)


async def upsert_run_failed(
    namespace: str, job_id: str, epoch: str, *, error: str, phase: str = "Failed"
) -> None:
    """Record a failure — phase + error string, end_time stamped now."""
    now = int(time.time())
    await _conn().execute(
        """
        INSERT INTO runs (namespace, job_id, epoch, phase, error, created_unix, end_time)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(namespace, job_id, epoch) DO UPDATE SET
            phase    = excluded.phase,
            error    = excluded.error,
            end_time = excluded.end_time
        """,
        (namespace, job_id, epoch, phase, error, now),
    )


async def set_latest(namespace: str, job_id: str, epoch: str) -> None:
    """Atomically flip ``is_latest`` so exactly one row per (ns, job) is latest.

    Uses a single transaction: clear all is_latest rows for the job, then set
    the target. The ``runs_one_latest`` partial unique index turns any race
    into a hard error rather than silent dual-latest.
    """
    db = _conn()
    await db.execute("BEGIN IMMEDIATE")
    try:
        await db.execute(
            "UPDATE runs SET is_latest = 0 WHERE namespace = ? AND job_id = ? AND is_latest = 1",
            (namespace, job_id),
        )
        await db.execute(
            "UPDATE runs SET is_latest = 1 WHERE namespace = ? AND job_id = ? AND epoch = ?",
            (namespace, job_id, epoch),
        )
        await db.execute("COMMIT")
    except Exception:
        await db.execute("ROLLBACK")
        raise


async def delete_run(namespace: str, job_id: str, epoch: str) -> None:
    """Remove one run row. Used by retention and on_delete handlers."""
    await _conn().execute(
        "DELETE FROM runs WHERE namespace = ? AND job_id = ? AND epoch = ?",
        (namespace, job_id, epoch),
    )


_RUN_ROW_COLS = (
    "namespace, job_id, epoch, phase, is_latest, start_time, end_time, "
    "created_unix, mtime_epoch, error, model, endpoint, gpu_count, gpu_name, "
    "file_count, total_size_bytes, sweep_namespace, sweep_name, sweep_epoch, "
    "sweep_variation_idx"
)


def _row_to_run(row: tuple) -> RunIndexRow:
    return RunIndexRow(
        namespace=row[0],
        job_id=row[1],
        epoch=row[2],
        phase=row[3],
        is_latest=bool(row[4]),
        start_time=row[5],
        end_time=row[6],
        created_unix=row[7],
        mtime_epoch=row[8],
        error=row[9],
        model=row[10],
        endpoint=row[11],
        gpu_count=row[12],
        gpu_name=row[13],
        file_count=row[14],
        total_size_bytes=row[15],
        sweep_namespace=row[16],
        sweep_name=row[17],
        sweep_epoch=row[18],
        sweep_variation_idx=row[19],
    )


async def get_run(namespace: str, job_id: str, epoch: str) -> RunIndexRow | None:
    cur = await _conn().execute(
        f"SELECT {_RUN_ROW_COLS} FROM runs WHERE namespace = ? AND job_id = ? AND epoch = ?",
        (namespace, job_id, epoch),
    )
    row = await cur.fetchone()
    await cur.close()
    return _row_to_run(row) if row else None


async def list_runs_for_job(namespace: str, job_id: str) -> list[RunIndexRow]:
    cur = await _conn().execute(
        f"SELECT {_RUN_ROW_COLS} FROM runs WHERE namespace = ? AND job_id = ? "
        "ORDER BY mtime_epoch DESC NULLS LAST, epoch DESC",
        (namespace, job_id),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [_row_to_run(r) for r in rows]


async def get_summary_blob(namespace: str, job_id: str, epoch: str) -> bytes | None:
    cur = await _conn().execute(
        "SELECT metrics_json FROM runs WHERE namespace = ? AND job_id = ? AND epoch = ?",
        (namespace, job_id, epoch),
    )
    row = await cur.fetchone()
    await cur.close()
    return row[0] if row and row[0] else None


async def upsert_sweep_variation(
    namespace: str,
    sweep_name: str,
    sweep_epoch: str,
    variation_idx: int,
    *,
    variation_values: dict[str, Any],
    mode: str,
    phase: str | None,
    metrics: dict[str, Any],
    child_ref: tuple[str, str, str] | None,
    metrics_blob: bytes,
) -> None:
    """Insert (or update on conflict) one variation row.

    ``child_ref`` is ``(namespace, job_id, epoch)`` of the runs row produced by
    the variation's controller pod, or ``None`` for in-process / aggregate-only
    variations.
    """
    narrow = _narrow_metric_columns(metrics)
    child_ns, child_job, child_epoch = child_ref or (None, None, None)

    cols = [
        "namespace",
        "sweep_name",
        "sweep_epoch",
        "variation_idx",
        "variation_values_json",
        "mode",
        "phase",
        "child_namespace",
        "child_job_id",
        "child_epoch",
        "metrics_json",
    ]
    vals: list[Any] = [
        namespace,
        sweep_name,
        sweep_epoch,
        variation_idx,
        _zstd_compress(variation_values),
        mode,
        phase,
        child_ns,
        child_job,
        child_epoch,
        metrics_blob,
    ]
    for k, v in narrow.items():
        cols.append(k)
        vals.append(v)

    placeholders = ", ".join("?" * len(cols))
    update_assignments = ", ".join(
        f"{c} = excluded.{c}"
        for c in cols
        if c not in ("namespace", "sweep_name", "sweep_epoch", "variation_idx")
    )
    sql = (
        f"INSERT INTO sweep_variations ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(namespace, sweep_name, sweep_epoch, variation_idx) "
        f"DO UPDATE SET {update_assignments}"
    )
    await _conn().execute(sql, vals)


async def mark_sweep_pareto(
    namespace: str,
    sweep_name: str,
    sweep_epoch: str,
    *,
    rankings: list[tuple[int, int, bool]],
) -> None:
    """Apply ``[(variation_idx, pareto_rank, is_best), ...]`` in one transaction."""
    db = _conn()
    await db.execute("BEGIN IMMEDIATE")
    try:
        for idx, rank, best in rankings:
            await db.execute(
                "UPDATE sweep_variations SET pareto_rank = ?, is_best = ? "
                "WHERE namespace = ? AND sweep_name = ? AND sweep_epoch = ? "
                "AND variation_idx = ?",
                (rank, 1 if best else 0, namespace, sweep_name, sweep_epoch, idx),
            )
        await db.execute("COMMIT")
    except Exception:
        await db.execute("ROLLBACK")
        raise


async def list_sweep_variations(
    namespace: str, sweep_name: str, sweep_epoch: str
) -> list[SweepVariationRow]:
    cur = await _conn().execute(
        "SELECT namespace, sweep_name, sweep_epoch, variation_idx, mode, phase, "
        "       pareto_rank, is_best, child_namespace, child_job_id, child_epoch "
        "FROM sweep_variations "
        "WHERE namespace = ? AND sweep_name = ? AND sweep_epoch = ? "
        "ORDER BY variation_idx ASC",
        (namespace, sweep_name, sweep_epoch),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [
        SweepVariationRow(
            namespace=r[0],
            sweep_name=r[1],
            sweep_epoch=r[2],
            variation_idx=r[3],
            mode=r[4],
            phase=r[5],
            pareto_rank=r[6],
            is_best=bool(r[7]),
            child_namespace=r[8],
            child_job_id=r[9],
            child_epoch=r[10],
        )
        for r in rows
    ]


async def list_sweep_epochs_for_sweep(namespace: str, sweep_name: str) -> list[str]:
    cur = await _conn().execute(
        "SELECT DISTINCT sweep_epoch FROM sweep_variations "
        "WHERE namespace = ? AND sweep_name = ? ORDER BY sweep_epoch DESC",
        (namespace, sweep_name),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [r[0] for r in rows]
