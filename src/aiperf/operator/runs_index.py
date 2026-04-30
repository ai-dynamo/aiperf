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
import sqlite3
import time
from pathlib import Path
from typing import Any

import aiosqlite
import orjson
import zstandard

from aiperf.operator.results_layout import (
    EPOCH_RE,
    list_run_epochs,
    resolve_latest,
)
from aiperf.operator.runs_index_models import (
    BootstrapStats,
    RunIndexRow,
    SweepVariationRow,
)

READY_MARKER = ".aiperf_results_ready.json"

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


def is_open() -> bool:
    """Return True iff a runs_index DB is currently open."""
    return _DB is not None


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


async def list_all_latest() -> list[RunIndexRow]:
    """All ``is_latest=1`` rows, ordered by end_time DESC NULLS LAST."""
    cur = await _conn().execute(
        f"SELECT {_RUN_ROW_COLS} FROM runs WHERE is_latest = 1 "
        "ORDER BY end_time DESC NULLS LAST, created_unix DESC"
    )
    rows = await cur.fetchall()
    await cur.close()
    return [_row_to_run(r) for r in rows]


async def get_latest_run(namespace: str, job_id: str) -> RunIndexRow | None:
    """Return the ``is_latest=1`` row for a job, or None if no latest is set."""
    cur = await _conn().execute(
        f"SELECT {_RUN_ROW_COLS} FROM runs "
        "WHERE namespace = ? AND job_id = ? AND is_latest = 1 LIMIT 1",
        (namespace, job_id),
    )
    row = await cur.fetchone()
    await cur.close()
    return _row_to_run(row) if row else None


async def get_run_spec(
    namespace: str, job_id: str, epoch: str | None = None
) -> dict[str, Any] | None:
    """Decompress and return the CR spec stored in ``runs.spec_json``.

    When ``epoch`` is None, uses the is_latest row. Returns None when no row
    matches or spec_json is null.
    """
    if epoch is None:
        cur = await _conn().execute(
            "SELECT spec_json FROM runs "
            "WHERE namespace = ? AND job_id = ? AND is_latest = 1 LIMIT 1",
            (namespace, job_id),
        )
    else:
        cur = await _conn().execute(
            "SELECT spec_json FROM runs "
            "WHERE namespace = ? AND job_id = ? AND epoch = ?",
            (namespace, job_id, epoch),
        )
    row = await cur.fetchone()
    await cur.close()
    if row is None or row[0] is None:
        return None
    return orjson.loads(zstandard.ZstdDecompressor().decompress(row[0]))


async def list_sweep_epochs_for_sweep(namespace: str, sweep_name: str) -> list[str]:
    cur = await _conn().execute(
        "SELECT DISTINCT sweep_epoch FROM sweep_variations "
        "WHERE namespace = ? AND sweep_name = ? ORDER BY sweep_epoch DESC",
        (namespace, sweep_name),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [r[0] for r in rows]


async def bootstrap(base: Path, *, force: bool = False) -> BootstrapStats:
    """Walk the PVC and ingest every run + sweep variation found.

    - ``<base>/<ns>/<job>/<epoch>/`` for runs (excludes name == 'sweeps')
    - ``<base>/<ns>/sweeps/<name>/<epoch>/`` for sweep variations
    - Only indexes runs whose ``.aiperf_results_ready.json`` marker is present
    - is_latest is set per ``latest.txt``, not "newest mtime in the table"
    - When ``force=True``, drops + recreates the tables before walking
    """
    if force:
        db = _conn()
        await db.execute("DELETE FROM runs")
        await db.execute("DELETE FROM sweep_variations")

    started = time.monotonic()
    runs_count = 0
    sweep_count = 0

    if not base.is_dir():
        return BootstrapStats(0, 0, time.monotonic() - started)

    for ns_dir in base.iterdir():
        if not ns_dir.is_dir():
            continue

        # Runs walk: every <ns>/<job>/, EXCLUDING <ns>/sweeps/.
        for job_dir_path in ns_dir.iterdir():
            if not job_dir_path.is_dir() or job_dir_path.name == "sweeps":
                continue
            ns = ns_dir.name
            job = job_dir_path.name
            latest_epoch = resolve_latest(base, ns, job)
            for epoch in list_run_epochs(base, ns, job):
                run_path = job_dir_path / epoch
                marker = run_path / READY_MARKER
                if not marker.exists():
                    continue
                if await _index_run_from_disk(
                    base, ns, job, epoch, is_latest=(epoch == latest_epoch)
                ):
                    runs_count += 1

        # Sweeps walk
        sweeps_root = ns_dir / "sweeps"
        if sweeps_root.is_dir():
            for sweep_dir in sweeps_root.iterdir():
                if not sweep_dir.is_dir():
                    continue
                for epoch_dir in sweep_dir.iterdir():
                    if not epoch_dir.is_dir() or not EPOCH_RE.match(epoch_dir.name):
                        continue
                    if await _index_sweep_from_disk(
                        ns_dir.name, sweep_dir.name, epoch_dir.name, epoch_dir
                    ):
                        sweep_count += 1

    elapsed = time.monotonic() - started
    await set_meta("last_bootstrap_unix", str(int(time.time())))
    logger.info(
        "bootstrap: indexed %d runs, %d sweep variations in %.2fs",
        runs_count,
        sweep_count,
        elapsed,
    )
    return BootstrapStats(runs_count, sweep_count, elapsed)


async def _index_run_from_disk(
    base: Path, namespace: str, job_id: str, epoch: str, *, is_latest: bool
) -> bool:
    """Read profile_export_aiperf.json[.zst] and upsert a runs row.

    Returns True on success, False on read error / missing summary.
    Skips when the row already exists with metrics_json populated (post-restart
    no-op). Bootstrap can be re-run safely.
    """
    run_path = base / namespace / job_id / epoch
    summary_path_zst = run_path / "profile_export_aiperf.json.zst"
    summary_path_raw = run_path / "profile_export_aiperf.json"

    try:
        if summary_path_zst.exists():
            blob = summary_path_zst.read_bytes()
            metrics = orjson.loads(zstandard.ZstdDecompressor().decompress(blob))
            summary_blob = blob
        elif summary_path_raw.exists():
            raw = summary_path_raw.read_bytes()
            metrics = orjson.loads(raw)
            summary_blob = zstandard.ZstdCompressor().compress(raw)
        else:
            return False
    except (OSError, orjson.JSONDecodeError, zstandard.ZstdError) as exc:
        logger.warning("bootstrap: cannot read summary at %s: %s", run_path, exc)
        return False

    files = [f.name for f in run_path.iterdir() if f.is_file()]
    total_size = sum((run_path / f).stat().st_size for f in files)
    mtime_epoch = int(run_path.stat().st_mtime)

    spec = metrics.get("input_config", {}) or {}
    await upsert_run_created(namespace, job_id, epoch, spec={"benchmark": spec})
    await upsert_run_completed(
        namespace,
        job_id,
        epoch,
        summary_blob=summary_blob,
        metrics=metrics,
        files=files,
        mtime_epoch=mtime_epoch,
        start_time=metrics.get("start_time"),
        end_time=metrics.get("end_time"),
        total_size_bytes=total_size,
        phase="Succeeded",
    )
    if is_latest:
        await set_latest(namespace, job_id, epoch)
    return True


async def _index_sweep_from_disk(
    namespace: str, sweep_name: str, sweep_epoch: str, epoch_dir: Path
) -> bool:
    """Ingest <ns>/sweeps/<name>/<epoch>/ — variations + pareto if present.

    Looks for ``aggregate.json`` (the format ``aggregate_sweep_and_export``
    writes). Variations without it are skipped.
    """
    aggregate_path = epoch_dir / "aggregate.json"
    if not aggregate_path.exists():
        return False
    try:
        agg = orjson.loads(aggregate_path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return False

    indexed = False
    for v in agg.get("per_combination_metrics", []) or []:
        idx = v.get("variation_idx")
        if idx is None:
            continue
        await upsert_sweep_variation(
            namespace,
            sweep_name,
            sweep_epoch,
            int(idx),
            variation_values=v.get("variation_values", {}),
            mode=agg.get("metadata", {}).get("mode", "INDEPENDENT"),
            phase="Succeeded",
            metrics=v.get("metrics", {}),
            child_ref=None,
            metrics_blob=zstandard.ZstdCompressor().compress(orjson.dumps(v)),
        )
        indexed = True

    pareto_idxs = {p.get("variation_idx") for p in agg.get("pareto_optimal", []) or []}
    best_idxs = {
        b.get("variation_idx") for b in agg.get("best_configurations", []) or []
    }
    if pareto_idxs or best_idxs:
        rankings: list[tuple[int, int, bool]] = []
        for v in agg.get("per_combination_metrics", []) or []:
            idx = v.get("variation_idx")
            if idx is None:
                continue
            i = int(idx)
            rankings.append(
                (
                    i,
                    i if i in pareto_idxs else 999,
                    i in best_idxs,
                )
            )
        await mark_sweep_pareto(
            namespace,
            sweep_name,
            sweep_epoch,
            rankings=rankings,
        )

    return indexed


async def lazy_backfill_run(
    base: Path, namespace: str, job_id: str, epoch: str
) -> None:
    """Background task fired from read-path fallback. Best-effort, never raises."""
    try:
        latest_epoch = resolve_latest(base, namespace, job_id)
        await _index_run_from_disk(
            base,
            namespace,
            job_id,
            epoch,
            is_latest=(epoch == latest_epoch),
        )
    except Exception as exc:
        logger.warning(
            "lazy_backfill_run failed for %s/%s/%s: %s",
            namespace,
            job_id,
            epoch,
            exc,
        )


_VALID_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz_0123456789")


def _validate_identifier(name: str) -> None:
    if not name or not all(c in _VALID_IDENTIFIER_CHARS for c in name.lower()):
        raise ValueError(f"Invalid SQL identifier: {name!r}")


async def leaderboard(
    metric: str = "request_throughput",
    stat: str = "avg",
    order: str = "desc",
    limit: int = 20,
    *,
    epoch: str | None = None,
) -> list[dict[str, Any]]:
    _validate_identifier(metric)
    _validate_identifier(stat)
    order_dir = "DESC" if order.lower() == "desc" else "ASC"
    col = f"{metric}_{stat}"

    if epoch is None:
        sql = (
            f"SELECT namespace, job_id, epoch, {col} AS value, "
            f"       {metric}_unit AS unit, start_time, end_time, model, endpoint "
            f"FROM runs WHERE is_latest = 1 AND {col} IS NOT NULL "
            f"ORDER BY value {order_dir} LIMIT ?"
        )
        params: tuple[Any, ...] = (limit,)
    else:
        sql = (
            f"SELECT namespace, job_id, epoch, {col} AS value, "
            f"       {metric}_unit AS unit, start_time, end_time, model, endpoint "
            f"FROM runs WHERE epoch = ? AND {col} IS NOT NULL "
            f"ORDER BY value {order_dir} LIMIT ?"
        )
        params = (epoch, limit)

    return await _select_dicts(sql, params)


async def history(
    *,
    model: str | None = None,
    endpoint: str | None = None,
    metric: str = "request_throughput",
    stat: str = "avg",
    limit: int = 100,
    epoch: str | None = None,
) -> list[dict[str, Any]]:
    _validate_identifier(metric)
    _validate_identifier(stat)
    col = f"{metric}_{stat}"

    where = [f"{col} IS NOT NULL"]
    params: list[Any] = []
    if epoch is None:
        where.append("is_latest = 1")
    else:
        where.append("epoch = ?")
        params.append(epoch)
    if model:
        where.append("model LIKE ?")
        params.append(f"%{model}%")
    if endpoint:
        where.append("endpoint LIKE ?")
        params.append(f"%{endpoint}%")
    params.append(limit)

    sql = (
        f"SELECT namespace, job_id, epoch, {col} AS value, "
        f"       {metric}_unit AS unit, start_time, model, endpoint "
        f"FROM runs WHERE {' AND '.join(where)} "
        f"ORDER BY start_time ASC LIMIT ?"
    )
    return await _select_dicts(sql, tuple(params))


async def compare(
    job_ids: list[str],
    metrics: list[str] | None = None,
    *,
    epoch: str | None = None,
) -> list[dict[str, Any]]:
    if not job_ids:
        return []
    if metrics is None:
        metrics = list(_NARROW_METRICS)
    for m in metrics:
        _validate_identifier(m)

    cols = [
        "namespace",
        "job_id",
        "epoch",
        "start_time",
        "model",
        "endpoint",
        "gpu_count",
        "gpu_name",
    ]
    for m in metrics:
        for stat in ("avg", "p50", "p99"):
            cols.append(f"{m}_{stat}")
        cols.append(f"{m}_unit")

    placeholders = ", ".join("?" * len(job_ids))
    where = [f"job_id IN ({placeholders})"]
    params: list[Any] = list(job_ids)
    if epoch is None:
        where.append("is_latest = 1")
    else:
        where.append("epoch = ?")
        params.append(epoch)

    sql = f"SELECT {', '.join(cols)} FROM runs WHERE {' AND '.join(where)}"
    return await _select_dicts(sql, tuple(params))


async def _select_dicts(sql: str, params: tuple) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as dicts. Empty list on column-not-found.

    Analytics callers pass user-supplied metric names as column references
    (e.g. ``request_throughput_avg``) — when a metric does not exist as a
    column, SQLite raises ``OperationalError``. The legacy DuckDB read path
    swallowed the equivalent error and returned an empty list, and routers
    rely on that contract; preserve it here.
    """
    try:
        cur = await _conn().execute(sql, params)
    except sqlite3.OperationalError as exc:
        if "no such column" in str(exc):
            logger.debug("select returned no rows (no such column): %s", exc)
            return []
        raise
    cols = [d[0] for d in cur.description]
    rows = await cur.fetchall()
    await cur.close()
    return [dict(zip(cols, r, strict=True)) for r in rows]
