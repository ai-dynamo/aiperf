# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed index of runs and sweep variations.

Single-writer model: only the operator's kopf-owning process writes; readers
(operator FastAPI workers, results-server sidecar) open the DB read-only.
The single-writer assumption matches the operator's existing single-replica
deployment and the completion-claim mechanic in ``client_cache.py``. If the
operator is ever scaled up, only the kopf-owning process must call write APIs.

The DB lives at ``<RESULTS.DIR>/.aiperf_index.sqlite`` in WAL mode. WAL mode
gives us non-blocking readers across processes. All write APIs share the single
module connection, so concurrent in-process writers (kopf runs per-object
handlers concurrently) are serialized through ``_write_lock``: ``BEGIN
IMMEDIATE`` cannot be entered twice on one connection, and an autocommit write
must never be absorbed into another coroutine's open transaction. ``busy_timeout
=5000`` only covers cross-process contention on the WAL.

The index is a cache, never a source of truth. Every read site falls back to
a filesystem scan on miss and lazy-backfills the row in the background, so a
corrupt or stale index degrades to slower, never wrong.

Index-only reads additionally require a completed-bootstrap marker and no
pending result publication markers. A nonempty query result cannot itself
prove that every ready run on the PVC is represented in SQLite.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiosqlite
import orjson
import zstandard

from aiperf.common.redact import redact_endpoint_spec, redact_url
from aiperf.common.results_markers import EPOCH_RE
from aiperf.operator.artifact_names import find_summary_path
from aiperf.operator.results_layout import (
    is_run_ready,
    list_run_epochs,
    resolve_latest,
)
from aiperf.operator.runs_index_models import (
    BootstrapStats,
    RunIndexRow,
    SweepVariationRow,
)

READY_MARKER = ".aiperf_results_ready.json"
_CATALOG_COMPLETE_MARKER = ".aiperf_index_complete"
_CATALOG_PENDING_DIR = ".aiperf_index_pending"

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_DB: aiosqlite.Connection | None = None
_DB_PATH: Path | None = None
_READ_ONLY = False

# Serializes every write API on the single shared connection. The lock is
# bound to the running loop the first time a writer acquires it, so unit tests
# that spin a fresh event loop per test never inherit a lock from a dead loop.
_write_lock: asyncio.Lock | None = None
_write_lock_loop: asyncio.AbstractEventLoop | None = None


def _writer_lock() -> asyncio.Lock:
    """Return the write lock, (re)binding it to the current running loop."""
    global _write_lock, _write_lock_loop
    loop = asyncio.get_running_loop()
    if _write_lock is None or _write_lock_loop is not loop:
        _write_lock = asyncio.Lock()
        _write_lock_loop = loop
    return _write_lock


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
    global _DB, _DB_PATH, _READ_ONLY

    if _DB is not None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path), isolation_level=None)
    # Any failure after connect but before _DB is assigned must close the
    # connection: aiosqlite spawns a non-daemon worker thread per connection,
    # and callers (e.g. ResultsDB) retry open() per request, so a leak here
    # accumulates threads + fds unboundedly.
    try:
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
    except BaseException:
        await db.close()
        raise

    _DB = db
    _DB_PATH = path
    _READ_ONLY = False
    logger.info("runs_index opened at %s (schema_version=%d)", path, SCHEMA_VERSION)


async def open_readonly(path: Path) -> None:
    """Open an existing runs_index DB for read-only serving.

    Results-server sidecars use this path because the operator process is the
    single writer. The DB must already exist; missing or migrated schemas are
    operator-startup responsibilities, not sidecar side effects.
    """
    global _DB, _DB_PATH, _READ_ONLY

    if _DB is not None:
        return

    uri = f"file:{quote(str(path), safe='/')}?mode=ro&cache=shared"
    db = await aiosqlite.connect(uri, uri=True, isolation_level=None)
    # Same leak guard as open(): a failure between connect and _DB assignment
    # (e.g. missing meta table on a foreign sqlite file) would otherwise leak
    # the connection's non-daemon thread on every retry.
    try:
        await db.execute("PRAGMA query_only = ON")
        await db.execute("PRAGMA busy_timeout = 5000")

        cur = await db.execute("SELECT value FROM meta WHERE key = 'schema_version'")
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            raise RuntimeError(f"runs_index DB at {path} is missing schema_version")
        existing = int(row[0])
        if existing > SCHEMA_VERSION:
            raise RuntimeError(
                f"runs_index DB at {path} has schema_version={existing} but this "
                f"build only knows up to {SCHEMA_VERSION}. Refusing to open."
            )
    except BaseException:
        await db.close()
        raise

    _DB = db
    _DB_PATH = path
    _READ_ONLY = True
    logger.info("runs_index opened read-only at %s (schema_version=%d)", path, existing)


async def close() -> None:
    """Close the DB. Safe to call when never opened."""
    global _DB, _DB_PATH, _READ_ONLY, _write_lock, _write_lock_loop
    if _DB is not None:
        await _DB.close()
    _DB = None
    _DB_PATH = None
    _READ_ONLY = False
    _write_lock = None
    _write_lock_loop = None


def is_open() -> bool:
    """Return True iff a runs_index DB is currently open."""
    return _DB is not None


def is_readonly() -> bool:
    """Return True iff the open runs_index connection is read-only."""
    return _DB is not None and _READ_ONLY


def _catalog_update_marker(base: Path, namespace: str, job_id: str, epoch: str) -> Path:
    identity = f"{namespace}\0{job_id}\0{epoch}".encode()
    digest = hashlib.sha256(identity).hexdigest()
    return base / _CATALOG_PENDING_DIR / digest


def begin_catalog_update(
    base: Path, namespace: str, job_id: str, epoch: str
) -> Path | None:
    """Gate index-only reads while a run becomes visible on disk."""
    marker = _catalog_update_marker(base, namespace, job_id, epoch)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
    except OSError as exc:
        mark_catalog_incomplete(base)
        logger.warning(
            "cannot create runs_index publication marker for %s/%s/%s: %s",
            namespace,
            job_id,
            epoch,
            exc,
        )
        return None
    return marker


def finish_catalog_update(marker: Path | None) -> None:
    """Clear one publication gate after its matching index write succeeds."""
    if marker is None:
        return
    try:
        marker.unlink(missing_ok=True)
    except OSError:
        logger.warning("cannot clear runs_index publication marker at %s", marker)


def mark_catalog_incomplete(base: Path) -> None:
    """Disable index-only reads before a full catalog rebuild."""
    try:
        (base / _CATALOG_COMPLETE_MARKER).unlink(missing_ok=True)
    except OSError:
        logger.warning("cannot clear runs_index completeness marker at %s", base)


def mark_catalog_complete(base: Path) -> bool:
    """Record successful bootstrap proof; pending publications still gate reads."""
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / _CATALOG_COMPLETE_MARKER).touch(exist_ok=True)
    except OSError:
        return False
    return True


def catalog_is_complete(base: Path) -> bool:
    """Return whether the index has proven coverage of visible disk runs."""
    if not (base / _CATALOG_COMPLETE_MARKER).is_file():
        return False
    pending_dir = base / _CATALOG_PENDING_DIR
    try:
        return not pending_dir.is_dir() or next(pending_dir.iterdir(), None) is None
    except OSError:
        return False


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
    async with _writer_lock():
        await _conn().execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


async def stats(db_path: Path) -> dict[str, Any]:
    """Return summary counts + on-disk size of the runs index.

    Backs ``GET /admin/index/stats``. Reads
    from the open connection (not from ``db_path``); ``db_path`` is used only
    for ``stat().st_size``.
    """
    cur = await _conn().execute("SELECT COUNT(*) FROM runs")
    runs_count = (await cur.fetchone())[0]
    await cur.close()
    cur = await _conn().execute("SELECT COUNT(*) FROM sweep_variations")
    sweep_count = (await cur.fetchone())[0]
    await cur.close()
    last = await get_meta("last_bootstrap_unix")
    return {
        "runs_count": runs_count,
        "sweep_variations_count": sweep_count,
        "db_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "last_bootstrap_unix": int(last) if last else None,
        "schema_version": SCHEMA_VERSION,
    }


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
    return model, redact_url(endpoint) if endpoint is not None else None


def _zstd_compress(payload: dict[str, Any]) -> bytes:
    return zstandard.ZstdCompressor().compress(orjson.dumps(payload))


def zstd_decompress(blob: bytes) -> bytes:
    """Decompress a zstd blob whether or not the frame carries a content size.

    The completion handler writes ``profile_export_aiperf.json.zst`` via a
    streaming compressor that does NOT embed the content size in the frame
    header; ``ZstdDecompressor().decompress(blob)`` then raises ``ZstdError:
    could not determine content size in frame header``. ``stream_reader``
    handles both framed and stream-mode blobs, so use it uniformly.
    """
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob)) as reader:
        return reader.read()


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


def _metric_stat_value(payload: dict[str, Any]) -> Any:
    return payload.get("avg", payload.get("mean"))


def _normalize_sweep_metrics(metrics: Any) -> dict[str, Any]:
    """Normalize sweep per-combination stats for narrow-column extraction."""
    if not isinstance(metrics, dict):
        return {}

    normalized: dict[str, Any] = {}
    for name in _NARROW_METRICS:
        direct = metrics.get(name)
        if isinstance(direct, dict):
            entry = dict(direct)
            if "avg" not in entry and "mean" in entry:
                entry["avg"] = entry["mean"]
            normalized[name] = entry

        for stat in ("avg", "p50", "p99"):
            stat_payload = metrics.get(f"{name}_{stat}")
            if not isinstance(stat_payload, dict):
                continue
            entry = normalized.setdefault(name, {})
            entry[stat] = _metric_stat_value(stat_payload)
            if "unit" not in entry and stat_payload.get("unit") is not None:
                entry["unit"] = stat_payload["unit"]
    return normalized


async def upsert_run_created(
    namespace: str, job_id: str, epoch: str, *, spec: dict[str, Any]
) -> None:
    """Insert (or refresh-on-conflict) the row for a newly-observed AIPerfJob.

    Sets ``phase='Pending'`` and ``created_unix=now``. Pre-existing fields
    populated by a previous completion (e.g. on operator restart) are preserved
    via ``COALESCE`` so an out-of-order create event after completion does not
    erase metrics.
    """
    safe_spec = redact_endpoint_spec(spec)
    model, endpoint = _extract_model_endpoint(safe_spec)
    spec_blob = _zstd_compress(safe_spec)
    now = int(time.time())
    async with _writer_lock():
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
    async with _writer_lock():
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
    error: str | None = None,
) -> None:
    """Record the post-run state: phase, metrics, blob, file inventory.

    ``metrics`` here is the full ``/api/metrics`` envelope returned by the
    controller (top-level metadata plus a nested ``metrics`` mapping). The
    narrow compare columns come from the nested payload, not the envelope
    itself — flattening the envelope produces all-NULL narrow columns even
    though the summary blob is present and valid.
    """
    gpu_count, gpu_name = _summarize_telemetry(metrics.get("telemetry_data"))
    metrics_payload = (
        metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else metrics
    )
    narrow = _narrow_metric_columns(metrics_payload)

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
        "error",
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
        error,
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
    async with _writer_lock():
        await _conn().execute(sql, vals)


async def upsert_run_failed(
    namespace: str, job_id: str, epoch: str, *, error: str, phase: str = "Failed"
) -> None:
    """Record a failure — phase + error string, end_time stamped now.

    ``end_time`` is written as an offset-carrying ISO-8601 string, matching
    what completed rows store. SQLite's ``datetime('now')`` produces an
    offset-less, space-separated form that ``_iso_to_unix`` reads as *local*
    time, and that ``ORDER BY end_time DESC`` sorts against the completed
    rows' ``T`` separator by ASCII (' ' < 'T'), interleaving failed and
    succeeded runs wrongly regardless of when they happened.
    """
    now = int(time.time())
    ended_at = datetime.now(UTC).isoformat()
    async with _writer_lock():
        await _conn().execute(
            """
            INSERT INTO runs (namespace, job_id, epoch, phase, error, created_unix, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, job_id, epoch) DO UPDATE SET
                phase    = excluded.phase,
                error    = excluded.error,
                end_time = excluded.end_time
            """,
            (namespace, job_id, epoch, phase, error, now, ended_at),
        )


async def set_latest(namespace: str, job_id: str, epoch: str) -> None:
    """Atomically flip ``is_latest`` so exactly one row per (ns, job) is latest.

    Uses a single transaction: clear all is_latest rows for the job, then set
    the target. The ``runs_one_latest`` partial unique index turns any race
    into a hard error rather than silent dual-latest.
    """
    db = _conn()
    async with _writer_lock():
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


async def clear_latest(namespace: str, job_id: str) -> None:
    """Clear the latest marker when a job has no surviving result epoch."""
    async with _writer_lock():
        await _conn().execute(
            "UPDATE runs SET is_latest = 0 "
            "WHERE namespace = ? AND job_id = ? AND is_latest = 1",
            (namespace, job_id),
        )


async def delete_run(namespace: str, job_id: str, epoch: str) -> None:
    """Remove one run row. Used by retention and on_delete handlers."""
    async with _writer_lock():
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
    """Fetch the indexed row for one ``(namespace, job_id, epoch)`` run, if present."""
    cur = await _conn().execute(
        f"SELECT {_RUN_ROW_COLS} FROM runs WHERE namespace = ? AND job_id = ? AND epoch = ?",
        (namespace, job_id, epoch),
    )
    row = await cur.fetchone()
    await cur.close()
    return _row_to_run(row) if row else None


async def get_run_narrow_metrics(
    namespace: str, job_id: str, epoch: str | None = None
) -> dict[str, Any] | None:
    """Return the narrow metric columns for a single run row.

    Used by the K8s-vs-local audit suite's ``index_consistency`` check to
    confirm that the flat-column projection in ``runs`` matches the on-disk
    ``profile_export_aiperf.json``. If ``epoch`` is None the latest run for
    the job is selected. Returns None when no row exists.

    The keys mirror the narrow column names: ``epoch`` plus 18 metric/stat
    pairs of the form ``<metric>_<stat>`` for stats avg/p50/p99 across the
    six ``DEFAULT_COMPARE_METRICS``.
    """
    metric_cols = ", ".join(
        f"{name}_{stat}" for name in _NARROW_METRICS for stat in ("avg", "p50", "p99")
    )
    if epoch is None:
        cur = await _conn().execute(
            f"SELECT epoch, phase, {metric_cols} FROM runs "
            "WHERE namespace = ? AND job_id = ? AND is_latest = 1",
            (namespace, job_id),
        )
    else:
        cur = await _conn().execute(
            f"SELECT epoch, phase, {metric_cols} FROM runs "
            "WHERE namespace = ? AND job_id = ? AND epoch = ?",
            (namespace, job_id, epoch),
        )
    row = await cur.fetchone()
    await cur.close()
    if row is None:
        return None
    out: dict[str, Any] = {"epoch": row[0], "phase": row[1]}
    idx = 2
    for name in _NARROW_METRICS:
        for stat in ("avg", "p50", "p99"):
            out[f"{name}_{stat}"] = row[idx]
            idx += 1
    return out


async def scatter_data() -> list[dict[str, Any]]:
    """Return flat scatter metrics for all runs with at least one metric populated.

    Used by ``GET /api/v1/analytics/scatter`` to power the dashboard scatter chart
    in a single query instead of N+1 leaderboard+summary round-trips.
    """
    cur = await _conn().execute(
        """
        SELECT namespace, job_id, epoch, model,
               request_throughput_avg,
               request_latency_p99,
               time_to_first_token_avg,
               output_token_throughput_avg
        FROM runs
        WHERE is_latest = 1
          AND (
            request_throughput_avg IS NOT NULL
            OR request_latency_p99 IS NOT NULL
            OR time_to_first_token_avg IS NOT NULL
            OR output_token_throughput_avg IS NOT NULL
          )
        ORDER BY epoch DESC
        LIMIT 500
        """
    )
    rows = await cur.fetchall()
    await cur.close()
    return [
        {
            "namespace": r[0],
            "job_id": r[1],
            "epoch": r[2],
            "model": r[3],
            "request_throughput_avg": r[4],
            "request_latency_p99": r[5],
            "time_to_first_token_avg": r[6],
            "output_token_throughput_avg": r[7],
        }
        for r in rows
    ]


async def list_runs_for_job(namespace: str, job_id: str) -> list[RunIndexRow]:
    """List indexed runs for a job, newest epoch first."""
    cur = await _conn().execute(
        f"SELECT {_RUN_ROW_COLS} FROM runs WHERE namespace = ? AND job_id = ? "
        "ORDER BY mtime_epoch DESC NULLS LAST, epoch DESC",
        (namespace, job_id),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [_row_to_run(r) for r in rows]


async def get_summary_blob(namespace: str, job_id: str, epoch: str) -> bytes | None:
    """Return the zstd-compressed ``metrics_json`` blob for a run, if indexed."""
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
    async with _writer_lock():
        await _conn().execute(sql, vals)


async def mark_sweep_pareto(
    namespace: str,
    sweep_name: str,
    sweep_epoch: str,
    *,
    rankings: list[tuple[int, int | None, bool]],
) -> None:
    """Apply ``[(variation_idx, pareto_rank, is_best), ...]`` in one transaction.

    ``pareto_rank`` is 0 for a front member and None for everything else.
    """
    db = _conn()
    async with _writer_lock():
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
    """List indexed variation rows for one sweep epoch."""
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
    return orjson.loads(zstd_decompress(row[0]))


async def list_sweep_epochs_for_sweep(namespace: str, sweep_name: str) -> list[str]:
    """List distinct sweep epochs recorded for a sweep, newest first."""
    cur = await _conn().execute(
        "SELECT DISTINCT sweep_epoch FROM sweep_variations "
        "WHERE namespace = ? AND sweep_name = ? ORDER BY sweep_epoch DESC",
        (namespace, sweep_name),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [r[0] for r in rows]


async def delete_sweep_epoch(
    namespace: str,
    sweep_name: str,
    sweep_epoch: str,
) -> None:
    """Delete every cached variation row for one removed sweep archive."""
    async with _writer_lock():
        await _conn().execute(
            "DELETE FROM sweep_variations "
            "WHERE namespace = ? AND sweep_name = ? AND sweep_epoch = ?",
            (namespace, sweep_name, sweep_epoch),
        )


async def bootstrap(base: Path, *, force: bool = False) -> BootstrapStats:
    """Walk the PVC and converge the index on the disk truth, both directions.

    - ``<base>/<ns>/<job>/<epoch>/`` for runs (excludes name == 'sweeps')
    - ``<base>/<ns>/sweeps/<name>/<epoch>/`` for sweep variations
    - is_latest is set per ``latest.txt``, not "newest mtime in the table"
    - When ``force=True``, drops + recreates the tables before walking
    - Before the forward ingest walk, reverse pruning drops indexed run and
      sweep epochs whose durable directories vanished from disk (a retention
      ``rmtree`` whose scheduled index drop was lost to an operator crash)

    Bootstrap only ingests runs with the ``.aiperf_results_ready.json`` marker.
    An operator restart can overlap result export, so a summary's presence
    alone does not prove it is a complete artifact set.
    """
    if base.is_dir():
        mark_catalog_incomplete(base)

    if force:
        async with _writer_lock():
            db = _conn()
            await db.execute("DELETE FROM runs")
            await db.execute("DELETE FROM sweep_variations")

    started = time.monotonic()
    runs_count = 0
    sweep_count = 0
    catalog_complete = True

    if not base.is_dir():
        # A missing base (PVC unmounted mid-startup) must NOT trigger the
        # prune pass — treating "nothing mounted" as "everything deleted"
        # would wipe a perfectly valid index.
        return BootstrapStats(0, 0, time.monotonic() - started)

    await _prune_stale_run_rows(base)
    await _prune_stale_sweep_rows(base)

    for ns_dir in base.iterdir():
        if not ns_dir.is_dir():
            continue
        namespace_runs, namespace_complete = await _bootstrap_namespace_runs(
            base, ns_dir
        )
        runs_count += namespace_runs
        catalog_complete = catalog_complete and namespace_complete
        sweep_count += await _bootstrap_namespace_sweeps(ns_dir)

    elapsed = time.monotonic() - started
    await set_meta("last_bootstrap_unix", str(int(time.time())))
    if catalog_complete:
        mark_catalog_complete(base)
    logger.info(
        "bootstrap: indexed %d runs, %d sweep variations in %.2fs",
        runs_count,
        sweep_count,
        elapsed,
    )
    return BootstrapStats(runs_count, sweep_count, elapsed)


async def _prune_stale_run_rows(base: Path) -> int:
    """Drop ``runs`` rows whose on-disk run dir no longer exists.

    Reverse pass of the ingest walk. Retention (``enforce_retention``) deletes
    run dirs on disk first and schedules the matching index drops second
    (``schedule_index_drops``), so a crash between the two strands rows the
    forward walk can never repair — ingest only ever adds. Pruning is
    restricted to rows that recorded on-disk artifacts (``mtime_epoch`` or
    ``metrics_json`` populated by completion/bootstrap): a ``Pending`` stub
    for an in-flight job and a ``Failed`` row for a job that died before
    writing results legitimately have no run dir and must survive.

    Runs inside bootstrap on the operator's single-writer connection, so no
    other writer can race the deletes; the per-row directory checks are
    offloaded to a worker thread. Returns the number of rows pruned.
    """
    cur = await _conn().execute(
        "SELECT namespace, job_id, epoch FROM runs "
        "WHERE mtime_epoch IS NOT NULL OR metrics_json IS NOT NULL"
    )
    rows = [(r[0], r[1], r[2]) for r in await cur.fetchall()]
    await cur.close()
    if not rows:
        return 0

    def _missing_on_disk() -> list[tuple[str, str, str]]:
        return [
            (ns, job, epoch)
            for ns, job, epoch in rows
            if not (base / ns / job / epoch).is_dir()
        ]

    stale = await asyncio.to_thread(_missing_on_disk)
    for ns, job, epoch in stale:
        await delete_run(ns, job, epoch)
    if stale:
        logger.info(
            "bootstrap: pruned %d stale index row(s) with no run dir on disk",
            len(stale),
        )
    return len(stale)


async def _prune_stale_sweep_rows(base: Path) -> int:
    """Drop sweep variation rows whose durable epoch dir no longer exists."""
    cur = await _conn().execute(
        "SELECT DISTINCT namespace, sweep_name, sweep_epoch FROM sweep_variations"
    )
    rows = [(r[0], r[1], r[2]) for r in await cur.fetchall()]
    await cur.close()
    if not rows:
        return 0

    def _missing_on_disk() -> list[tuple[str, str, str]]:
        return [
            (namespace, sweep_name, epoch)
            for namespace, sweep_name, epoch in rows
            if not (base / namespace / "sweeps" / sweep_name / epoch).is_dir()
        ]

    stale = await asyncio.to_thread(_missing_on_disk)
    for namespace, sweep_name, epoch in stale:
        await delete_sweep_epoch(namespace, sweep_name, epoch)
    if stale:
        logger.info(
            "bootstrap: pruned %d stale sweep epoch(s) with no directory on disk",
            len(stale),
        )
    return len(stale)


async def _bootstrap_namespace_runs(base: Path, ns_dir: Path) -> tuple[int, bool]:
    """Ingest every run epoch under ``<ns>/<job>/``, excluding ``<ns>/sweeps/``.

    Returns the number of runs indexed and whether every ready run was indexed.
    """
    runs_count = 0
    catalog_complete = True
    for job_dir_path in ns_dir.iterdir():
        if not job_dir_path.is_dir() or job_dir_path.name == "sweeps":
            continue
        ns = ns_dir.name
        job = job_dir_path.name
        latest_epoch = resolve_latest(base, ns, job)
        for epoch in list_run_epochs(base, ns, job):
            ready = is_run_ready(base / ns / job / epoch)
            # Per-iteration guard so one corrupt run dir (malformed
            # input_config, sqlite hiccup, partial filesystem permission
            # error) cannot abort the WHOLE bootstrap. Same shape as
            # the per-file try/except in completion._parse_metrics_from_files.
            try:
                indexed = await _index_run_from_disk(
                    base, ns, job, epoch, is_latest=(epoch == latest_epoch)
                )
            except Exception as exc:
                logger.warning(
                    f"runs index bootstrap: skipping {ns}/{job} epoch={epoch} "
                    f"({type(exc).__name__}: {exc})"
                )
                if ready:
                    catalog_complete = False
                continue
            if indexed:
                runs_count += 1
                finish_catalog_update(_catalog_update_marker(base, ns, job, epoch))
            elif ready:
                catalog_complete = False
    return runs_count, catalog_complete


async def _bootstrap_namespace_sweeps(ns_dir: Path) -> int:
    """Ingest every sweep-variation epoch under ``<ns>/sweeps/``.

    Returns the number of sweep variations indexed for this namespace dir;
    corrupt epoch dirs are logged and skipped (same per-iteration guard as
    the runs walk) so one bad sweep cannot abort the whole bootstrap.
    """
    sweeps_root = ns_dir / "sweeps"
    if not sweeps_root.is_dir():
        return 0
    sweep_count = 0
    for sweep_dir in sweeps_root.iterdir():
        if not sweep_dir.is_dir():
            continue
        for epoch_dir in sweep_dir.iterdir():
            if not epoch_dir.is_dir() or not EPOCH_RE.match(epoch_dir.name):
                continue
            try:
                indexed = await _index_sweep_from_disk(
                    ns_dir.name, sweep_dir.name, epoch_dir.name, epoch_dir
                )
            except Exception as exc:
                logger.warning(
                    f"runs index bootstrap: skipping sweep "
                    f"{ns_dir.name}/{sweep_dir.name} epoch={epoch_dir.name} "
                    f"({type(exc).__name__}: {exc})"
                )
                continue
            sweep_count += indexed
    return sweep_count


_TERMINAL_RUN_PHASES: frozenset[str] = frozenset({"Failed", "Cancelled"})
"""Phases a disk backfill must never overwrite with "Succeeded"."""


async def _index_run_from_disk(
    base: Path, namespace: str, job_id: str, epoch: str, *, is_latest: bool
) -> bool:
    """Read the run-specific summary JSON and upsert a runs row.

    Returns True on success, False on read error / missing summary.
    Skips when the row already exists with metrics_json populated (post-restart
    no-op). Bootstrap can be re-run safely.
    """
    run_path = base / namespace / job_id / epoch
    if not is_run_ready(run_path):
        return False
    try:
        summary_path = find_summary_path(run_path)
        if summary_path is None:
            return False
        if summary_path.suffix == ".zst":
            blob = summary_path.read_bytes()
            metrics = orjson.loads(zstd_decompress(blob))
            summary_blob = blob
        else:
            raw = summary_path.read_bytes()
            metrics = orjson.loads(raw)
            summary_blob = zstandard.ZstdCompressor().compress(raw)
    except (OSError, orjson.JSONDecodeError, zstandard.ZstdError) as exc:
        logger.warning("bootstrap: cannot read summary at %s: %s", run_path, exc)
        return False

    files = [
        f.name for f in run_path.iterdir() if f.is_file() and f.name != READY_MARKER
    ]
    total_size = sum((run_path / f).stat().st_size for f in files)
    mtime_epoch = int(run_path.stat().st_mtime)

    spec = metrics.get("input_config", {}) or {}
    # An exported summary on disk proves artifacts exist, NOT that the run
    # succeeded. Stamping "Succeeded" unconditionally let a lazy backfill flip
    # a genuinely Failed row back to Succeeded while `error` still held the
    # failure text. A terminal phase already recorded by the operator wins.
    existing = await get_run(namespace, job_id, epoch)
    phase = (
        existing.phase
        if existing is not None and existing.phase in _TERMINAL_RUN_PHASES
        else "Succeeded"
    )
    error = existing.error if phase in _TERMINAL_RUN_PHASES and existing else None
    try:
        marker = orjson.loads((run_path / READY_MARKER).read_bytes())
    except (OSError, orjson.JSONDecodeError):
        marker = {}
    if isinstance(marker, dict):
        marker_phase = marker.get("terminal_phase")
        marker_error = marker.get("terminal_error")
        if marker_phase in _TERMINAL_RUN_PHASES:
            phase = marker_phase
            error = marker_error if isinstance(marker_error, str) else None
        elif marker.get("was_cancelled") is True:
            phase = "Cancelled"
            error = None
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
        phase=phase,
        error=error,
    )
    if is_latest:
        await set_latest(namespace, job_id, epoch)
    return True


SweepRow = tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]


def _sweep_mode(agg: dict[str, Any]) -> str:
    metadata = agg.get("metadata") if isinstance(agg.get("metadata"), dict) else {}
    mode = metadata.get("mode") or metadata.get("sweep_mode") or "INDEPENDENT"
    return str(mode).rsplit(".", 1)[-1]


def _legacy_sweep_rows(agg: dict[str, Any]) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for row in agg.get("per_combination_metrics", []) or []:
        if not isinstance(row, dict):
            continue
        idx = row.get("variation_idx")
        if idx is None:
            continue
        rows.append(
            (
                idx,
                row.get("variation_values", {}),
                _normalize_sweep_metrics(row.get("metrics", {}) or {}),
                row,
            )
        )
    return rows


def _load_strategy_sweep_aggregate(epoch_dir: Path) -> dict[str, Any] | None:
    aggregate_dir = epoch_dir / "sweep_aggregate"
    loaded: list[dict[str, Any]] = []
    for filename in (
        "profile_export_aiperf_sweep.json",
        "profile_export_aiperf_aggregate.json",
    ):
        try:
            doc = orjson.loads((aggregate_dir / filename).read_bytes())
        except (FileNotFoundError, OSError, orjson.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        loaded.append(doc)
        rows = doc.get("per_combination_metrics")
        if isinstance(rows, list) and rows:
            return doc
    return loaded[0] if loaded else None


def _strategy_sweep_rows(agg: dict[str, Any]) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for fallback_idx, row in enumerate(agg.get("per_combination_metrics", []) or []):
        if not isinstance(row, dict):
            continue
        idx = row.get("variation_idx", fallback_idx)
        rows.append(
            (
                idx,
                row.get("variation_values") or row.get("parameters", {}) or {},
                _normalize_sweep_metrics(row.get("metrics", {}) or {}),
                row,
            )
        )
    return rows


def _results_base_from_sweep_epoch_dir(epoch_dir: Path) -> Path | None:
    try:
        return epoch_dir.parents[3]
    except IndexError:
        return None


def _load_profile_export(run_path: Path) -> dict[str, Any] | None:
    try:
        summary_path = find_summary_path(run_path)
        if summary_path is None:
            return None
        payload = summary_path.read_bytes()
        if summary_path.suffix == ".zst":
            payload = zstd_decompress(payload)
        return orjson.loads(payload)
    except (OSError, orjson.JSONDecodeError, zstandard.ZstdError):
        return None
    return None


def _child_variation_values(child: dict[str, Any]) -> dict[str, Any]:
    """Build the sweep_variations row's variation-values payload for one child.

    The SWEPT PARAMETERS are the point of this column -- it is what makes two
    rows comparable, and ``_stable_variation_values`` strips ``trial_index``
    from it precisely so trials of one parameter point collapse together. Until
    the children manifest carried ``variation_values`` there was nothing else to
    put here, so the column held only ``variation_label``. For an adaptive sweep
    that label is ``search_iter_NNNN`` (``optuna_planner.py:226``) -- an
    artifact-path cell id that says nothing about what was tried, and which
    differs for every trial of the SAME parameter point. Grouping on it
    therefore treats identical configurations as distinct.

    The manifest value is a JSON string bounded by
    ``_bounded_variation_values_json`` (``k8s_executor.py:188``), which
    substitutes a ``__aiperf_truncated__`` marker when the real payload exceeds
    the annotation budget. That marker is dropped rather than stored: it is a
    statement that the values are UNKNOWN, and indexing it would mint a bogus
    group shared by every oversized variation. Same rule the read side applies
    at ``kubernetes/results.py:704``.

    ``variation_label`` is still recorded so a row remains identifiable when no
    values were captured.
    """
    values: dict[str, Any] = {}
    raw = child.get("variation_values")
    parsed: Any = raw
    if isinstance(raw, str) and raw:
        try:
            parsed = orjson.loads(raw)
        except orjson.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict) and not parsed.get("__aiperf_truncated__"):
        values.update(parsed)
    label = child.get("variation_label")
    if label:
        values["variation_label"] = label
    trial_index = child.get("trial_index")
    if trial_index is not None:
        values["trial_index"] = trial_index
    return values


def _parse_child_sweep_manifest(
    epoch_dir: Path, children: list[Any]
) -> tuple[list[tuple[int, str, str, str, dict[str, Any]]], dict[int, int]]:
    parsed_children: list[tuple[int, str, str, str, dict[str, Any]]] = []
    manifest_counts: dict[int, int] = {}
    for child in children:
        if not isinstance(child, dict):
            continue
        try:
            idx = int(child["variation_index"])
            namespace = str(child.get("namespace") or epoch_dir.parents[2].name)
            name = str(child["name"])
            child_epoch = str(child.get("child_run_epoch") or "")
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if not child_epoch:
            continue
        parsed_children.append((idx, namespace, name, child_epoch, child))
        manifest_counts[idx] = manifest_counts.get(idx, 0) + 1
    return parsed_children, manifest_counts


def _child_sweep_rows(epoch_dir: Path) -> list[SweepRow]:
    base = _results_base_from_sweep_epoch_dir(epoch_dir)
    if base is None:
        return []
    try:
        doc = orjson.loads((epoch_dir / "children.json").read_bytes())
    except (FileNotFoundError, OSError, orjson.JSONDecodeError):
        return []

    if not isinstance(doc, dict):
        return []
    children = doc.get("children")
    if not isinstance(children, list):
        return []

    parsed_children, manifest_counts = _parse_child_sweep_manifest(epoch_dir, children)

    rows: list[SweepRow] = []
    loaded_counts: dict[int, int] = {}
    for idx, namespace, name, child_epoch, child in parsed_children:
        summary = _load_profile_export(base / namespace / name / child_epoch)
        if summary is None:
            continue
        metrics_payload = (
            summary.get("metrics")
            if isinstance(summary.get("metrics"), dict)
            else summary
        )
        rows.append(
            (
                idx,
                _child_variation_values(child),
                _normalize_sweep_metrics(metrics_payload),
                {"child": child, "metrics": summary},
            )
        )
        loaded_counts[idx] = loaded_counts.get(idx, 0) + 1

    incomplete_repeated_idxs = {
        idx
        for idx, manifest_count in manifest_counts.items()
        if manifest_count > 1 and loaded_counts.get(idx, 0) != manifest_count
    }
    if incomplete_repeated_idxs:
        rows = [row for row in rows if int(row[0]) not in incomplete_repeated_idxs]
    return rows


def _load_sweep_child_refs(epoch_dir: Path) -> dict[int, tuple[str, str, str]]:
    path = epoch_dir / "children.json"
    try:
        doc = orjson.loads(path.read_bytes())
    except (FileNotFoundError, OSError, orjson.JSONDecodeError):
        return {}

    if not isinstance(doc, dict):
        return {}
    children = doc.get("children")
    if not isinstance(children, list):
        return {}

    refs: dict[int, tuple[str, str, str]] = {}
    duplicate_idxs: set[int] = set()
    for child in children:
        if not isinstance(child, dict):
            continue
        try:
            idx = int(child["variation_index"])
            ref = (
                child.get("namespace") or "",
                child["name"],
                child.get("child_run_epoch") or "",
            )
        except (KeyError, TypeError, ValueError):
            continue
        if idx in refs:
            duplicate_idxs.add(idx)
            refs.pop(idx, None)
            continue
        if idx not in duplicate_idxs:
            refs[idx] = ref
    return refs


def _child_ref_from_row_blob(row_blob: dict[str, Any]) -> tuple[str, str, str] | None:
    child = row_blob.get("child")
    if not isinstance(child, dict):
        return None
    namespace = child.get("namespace")
    name = child.get("name")
    epoch = child.get("child_run_epoch")
    if not (namespace and name and epoch):
        return None
    return str(namespace), str(name), str(epoch)


def _stable_variation_values(values: dict[str, Any]) -> dict[str, Any]:
    stable = dict(values)
    stable.pop("trial_index", None)
    return stable


def _mean_finite_numeric(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    finite = [value for value in numeric if math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _aggregate_sweep_metrics(trial_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for name in _NARROW_METRICS:
        metric_trials = [
            m.get(name) for m in trial_metrics if isinstance(m.get(name), dict)
        ]
        if not metric_trials:
            continue
        entry: dict[str, Any] = {}
        for stat in ("avg", "p50", "p99"):
            mean = _mean_finite_numeric([m.get(stat) for m in metric_trials])
            if mean is not None:
                entry[stat] = mean
        units = [m.get("unit") for m in metric_trials if m.get("unit") is not None]
        if units and all(unit == units[0] for unit in units):
            entry["unit"] = units[0]
        if entry:
            aggregated[name] = entry
    return aggregated


def _aggregate_sweep_rows(
    variation_idx: int,
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, str, str] | None, bytes]:
    variation_values = _stable_variation_values(rows[0][0])
    metrics = rows[0][1]
    child_ref = _child_ref_from_row_blob(rows[0][2])
    metrics_blob = zstandard.ZstdCompressor().compress(orjson.dumps(rows[0][2]))
    if len(rows) == 1:
        return variation_values, metrics, child_ref, metrics_blob

    trial_entries: list[dict[str, Any]] = []
    for values, trial_metrics, row_blob in rows:
        trial = {
            "variation_values": values,
            "stable_variation_values": _stable_variation_values(values),
            "metrics": trial_metrics,
            "row": row_blob,
        }
        ref = _child_ref_from_row_blob(row_blob)
        if ref is not None:
            child_ns, child_job, child_epoch = ref
            trial["child_ref"] = {
                "namespace": child_ns,
                "job_id": child_job,
                "epoch": child_epoch,
            }
        trial_entries.append(trial)

    aggregate_blob = {
        "aggregation_method": "mean_finite_numeric_default_compare_metrics",
        "variation_idx": variation_idx,
        "variation_values": variation_values,
        "trial_count": len(rows),
        "trials": trial_entries,
    }
    return (
        variation_values,
        _aggregate_sweep_metrics([metrics for _, metrics, _ in rows]),
        None,
        zstandard.ZstdCompressor().compress(orjson.dumps(aggregate_blob)),
    )


def _variation_key(values: dict[str, Any]) -> bytes:
    return orjson.dumps(values, option=orjson.OPT_SORT_KEYS)


def _best_parameter_keys(best_configurations: Any) -> set[bytes]:
    if not isinstance(best_configurations, dict):
        return set()
    keys: set[bytes] = set()
    for item in best_configurations.values():
        if isinstance(item, dict) and isinstance(item.get("parameters"), dict):
            keys.add(_variation_key(item["parameters"]))
    return keys


def _parameter_keys(entries: Any) -> set[bytes]:
    """Hash the ``parameters`` sub-dict of each entry, like _best_parameter_keys.

    The pareto set used to be hashed as whole entries (index + metrics +
    parameters), which can never equal a row's parameter hash -- so the
    parameter-matching path was dead and only variation_idx could match.
    """
    if not isinstance(entries, list):
        return set()
    return {
        _variation_key(e["parameters"])
        for e in entries
        if isinstance(e, dict) and isinstance(e.get("parameters"), dict)
    }


def _label_keys(entries: Any) -> set[tuple[Any, Any]]:
    """(variation_label, trial_index) pairs, the shape child rows carry.

    On the Kubernetes children path a row's values are
    ``{variation_label, trial_index}`` -- there are no parameter dicts to hash
    at all, so matching had to fall through to variation_idx or nothing.
    """
    if not isinstance(entries, list):
        return set()
    return {
        (e.get("variation_label"), e.get("trial_index"))
        for e in entries
        if isinstance(e, dict) and e.get("variation_label") is not None
    }


def _sweep_rankings(
    agg: dict[str, Any], indexed_rows: list[tuple[int, dict[str, Any]]]
) -> list[tuple[int, int | None, bool]]:
    pareto = agg.get("pareto_optimal", []) or []
    best = agg.get("best_configurations", []) or []
    pareto_idxs = {p.get("variation_idx") for p in pareto if isinstance(p, dict)}
    best_idxs = {b.get("variation_idx") for b in best if isinstance(b, dict)}
    pareto_param_keys = _parameter_keys(pareto)
    best_param_keys = _best_parameter_keys(best)
    pareto_labels = _label_keys(pareto)
    best_labels = _label_keys(
        [b.get("parameters", b) if isinstance(b, dict) else b for b in best]
    )
    if not (
        pareto_idxs
        or best_idxs
        or pareto_param_keys
        or best_param_keys
        or pareto_labels
        or best_labels
    ):
        return []

    rankings: list[tuple[int, int | None, bool]] = []
    for idx, variation_values in indexed_rows:
        key = _variation_key(variation_values)
        label = (
            variation_values.get("variation_label"),
            variation_values.get("trial_index"),
        )
        on_front = (
            idx in pareto_idxs or key in pareto_param_keys or label in pareto_labels
        )
        rankings.append(
            (
                idx,
                # 0 means "on the Pareto front", matching search_history's
                # convention; None means "not ranked". This used to store the
                # variation index for front members and a magic 999 for the
                # rest, so a rank column held neither a rank nor a null.
                0 if on_front else None,
                idx in best_idxs or key in best_param_keys or label in best_labels,
            )
        )
    return rankings


def _select_sweep_rows(
    epoch_dir: Path, parent_agg: dict[str, Any]
) -> tuple[dict[str, Any], list[SweepRow]]:
    strategy_agg = _load_strategy_sweep_aggregate(epoch_dir)
    if strategy_agg is not None:
        strategy_rows = _strategy_sweep_rows(strategy_agg)
        if strategy_rows:
            return strategy_agg, strategy_rows

    source_agg = parent_agg
    rows = _legacy_sweep_rows(parent_agg)
    if rows:
        return source_agg, rows

    return source_agg, _child_sweep_rows(epoch_dir)


def _group_sweep_rows(
    namespace: str,
    sweep_name: str,
    sweep_epoch: str,
    rows: list[SweepRow],
) -> dict[int, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]]:
    grouped_rows: dict[
        int, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
    ] = {}
    for idx, variation_values, metrics, row_blob in rows:
        try:
            variation_idx = int(idx)
        except (TypeError, ValueError) as exc:
            logger.warning(
                f"sweep variation {namespace}/{sweep_name} epoch={sweep_epoch} "
                f"idx={idx}: skipping index upsert "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        grouped_rows.setdefault(variation_idx, []).append(
            (variation_values, metrics, row_blob)
        )
    return grouped_rows


async def _upsert_grouped_sweep_rows(
    namespace: str,
    sweep_name: str,
    sweep_epoch: str,
    *,
    source_agg: dict[str, Any],
    grouped_rows: dict[
        int, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
    ],
) -> dict[int, dict[str, Any]]:
    indexed_rows: dict[int, dict[str, Any]] = {}
    for variation_idx, variation_rows in grouped_rows.items():
        try:
            variation_values, metrics, child_ref, metrics_blob = _aggregate_sweep_rows(
                variation_idx, variation_rows
            )
            await upsert_sweep_variation(
                namespace,
                sweep_name,
                sweep_epoch,
                variation_idx,
                variation_values=variation_values,
                mode=_sweep_mode(source_agg),
                phase="Succeeded",
                metrics=metrics,
                child_ref=child_ref,
                metrics_blob=metrics_blob,
            )
        except (
            sqlite3.Error,
            orjson.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            # One bad variation row (sqlite constraint, unencodable metric value,
            # bogus variation_idx coercion) must not poison the rest. The
            # backfill path retries on next call so a transient sqlite error
            # is self-healing.
            logger.warning(
                f"sweep variation {namespace}/{sweep_name} epoch={sweep_epoch} "
                f"idx={variation_idx}: skipping index upsert "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        indexed_rows[variation_idx] = variation_values
    return indexed_rows


def _collect_sweep_rows_from_disk(
    epoch_dir: Path,
) -> tuple[dict[str, Any], list[SweepRow]] | None:
    """Read aggregate/children/child-export files for one sweep epoch dir.

    Pure filesystem reads + zstd decompression — no sqlite access, no module
    state — so :func:`_index_sweep_from_disk` can offload it via
    ``asyncio.to_thread`` without racing the single-writer connection.
    Returns ``(source_agg, rows)`` or ``None`` when there is nothing to ingest.
    """
    aggregate_path = epoch_dir / "aggregate.json"
    if not aggregate_path.exists():
        return None
    try:
        parent_agg = orjson.loads(aggregate_path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    return _select_sweep_rows(epoch_dir, parent_agg)


async def _index_sweep_from_disk(
    namespace: str, sweep_name: str, sweep_epoch: str, epoch_dir: Path
) -> int:
    """Ingest <ns>/sweeps/<name>/<epoch>/ — variations + pareto if present.

    Returns the number of variation rows indexed. K8s sweep-controller archives
    put parent status in ``aggregate.json``, child linkage in ``children.json``,
    and per-cell metrics in the child runs' ``profile_export_aiperf.json`` files;
    legacy archives may use ``profile_export_aiperf_sweep.json`` or put
    ``per_combination_metrics`` directly in ``aggregate.json``.

    The row collection (read + decompress every child run's export) runs in a
    worker thread so a large sweep cannot stall the kopf event loop; the
    sqlite upserts stay on the loop/writer path.
    """
    collected = await asyncio.to_thread(_collect_sweep_rows_from_disk, epoch_dir)
    if collected is None:
        return 0

    source_agg, rows = collected
    grouped_rows = _group_sweep_rows(namespace, sweep_name, sweep_epoch, rows)
    indexed_rows = await _upsert_grouped_sweep_rows(
        namespace,
        sweep_name,
        sweep_epoch,
        source_agg=source_agg,
        grouped_rows=grouped_rows,
    )

    rankings = _sweep_rankings(source_agg, list(indexed_rows.items()))
    if rankings:
        await mark_sweep_pareto(
            namespace,
            sweep_name,
            sweep_epoch,
            rankings=rankings,
        )

    return len(indexed_rows)


async def lazy_backfill_run(
    base: Path, namespace: str, job_id: str, epoch: str
) -> None:
    """Background task fired from writer read-path fallback. Best-effort, never raises."""
    if is_readonly():
        return
    try:
        latest_epoch = resolve_latest(base, namespace, job_id)
        indexed = await _index_run_from_disk(
            base,
            namespace,
            job_id,
            epoch,
            is_latest=(epoch == latest_epoch),
        )
        if indexed:
            finish_catalog_update(
                _catalog_update_marker(base, namespace, job_id, epoch)
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


def _escape_like(value: str) -> str:
    r"""Escape LIKE wildcards in a user-supplied substring filter.

    Parameter binding stops SQL injection but not wildcard interpretation, so
    an unescaped filter silently over-matches: ``_`` matches any character and
    ``%`` matches anything at all. Model ids contain both. Callers must pair
    this with ``ESCAPE '\'`` in the SQL.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def leaderboard(
    metric: str = "request_throughput",
    stat: str = "avg",
    order: str = "desc",
    limit: int = 20,
    *,
    epoch: str | None = None,
) -> list[dict[str, Any]]:
    """Rank indexed runs by one ``metric``/``stat`` column, best first."""
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
    namespace: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    metric: str = "request_throughput",
    stat: str = "avg",
    limit: int = 100,
    epoch: str | None = None,
) -> list[dict[str, Any]]:
    """List indexed runs over time, optionally filtered by model/endpoint."""
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
    if namespace:
        where.append("namespace = ?")
        params.append(namespace)
    if model:
        where.append(r"model LIKE ? ESCAPE '\'")
        params.append(f"%{_escape_like(model)}%")
    if endpoint:
        where.append(r"endpoint LIKE ? ESCAPE '\'")
        params.append(f"%{_escape_like(endpoint)}%")
    params.append(limit)

    # Newest N, returned oldest-first. LIMIT applies to a DESC scan and the
    # outer query re-sorts for the caller: with a plain ASC LIMIT the endpoint
    # returned runs 1-100 of a 300-run namespace, so the trend chart silently
    # froze once history outgrew the limit.
    sql = (
        f"SELECT * FROM ("
        f"SELECT namespace, job_id, epoch, {col} AS value, "
        f"       {metric}_unit AS unit, start_time, model, endpoint "
        f"FROM runs WHERE {' AND '.join(where)} "
        f"ORDER BY start_time DESC LIMIT ?"
        f") ORDER BY start_time ASC"
    )
    return await _select_dicts(sql, tuple(params))


def _split_compare_job_ids(
    job_ids: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    bare_job_ids: list[str] = []
    qualified_refs: list[tuple[str, str]] = []
    for job_id in job_ids:
        if "/" in job_id:
            namespace, name = job_id.split("/", 1)
            if namespace and name:
                qualified_refs.append((namespace, name))
        else:
            bare_job_ids.append(job_id)
    return bare_job_ids, qualified_refs


def _compare_identity_filter(job_ids: list[str]) -> tuple[str | None, list[Any]]:
    bare_job_ids, qualified_refs = _split_compare_job_ids(job_ids)
    identity_clauses: list[str] = []
    params: list[Any] = []
    if bare_job_ids:
        placeholders = ", ".join("?" * len(bare_job_ids))
        identity_clauses.append(f"job_id IN ({placeholders})")
        params.extend(bare_job_ids)
    for namespace, name in qualified_refs:
        identity_clauses.append("(namespace = ? AND job_id = ?)")
        params.extend([namespace, name])
    if not identity_clauses:
        return None, []
    return f"({' OR '.join(identity_clauses)})", params


async def compare(
    job_ids: list[str],
    metrics: list[str] | None = None,
    *,
    epoch: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch the requested metric columns for specific ``job_ids`` side by side."""
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

    identity_filter, params = _compare_identity_filter(job_ids)
    if identity_filter is None:
        return []

    where = [identity_filter]
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
