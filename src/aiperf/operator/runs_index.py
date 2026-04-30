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
from pathlib import Path

import aiosqlite

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
