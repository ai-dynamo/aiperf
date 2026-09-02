# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runs_index.scatter_data()."""

from __future__ import annotations

import aiosqlite
import pytest


@pytest.mark.asyncio
async def test_scatter_data_returns_rows_with_metrics(tmp_path):
    """scatter_data() returns flat metric rows from the runs table.

    Only is_latest=1 rows are returned (one row per job, the most recent run).
    Older epochs and rows with all-NULL metrics are excluded.
    """
    db_path = tmp_path / "test.sqlite"

    # Minimal schema: only the columns scatter_data() touches, including
    # is_latest which the WHERE clause now requires.
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("""
            CREATE TABLE runs (
                namespace TEXT NOT NULL,
                job_id TEXT NOT NULL,
                epoch TEXT NOT NULL,
                is_latest INTEGER NOT NULL DEFAULT 0,
                model TEXT,
                request_throughput_avg REAL,
                request_latency_p99 REAL,
                time_to_first_token_avg REAL,
                output_token_throughput_avg REAL
            )
        """)
        # job-a has two epochs: older (is_latest=0) and latest (is_latest=1).
        # Only the latest epoch should appear in scatter results.
        # job-b is latest but has all-NULL metrics — excluded by the OR clause.
        await db.execute("""
            INSERT INTO runs VALUES
              ('ns1', 'job-a', '1.0', 0, 'llama3', 99.9, 999.9, 99.9, 9999.9),
              ('ns1', 'job-a', '2.0', 1, 'llama3', 42.5, 300.1, 50.2, 1500.0),
              ('ns1', 'job-b', '1.0', 1, 'gpt-j',  NULL, NULL,  NULL, NULL)
        """)
        await db.commit()

    import aiperf.operator.runs_index as idx

    # Patch _DB to the test connection
    original_db = idx._DB
    async with aiosqlite.connect(f"file:{db_path}?mode=ro", uri=True) as test_conn:
        idx._DB = test_conn
        try:
            rows = await idx.scatter_data()
        finally:
            idx._DB = original_db

    # Only the is_latest=1 epoch of job-a with metrics should appear.
    # The older epoch '1.0' and job-b (all NULL) are excluded.
    assert len(rows) == 1
    assert rows[0]["job_id"] == "job-a"
    assert rows[0]["epoch"] == "2.0"
    assert rows[0]["request_throughput_avg"] == pytest.approx(42.5)
    assert rows[0]["request_latency_p99"] == pytest.approx(300.1)
    assert rows[0]["time_to_first_token_avg"] == pytest.approx(50.2)
    assert rows[0]["output_token_throughput_avg"] == pytest.approx(1500.0)
    assert rows[0]["namespace"] == "ns1"


@pytest.mark.asyncio
async def test_scatter_data_empty_table(tmp_path):
    """scatter_data() returns empty list when no rows match."""
    db_path = tmp_path / "test.sqlite"
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("""
            CREATE TABLE runs (
                namespace TEXT NOT NULL, job_id TEXT NOT NULL, epoch TEXT NOT NULL,
                is_latest INTEGER NOT NULL DEFAULT 0,
                model TEXT,
                request_throughput_avg REAL, request_latency_p99 REAL,
                time_to_first_token_avg REAL, output_token_throughput_avg REAL
            )
        """)
        await db.commit()

    import aiperf.operator.runs_index as idx

    original_db = idx._DB
    async with aiosqlite.connect(f"file:{db_path}?mode=ro", uri=True) as test_conn:
        idx._DB = test_conn
        try:
            rows = await idx.scatter_data()
        finally:
            idx._DB = original_db

    assert rows == []
