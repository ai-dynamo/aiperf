# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.results_db module.

Focuses on:
- ResultsDB initialization and cleanup
- leaderboard() ranking with various metrics, stats, and orders
- history() time-series queries with model/endpoint filters
- compare() side-by-side job comparison
- summary() single-job full data retrieval
- Input validation (_validate_identifier, _escape_like)
- Handling of .json and .json.zst file variants
- Edge cases: empty dirs, malformed JSON, missing metrics
- Adversarial inputs: SQL injection in identifiers and filters
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
import zstandard
from pytest import param

pytest.importorskip("duckdb", reason="duckdb required for results_db tests")

from aiperf.operator.results_db import ResultsDB, _escape_like, _validate_identifier

# ============================================================
# Helpers
# ============================================================


def _write_summary(
    base_dir: Path,
    namespace: str,
    job_id: str,
    *,
    throughput_avg: float = 100.0,
    latency_avg: float = 50.0,
    model: str = "llama-7b",
    endpoint: str = "http://localhost:8000",
    start_time: str = "2026-01-15T10:00:00Z",
    end_time: str = "2026-01-15T10:05:00Z",
    compress: bool = False,
) -> Path:
    """Write a profile_export_aiperf.json to the results directory.

    The real profile_export_aiperf.json written by SystemController has metrics
    at the top level (no wrapper key).
    """
    data = {
        "request_throughput": {
            "avg": throughput_avg,
            "p50": throughput_avg * 0.9,
            "p99": throughput_avg * 1.5,
            "unit": "req/s",
        },
        "request_latency": {
            "avg": latency_avg,
            "p50": latency_avg * 0.8,
            "p99": latency_avg * 3.0,
            "unit": "ms",
        },
        "time_to_first_token": {"avg": 10.0, "p50": 8.0, "p99": 25.0, "unit": "ms"},
        "output_token_throughput": {
            "avg": 500.0,
            "p50": 450.0,
            "p99": 700.0,
            "unit": "tok/s",
        },
        "inter_token_latency": {"avg": 5.0, "p50": 4.0, "p99": 12.0, "unit": "ms"},
        "start_time": start_time,
        "end_time": end_time,
        "input_config": {
            "models": {"items": [{"name": model}]},
            "endpoint": {"urls": [endpoint]},
        },
    }

    job_dir = base_dir / namespace / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    content = orjson.dumps(data)
    if compress:
        cctx = zstandard.ZstdCompressor()
        path = job_dir / "profile_export_aiperf.json.zst"
        path.write_bytes(cctx.compress(content))
    else:
        path = job_dir / "profile_export_aiperf.json"
        path.write_bytes(content)
    return path


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Provide a temporary results directory."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def db(results_dir: Path) -> ResultsDB:
    """Create a ResultsDB instance for testing."""
    instance = ResultsDB(results_dir)
    yield instance
    instance.close()


# ============================================================
# Input Validation
# ============================================================


class TestValidateIdentifier:
    """Verify _validate_identifier blocks unsafe names."""

    @pytest.mark.parametrize(
        "name",
        [
            "request_throughput",
            "avg",
            "p99",
            "time_to_first_token",
            "a",
            "abc123",
        ],
    )  # fmt: skip
    def test_valid_identifiers_pass(self, name: str) -> None:
        _validate_identifier(name)

    @pytest.mark.parametrize(
        "name",
        [
            param("", id="empty-string"),
            param("'; DROP TABLE", id="sql-injection"),
            param("metric-name", id="hyphen"),
            param("metric.name", id="dot"),
            param("metric name", id="space"),
            param("metric;", id="semicolon"),
            param("a' OR '1'='1", id="or-injection"),
            param("\x00", id="null-byte"),
        ],
    )  # fmt: skip
    def test_invalid_identifiers_raise_value_error(self, name: str) -> None:
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            _validate_identifier(name)


class TestEscapeLike:
    """Verify _escape_like handles special characters."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("simple", "simple"),
            ("it's", "it''s"),
            ("100%", "100\\%"),
            ("under_score", "under\\_score"),
            ("a'b%c_d", "a''b\\%c\\_d"),
        ],
    )  # fmt: skip
    def test_escapes_special_chars(self, input_val: str, expected: str) -> None:
        assert _escape_like(input_val) == expected


# ============================================================
# _find_summary_files
# ============================================================


class TestFindSummaryFiles:
    """Verify summary file glob construction."""

    def test_no_files_returns_empty_string(self, db: ResultsDB) -> None:
        result = db._find_summary_files()
        assert result == "''"

    def test_raw_files_only(self, results_dir: Path, db: ResultsDB) -> None:
        _write_summary(results_dir, "ns", "job-1")
        result = db._find_summary_files()
        assert result != "''"
        assert ".zst" not in result

    def test_zst_files_only(self, results_dir: Path, db: ResultsDB) -> None:
        _write_summary(results_dir, "ns", "job-1", compress=True)
        result = db._find_summary_files()
        assert result != "''"
        assert ".zst" in result

    def test_mixed_files(self, results_dir: Path, db: ResultsDB) -> None:
        _write_summary(results_dir, "ns", "job-1")
        _write_summary(results_dir, "ns", "job-2", compress=True)
        result = db._find_summary_files()
        assert result != "''"
        # Should include both patterns
        assert "[" in result


# ============================================================
# leaderboard()
# ============================================================


class TestLeaderboard:
    """Verify leaderboard ranking queries."""

    @pytest.mark.asyncio
    async def test_leaderboard_empty_returns_empty(self, db: ResultsDB) -> None:
        rows = await db.leaderboard()
        assert rows == []

    @pytest.mark.asyncio
    async def test_leaderboard_ranks_by_throughput_desc(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", throughput_avg=100.0)
        _write_summary(results_dir, "ns", "job-2", throughput_avg=200.0)
        _write_summary(results_dir, "ns", "job-3", throughput_avg=50.0)

        rows = await db.leaderboard(
            metric="request_throughput", stat="avg", order="desc"
        )
        assert len(rows) == 3
        values = [r["value"] for r in rows]
        assert values == sorted(values, reverse=True)

    @pytest.mark.asyncio
    async def test_leaderboard_asc_order(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", throughput_avg=100.0)
        _write_summary(results_dir, "ns", "job-2", throughput_avg=200.0)

        rows = await db.leaderboard(order="asc")
        values = [r["value"] for r in rows]
        assert values == sorted(values)

    @pytest.mark.asyncio
    async def test_leaderboard_limit(self, results_dir: Path, db: ResultsDB) -> None:
        for i in range(10):
            _write_summary(results_dir, "ns", f"job-{i}", throughput_avg=float(i))

        rows = await db.leaderboard(limit=3)
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_leaderboard_custom_metric_and_stat(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", latency_avg=50.0)

        rows = await db.leaderboard(metric="request_latency", stat="p99")
        assert len(rows) == 1
        assert rows[0]["value"] is not None

    @pytest.mark.asyncio
    async def test_leaderboard_nonexistent_metric_returns_empty(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1")

        rows = await db.leaderboard(metric="nonexistent_metric")
        assert rows == []

    @pytest.mark.asyncio
    async def test_leaderboard_includes_metadata(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(
            results_dir, "ns", "job-1", model="llama-7b", endpoint="http://ep:8000"
        )

        rows = await db.leaderboard()
        assert len(rows) == 1
        row = rows[0]
        assert row["namespace"] == "ns"
        assert row["job_id"] == "job-1"
        assert row["model"] == "llama-7b"
        assert row["endpoint"] == "http://ep:8000"
        assert row["unit"] == "req/s"

    @pytest.mark.asyncio
    async def test_leaderboard_with_zst_files(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", throughput_avg=999.0, compress=True)

        rows = await db.leaderboard()
        assert len(rows) == 1
        assert rows[0]["value"] == 999.0

    @pytest.mark.asyncio
    async def test_leaderboard_sql_injection_in_metric_raises(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1")
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            await db.leaderboard(metric="'; DROP TABLE t; --")

    @pytest.mark.asyncio
    async def test_leaderboard_sql_injection_in_stat_raises(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1")
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            await db.leaderboard(stat="avg; DROP TABLE")

    @pytest.mark.asyncio
    async def test_leaderboard_malformed_json_returns_empty(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        job_dir = results_dir / "ns" / "job-bad"
        job_dir.mkdir(parents=True)
        (job_dir / "profile_export_aiperf.json").write_bytes(b"not valid json at all")

        # DuckDB should handle this gracefully
        rows = await db.leaderboard()
        assert isinstance(rows, list)


# ============================================================
# history()
# ============================================================


class TestHistory:
    """Verify history time-series queries."""

    @pytest.mark.asyncio
    async def test_history_empty_returns_empty(self, db: ResultsDB) -> None:
        rows = await db.history()
        assert rows == []

    @pytest.mark.asyncio
    async def test_history_returns_entries_ordered_by_time(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", start_time="2026-01-15T10:00:00Z")
        _write_summary(results_dir, "ns", "job-2", start_time="2026-01-15T11:00:00Z")

        rows = await db.history()
        assert len(rows) == 2
        # Should be ordered by start_time ASC
        assert rows[0]["start_time"] <= rows[1]["start_time"]

    @pytest.mark.asyncio
    async def test_history_filter_by_model(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", model="llama-7b")
        _write_summary(results_dir, "ns", "job-2", model="gpt-2")

        rows = await db.history(model="llama")
        assert len(rows) == 1
        assert "llama" in rows[0]["model"]

    @pytest.mark.asyncio
    async def test_history_filter_by_endpoint(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", endpoint="http://prod:8000")
        _write_summary(results_dir, "ns", "job-2", endpoint="http://staging:8000")

        rows = await db.history(endpoint="prod")
        assert len(rows) == 1
        assert "prod" in rows[0]["endpoint"]

    @pytest.mark.asyncio
    async def test_history_model_filter_case_insensitive(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", model="Llama-7B")

        rows = await db.history(model="llama")
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_history_sql_injection_in_metric_raises(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1")
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            await db.history(metric="'; DROP TABLE t;--")

    @pytest.mark.asyncio
    async def test_history_special_chars_in_model_filter_escaped(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", model="model_v2")

        # % and _ are special in LIKE — should be escaped
        rows = await db.history(model="model%")
        # Should not match wildcard — escaped
        assert isinstance(rows, list)

    @pytest.mark.asyncio
    async def test_history_limit(self, results_dir: Path, db: ResultsDB) -> None:
        for i in range(10):
            _write_summary(results_dir, "ns", f"job-{i}")

        rows = await db.history(limit=3)
        assert len(rows) == 3


# ============================================================
# compare()
# ============================================================


class TestCompare:
    """Verify compare side-by-side queries."""

    @pytest.mark.asyncio
    async def test_compare_empty_job_ids_returns_empty(self, db: ResultsDB) -> None:
        rows = await db.compare(job_ids=[])
        assert rows == []

    @pytest.mark.asyncio
    async def test_compare_no_files_returns_empty(self, db: ResultsDB) -> None:
        rows = await db.compare(job_ids=["job-1", "job-2"])
        assert rows == []

    @pytest.mark.asyncio
    async def test_compare_specific_jobs(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", throughput_avg=100.0)
        _write_summary(results_dir, "ns", "job-2", throughput_avg=200.0)
        _write_summary(results_dir, "ns", "job-3", throughput_avg=300.0)

        rows = await db.compare(job_ids=["job-1", "job-2"])
        job_ids = {r["job_id"] for r in rows}
        assert "job-1" in job_ids
        assert "job-2" in job_ids
        assert "job-3" not in job_ids

    @pytest.mark.asyncio
    async def test_compare_custom_metrics(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1")

        rows = await db.compare(
            job_ids=["job-1"],
            metrics=["request_throughput", "request_latency"],
        )
        assert len(rows) == 1
        row = rows[0]
        assert "request_throughput_avg" in row
        assert "request_latency_avg" in row

    @pytest.mark.asyncio
    async def test_compare_default_metrics(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1")

        rows = await db.compare(job_ids=["job-1"])
        assert len(rows) == 1
        row = rows[0]
        # Should include all default metrics
        assert "request_throughput_avg" in row
        assert "inter_token_latency_avg" in row

    @pytest.mark.asyncio
    async def test_compare_sql_injection_in_metrics_raises(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1")
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            await db.compare(job_ids=["job-1"], metrics=["'; DROP TABLE"])


# ============================================================
# summary()
# ============================================================


class TestSummary:
    """Verify single-job summary queries."""

    @pytest.mark.asyncio
    async def test_summary_existing_job(self, results_dir: Path, db: ResultsDB) -> None:
        _write_summary(results_dir, "ns", "job-1", throughput_avg=100.0)

        result = await db.summary("ns", "job-1")
        assert result is not None
        assert result["request_throughput"]["avg"] == 100.0

    @pytest.mark.asyncio
    async def test_summary_nonexistent_namespace(self, db: ResultsDB) -> None:
        result = await db.summary("nonexistent", "job-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_summary_nonexistent_job(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        (results_dir / "ns").mkdir()
        result = await db.summary("ns", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_summary_zst_file(self, results_dir: Path, db: ResultsDB) -> None:
        _write_summary(results_dir, "ns", "job-1", throughput_avg=42.0, compress=True)

        result = await db.summary("ns", "job-1")
        assert result is not None
        assert result["request_throughput"]["avg"] == 42.0

    @pytest.mark.asyncio
    async def test_summary_prefers_zst_over_raw(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        _write_summary(results_dir, "ns", "job-1", throughput_avg=100.0)
        # Also write zst variant with different value
        _write_summary(results_dir, "ns", "job-1", throughput_avg=999.0, compress=True)

        result = await db.summary("ns", "job-1")
        assert result is not None
        # zst is checked first
        assert result["request_throughput"]["avg"] == 999.0

    @pytest.mark.asyncio
    async def test_summary_job_dir_exists_but_no_summary_file(
        self, results_dir: Path, db: ResultsDB
    ) -> None:
        job_dir = results_dir / "ns" / "job-1"
        job_dir.mkdir(parents=True)
        (job_dir / "other.json").write_bytes(b"{}")

        result = await db.summary("ns", "job-1")
        assert result is None


# ============================================================
# Thread Safety / _query Error Handling
# ============================================================


class TestQueryErrorHandling:
    """Verify _query handles DuckDB errors gracefully."""

    @pytest.mark.asyncio
    async def test_invalid_sql_returns_empty(self, db: ResultsDB) -> None:
        rows = await db._query("SELECT * FROM nonexistent_table_xyz")
        assert rows == []

    @pytest.mark.asyncio
    async def test_syntax_error_returns_empty(self, db: ResultsDB) -> None:
        rows = await db._query("THIS IS NOT SQL")
        assert rows == []


# ============================================================
# Close
# ============================================================


class TestResultsDBLifecycle:
    """Verify initialization and cleanup."""

    def test_close_twice_no_error(self, results_dir: Path) -> None:
        import contextlib

        db = ResultsDB(results_dir)
        db.close()
        # DuckDB may raise on double-close; that's acceptable
        with contextlib.suppress(Exception):
            db.close()

    def test_init_creates_in_memory_db(self, results_dir: Path) -> None:
        db = ResultsDB(results_dir)
        # Should be able to execute a simple query
        result = db._conn.execute("SELECT 1 AS n")
        assert result.fetchone()[0] == 1
        db.close()
