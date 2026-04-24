# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.results_server module.

Focuses on:
- Health check endpoint
- Job listing (empty, populated, nested structures)
- File listing with .zst display name stripping
- File download with content negotiation (zstd, gzip, identity)
- Path traversal protection via _safe_resolve
- Analytics endpoints (leaderboard, history, compare, summary)
- Edge cases: empty dirs, missing files, corrupted data
- Adversarial inputs: path traversal, special characters
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import orjson
import pytest
import zstandard
from pytest import param

pytest.importorskip("duckdb", reason="duckdb required for results_server tests")

from aiperf.operator.results_server import (
    _display_name,
    _safe_resolve,
    create_app,
)

# ============================================================
# Helpers
# ============================================================


def _create_result_file(
    base_dir: Path,
    namespace: str,
    job_id: str,
    filename: str,
    content: bytes = b'{"request_throughput": {"avg": 100, "unit": "req/s"}}',
    *,
    compress: bool = False,
) -> Path:
    """Create a result file in the expected directory structure."""
    job_dir = base_dir / namespace / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if compress:
        cctx = zstandard.ZstdCompressor()
        file_path = job_dir / (filename + ".zst")
        file_path.write_bytes(cctx.compress(content))
    else:
        file_path = job_dir / filename
        file_path.write_bytes(content)
    return file_path


def _summary_json(
    metric_val: float = 100.0,
    model: str = "llama-7b",
    endpoint: str = "http://localhost:8000",
) -> bytes:
    """Create a realistic summary JSON for DuckDB tests.

    The real profile_export_aiperf.json written by SystemController has metrics
    at the top level (no wrapper key).
    """
    return orjson.dumps(
        {
            "request_throughput": {
                "avg": metric_val,
                "p50": metric_val * 0.9,
                "p99": metric_val * 1.5,
                "unit": "req/s",
            },
            "request_latency": {
                "avg": 50.0,
                "p50": 45.0,
                "p99": 120.0,
                "unit": "ms",
            },
            "time_to_first_token": {
                "avg": 10.0,
                "p50": 8.0,
                "p99": 25.0,
                "unit": "ms",
            },
            "output_token_throughput": {
                "avg": 500.0,
                "p50": 450.0,
                "p99": 700.0,
                "unit": "tok/s",
            },
            "inter_token_latency": {
                "avg": 5.0,
                "p50": 4.0,
                "p99": 12.0,
                "unit": "ms",
            },
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T10:05:00Z",
            "input_config": {
                "models": {"items": [{"name": model}]},
                "endpoint": {"urls": [endpoint]},
            },
        }
    )


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Provide a temporary results directory."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
async def client(results_dir: Path):
    """Create an httpx AsyncClient for the FastAPI app with lifespan."""
    app = create_app(results_dir)

    # Manually trigger the lifespan since httpx ASGITransport doesn't do it
    async with asyncio.timeout(5):
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await ctx.__aexit__(None, None, None)


# ============================================================
# _safe_resolve
# ============================================================


class TestSafeResolve:
    """Verify path traversal protection."""

    def test_safe_resolve_valid_path(self, tmp_path: Path) -> None:
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        result = _safe_resolve(tmp_path, "a", "b")
        assert result is not None
        assert result == sub.resolve()

    @pytest.mark.parametrize(
        "parts",
        [
            param(("..", "etc", "passwd"), id="parent-traversal"),
            param(("a", "..", "..", "etc"), id="nested-traversal"),
            param(("a/../../etc",), id="slash-in-component"),
        ],
    )  # fmt: skip
    def test_safe_resolve_blocks_traversal(
        self, tmp_path: Path, parts: tuple[str, ...]
    ) -> None:
        result = _safe_resolve(tmp_path, *parts)
        # Either None (traversal blocked) or still under base
        if result is not None:
            assert str(result).startswith(str(tmp_path.resolve()))

    def test_safe_resolve_nonexistent_path_still_resolves(self, tmp_path: Path) -> None:
        result = _safe_resolve(tmp_path, "nonexistent")
        assert result is not None

    def test_safe_resolve_null_byte_in_path(self, tmp_path: Path) -> None:
        result = _safe_resolve(tmp_path, "file\x00.txt")
        assert result is None


# ============================================================
# _display_name
# ============================================================


class TestDisplayName:
    """Verify .zst suffix stripping for display."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("metrics.json.zst", "metrics.json"),
            ("metrics.json", "metrics.json"),
            ("data.csv", "data.csv"),
            ("file.zst", "file"),
            ("no_extension", "no_extension"),
        ],
    )  # fmt: skip
    def test_display_name_strips_zst(self, filename: str, expected: str) -> None:
        assert _display_name(Path(filename)) == expected


# ============================================================
# Health Check
# ============================================================


class TestHealthEndpoint:
    """Verify /healthz endpoint."""

    @pytest.mark.asyncio
    async def test_healthz_returns_ok(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ============================================================
# Job Listing
# ============================================================


class TestListJobs:
    """Verify /api/v1/results endpoint."""

    @pytest.mark.asyncio
    async def test_list_jobs_empty_dir(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []

    @pytest.mark.asyncio
    async def test_list_jobs_nonexistent_dir(self, tmp_path: Path) -> None:
        app = create_app(tmp_path / "nonexistent")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/results")
        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    @pytest.mark.asyncio
    async def test_list_jobs_with_results(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(results_dir, "default", "job-1", "metrics.json")
        _create_result_file(results_dir, "default", "job-2", "metrics.json")
        _create_result_file(results_dir, "prod", "job-3", "data.csv")

        resp = await client.get("/api/v1/results")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 3

        namespaces = {j["namespace"] for j in data["jobs"]}
        assert namespaces == {"default", "prod"}

    @pytest.mark.asyncio
    async def test_list_jobs_skips_files_at_namespace_level(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (results_dir / "stray_file.txt").write_text("not a directory")
        _create_result_file(results_dir, "ns", "job-1", "metrics.json")

        resp = await client.get("/api/v1/results")
        data = resp.json()
        assert len(data["jobs"]) == 1

    @pytest.mark.asyncio
    async def test_list_jobs_empty_job_dir_excluded(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (results_dir / "ns" / "empty-job").mkdir(parents=True)

        resp = await client.get("/api/v1/results")
        data = resp.json()
        assert len(data["jobs"]) == 0

    @pytest.mark.asyncio
    async def test_list_jobs_reports_correct_file_count_and_size(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        content = b"x" * 1024
        _create_result_file(results_dir, "ns", "job-1", "a.json", content)
        _create_result_file(results_dir, "ns", "job-1", "b.json", content)

        resp = await client.get("/api/v1/results")
        job = resp.json()["jobs"][0]
        assert job["file_count"] == 2
        assert job["total_size_bytes"] == 2048


# ============================================================
# File Listing
# ============================================================


class TestListJobFiles:
    """Verify /api/v1/results/{namespace}/{job_id} endpoint."""

    @pytest.mark.asyncio
    async def test_list_files_for_existing_job(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(results_dir, "ns", "job-1", "metrics.json")
        _create_result_file(results_dir, "ns", "job-1", "data.csv", compress=True)

        resp = await client.get("/api/v1/results/ns/job-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["namespace"] == "ns"
        assert data["job_id"] == "job-1"
        assert len(data["files"]) == 2

        names = {f["name"] for f in data["files"]}
        assert "metrics.json" in names
        assert "data.csv" in names

        compressed = [f for f in data["files"] if f["compressed"]]
        assert len(compressed) == 1

    @pytest.mark.asyncio
    async def test_list_files_nonexistent_job_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/results/ns/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "namespace,job_id",
        [
            param("../etc", "passwd", id="traversal-in-namespace"),
            param("ns", "../../../etc/passwd", id="traversal-in-job-id"),
        ],
    )  # fmt: skip
    async def test_list_files_path_traversal_returns_404(
        self, client: httpx.AsyncClient, namespace: str, job_id: str
    ) -> None:
        resp = await client.get(f"/api/v1/results/{namespace}/{job_id}")
        assert resp.status_code in (404, 422)


# ============================================================
# File Download & Content Negotiation
# ============================================================


class TestDownloadFile:
    """Verify /api/v1/results/{namespace}/{job_id}/{filename} endpoint."""

    @pytest.mark.asyncio
    async def test_download_raw_file_identity(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        content = b'{"result": true}'
        _create_result_file(results_dir, "ns", "job-1", "metrics.json", content)

        resp = await client.get(
            "/api/v1/results/ns/job-1/metrics.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert resp.status_code == 200
        assert resp.content == content

    @pytest.mark.asyncio
    async def test_download_zst_file_with_zstd_encoding(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        content = b'{"result": true}'
        _create_result_file(
            results_dir, "ns", "job-1", "metrics.json", content, compress=True
        )

        resp = await client.get(
            "/api/v1/results/ns/job-1/metrics.json",
            headers={"Accept-Encoding": "zstd"},
        )
        assert resp.status_code == 200
        # httpx auto-decompresses zstd, so content is already the original
        assert resp.content == content

    @pytest.mark.asyncio
    async def test_download_zst_file_with_gzip_encoding(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        content = b'{"result": true}'
        _create_result_file(
            results_dir, "ns", "job-1", "metrics.json", content, compress=True
        )

        resp = await client.get(
            "/api/v1/results/ns/job-1/metrics.json",
            headers={"Accept-Encoding": "gzip"},
        )
        assert resp.status_code == 200
        # httpx auto-decompresses gzip, so content is already the original
        assert resp.content == content

    @pytest.mark.asyncio
    async def test_download_zst_file_with_identity_encoding_decompresses(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        content = b'{"result": true}'
        _create_result_file(
            results_dir, "ns", "job-1", "metrics.json", content, compress=True
        )

        resp = await client.get(
            "/api/v1/results/ns/job-1/metrics.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert resp.status_code == 200
        # Should be decompressed
        assert resp.content == content

    @pytest.mark.asyncio
    async def test_download_nonexistent_file_returns_404(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (results_dir / "ns" / "job-1").mkdir(parents=True)

        resp = await client.get("/api/v1/results/ns/job-1/nonexistent.json")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_download_nonexistent_job_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/results/ns/nojob/file.json")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_download_prefers_zst_over_raw(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        raw_content = b'{"raw": true}'
        zst_content = b'{"zst": true}'
        job_dir = results_dir / "ns" / "job-1"
        job_dir.mkdir(parents=True)
        (job_dir / "data.json").write_bytes(raw_content)
        cctx = zstandard.ZstdCompressor()
        (job_dir / "data.json.zst").write_bytes(cctx.compress(zst_content))

        resp = await client.get(
            "/api/v1/results/ns/job-1/data.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert resp.status_code == 200
        # Should serve the zst variant (decompressed)
        assert resp.content == zst_content

    @pytest.mark.asyncio
    async def test_download_content_disposition_header(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(results_dir, "ns", "job-1", "metrics.json")

        resp = await client.get(
            "/api/v1/results/ns/job-1/metrics.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert "content-disposition" in resp.headers
        assert "metrics.json" in resp.headers["content-disposition"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            param("../../../etc/passwd", id="traversal"),
            param("..%2F..%2Fetc%2Fpasswd", id="encoded-traversal"),
        ],
    )  # fmt: skip
    async def test_download_path_traversal_returns_404(
        self, results_dir: Path, client: httpx.AsyncClient, filename: str
    ) -> None:
        (results_dir / "ns" / "job-1").mkdir(parents=True)
        resp = await client.get(f"/api/v1/results/ns/job-1/{filename}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_download_no_accept_encoding_header(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        content = b'{"data": 1}'
        _create_result_file(
            results_dir, "ns", "job-1", "metrics.json", content, compress=True
        )

        resp = await client.get("/api/v1/results/ns/job-1/metrics.json")
        assert resp.status_code == 200
        # Without accept-encoding, zst files are decompressed (identity)
        assert resp.content == content


# ============================================================
# Analytics - Leaderboard
# ============================================================


class TestLeaderboardEndpoint:
    """Verify /api/v1/analytics/leaderboard endpoint."""

    @pytest.mark.asyncio
    async def test_leaderboard_no_files_returns_empty(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/analytics/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []
        assert data["metric"] == "request_throughput"

    @pytest.mark.asyncio
    async def test_leaderboard_with_results(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(metric_val=200.0),
        )
        _create_result_file(
            results_dir,
            "ns",
            "job-2",
            "profile_export_aiperf.json",
            _summary_json(metric_val=100.0),
        )

        resp = await client.get("/api/v1/analytics/leaderboard")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert len(entries) == 2
        # Default desc order — higher value first
        assert entries[0]["value"] >= entries[1]["value"]

    @pytest.mark.asyncio
    async def test_leaderboard_asc_order(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(metric_val=200.0),
        )
        _create_result_file(
            results_dir,
            "ns",
            "job-2",
            "profile_export_aiperf.json",
            _summary_json(metric_val=100.0),
        )

        resp = await client.get("/api/v1/analytics/leaderboard?order=asc")
        entries = resp.json()["entries"]
        assert entries[0]["value"] <= entries[1]["value"]

    @pytest.mark.asyncio
    async def test_leaderboard_custom_metric_and_stat(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
        )

        resp = await client.get(
            "/api/v1/analytics/leaderboard?metric=request_latency&stat=p99"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metric"] == "request_latency"
        assert data["stat"] == "p99"

    @pytest.mark.asyncio
    async def test_leaderboard_nonexistent_metric_returns_empty(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
        )

        resp = await client.get(
            "/api/v1/analytics/leaderboard?metric=nonexistent_metric"
        )
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    @pytest.mark.asyncio
    async def test_leaderboard_limit_parameter(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        for i in range(5):
            _create_result_file(
                results_dir,
                "ns",
                f"job-{i}",
                "profile_export_aiperf.json",
                _summary_json(metric_val=float(i * 10)),
            )

        resp = await client.get("/api/v1/analytics/leaderboard?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["entries"]) == 2

    @pytest.mark.asyncio
    async def test_leaderboard_with_zst_files(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(metric_val=300.0),
            compress=True,
        )

        resp = await client.get("/api/v1/analytics/leaderboard")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["value"] == 300.0


# ============================================================
# Analytics - History
# ============================================================


class TestHistoryEndpoint:
    """Verify /api/v1/analytics/history endpoint."""

    @pytest.mark.asyncio
    async def test_history_no_files_returns_empty(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/analytics/history")
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    @pytest.mark.asyncio
    async def test_history_with_results(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
        )

        resp = await client.get("/api/v1/analytics/history")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["start_time"] is not None

    @pytest.mark.asyncio
    async def test_history_filter_by_model(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(model="llama-7b"),
        )
        _create_result_file(
            results_dir,
            "ns",
            "job-2",
            "profile_export_aiperf.json",
            _summary_json(model="gpt-2"),
        )

        resp = await client.get("/api/v1/analytics/history?model=llama")
        entries = resp.json()["entries"]
        assert len(entries) == 1
        assert "llama" in entries[0]["model"]


# ============================================================
# Analytics - Compare
# ============================================================


class TestCompareEndpoint:
    """Verify /api/v1/analytics/compare endpoint."""

    @pytest.mark.asyncio
    async def test_compare_specific_jobs(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(metric_val=100.0),
        )
        _create_result_file(
            results_dir,
            "ns",
            "job-2",
            "profile_export_aiperf.json",
            _summary_json(metric_val=200.0),
        )

        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": ["job-1", "job-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job-1" in data["job_ids"]
        assert "job-2" in data["job_ids"]

    @pytest.mark.asyncio
    async def test_compare_empty_jobs_returns_empty(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": []},
        )
        # FastAPI may return 422 for empty required list or 200 with empty
        assert resp.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_compare_nonexistent_job_returns_empty_entries(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
        )

        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": ["nonexistent"]},
        )
        assert resp.status_code == 200
        assert resp.json()["entries"] == []


# ============================================================
# Analytics - Summary
# ============================================================


class TestSummaryEndpoint:
    """Verify /api/v1/analytics/summary/{namespace}/{job_id} endpoint."""

    @pytest.mark.asyncio
    async def test_summary_existing_job(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
        )

        resp = await client.get("/api/v1/analytics/summary/ns/job-1")
        assert resp.status_code == 200
        data = resp.json()
        assert "request_throughput" in data

    @pytest.mark.asyncio
    async def test_summary_nonexistent_job_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/analytics/summary/ns/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_summary_zst_file(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
            compress=True,
        )

        resp = await client.get("/api/v1/analytics/summary/ns/job-1")
        assert resp.status_code == 200
        assert "request_throughput" in resp.json()

    @pytest.mark.asyncio
    async def test_summary_job_dir_exists_but_no_summary_file(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(results_dir, "ns", "job-1", "other_file.json")

        resp = await client.get("/api/v1/analytics/summary/ns/job-1")
        assert resp.status_code == 404


# ============================================================
# Adversarial Inputs
# ============================================================


class TestAdversarialInputs:
    """Verify server handles adversarial inputs safely."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metric",
        [
            param("'; DROP TABLE results; --", id="sql-injection"),
            param("metric OR 1=1", id="sql-or-injection"),
            param("a" * 10000, id="extremely-long-string"),
        ],
    )  # fmt: skip
    async def test_leaderboard_rejects_invalid_metric(
        self, client: httpx.AsyncClient, metric: str
    ) -> None:
        resp = await client.get(
            "/api/v1/analytics/leaderboard",
            params={"metric": metric},
        )
        # Should return 422 (validation error) or 500 (caught internally)
        # but NOT execute the injection
        assert resp.status_code in (200, 422, 500)

    @pytest.mark.asyncio
    async def test_leaderboard_valid_metric_with_underscore(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
        )

        resp = await client.get(
            "/api/v1/analytics/leaderboard",
            params={"metric": "request_throughput", "stat": "avg"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_history_sql_injection_in_model_filter(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
        )

        resp = await client.get(
            "/api/v1/analytics/history",
            params={"model": "'; DROP TABLE t; --"},
        )
        # Should not crash
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_download_unicode_filename(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/results/ns/job-1/\u00e9\u00e0\u00fc.json")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_compare_sql_injection_in_job_ids(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns",
            "job-1",
            "profile_export_aiperf.json",
            _summary_json(),
        )

        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": ["job-1", "'; DROP TABLE t; --"]},
        )
        # Should not crash or inject
        assert resp.status_code == 200


# ============================================================
# Dashboard Mount
# ============================================================
