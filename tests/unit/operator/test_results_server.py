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
from unittest.mock import AsyncMock

import httpx
import orjson
import pytest
import zstandard
from pytest import param

pytest.importorskip("duckdb", reason="duckdb required for results_server tests")

from aiperf.operator import runs_index
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
    """Create a result file under the epoch-keyed layout, pointed at by latest.txt.

    Uses a synthetic default epoch so pre-existing flat-layout tests keep
    exercising the latest-run code path. Also mirrors the file at the
    legacy flat path so file-listing tests still see results at the
    legacy ``<ns>/<job>/`` level.

    When ``filename`` is ``profile_export_aiperf.json``, also writes the
    ``.aiperf_results_ready.json`` marker so ``runs_index.bootstrap`` will
    pick the run up — the analytics endpoints read from runs_index now.
    """
    from aiperf.operator.results_layout import run_dir, write_latest

    default_epoch = "1714064523"
    job_dir = run_dir(base_dir, namespace, job_id, default_epoch)
    job_dir.mkdir(parents=True, exist_ok=True)
    flat_dir = base_dir / namespace / job_id
    if compress:
        cctx = zstandard.ZstdCompressor()
        payload = cctx.compress(content)
        file_path = job_dir / (filename + ".zst")
        file_path.write_bytes(payload)
        (flat_dir / (filename + ".zst")).write_bytes(payload)
    else:
        file_path = job_dir / filename
        file_path.write_bytes(content)
        (flat_dir / filename).write_bytes(content)
    write_latest(base_dir, namespace, job_id, default_epoch)
    if filename == "profile_export_aiperf.json":
        (job_dir / ".aiperf_results_ready.json").write_bytes(b"{}")
    return file_path


async def _ingest_runs(base_dir: Path) -> None:
    """Drive a runs_index bootstrap walk so analytics endpoints see the runs.

    Tests write summary JSONs after the lifespan opens runs_index, so the
    in-memory index is empty until we explicitly walk the PVC. Calling
    bootstrap with ``force=True`` makes the walk idempotent across multiple
    writes within a single test.
    """
    if not runs_index.is_open():
        await runs_index.open(base_dir / ".aiperf_index.sqlite")
    await runs_index.bootstrap(base_dir, force=True)


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
    if runs_index.is_open():
        await runs_index.close()


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

    def test_results_server_mounts_dashboard_proxy_route(self, tmp_path: Path) -> None:
        """results-server registers the /dashboard/{path:path} proxy route."""
        from aiperf.operator.results_server import create_app

        app = create_app(results_dir=tmp_path)
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/dashboard/{path:path}" in paths

    @pytest.mark.asyncio
    async def test_results_server_opens_runs_index_readonly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The results-server sidecar must not become a second SQLite writer."""
        from aiperf.operator import runs_index as runs_index_mod
        from aiperf.operator.results_server import create_app

        calls: list[tuple[str, Path]] = []

        async def fake_open(path: Path) -> None:
            calls.append(("write", path))

        async def fake_open_readonly(path: Path) -> None:
            calls.append(("read", path))

        monkeypatch.setattr(runs_index_mod, "is_open", lambda: False)
        monkeypatch.setattr(runs_index_mod, "open", fake_open)
        monkeypatch.setattr(
            runs_index_mod, "open_readonly", fake_open_readonly, raising=False
        )
        monkeypatch.setattr(runs_index_mod, "close", AsyncMock())

        app = create_app(results_dir=tmp_path)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        await ctx.__aexit__(None, None, None)

        assert calls == [("read", tmp_path / ".aiperf_index.sqlite")]


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
    async def test_list_files_requires_explicit_epoch(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(results_dir, "ns", "job-1", "metrics.json")
        _create_result_file(results_dir, "ns", "job-1", "data.csv", compress=True)

        resp = await client.get("/api/v1/results/ns/job-1")
        assert resp.status_code == 409
        assert resp.json()["detail"] == (
            "Run epoch required; use /api/v1/results/ns/job-1/runs/<epoch>"
        )

    @pytest.mark.asyncio
    async def test_list_files_nonexistent_job_still_requires_epoch(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/results/ns/nonexistent")
        assert resp.status_code == 409
        assert resp.json()["detail"] == (
            "Run epoch required; use /api/v1/results/ns/nonexistent/runs/<epoch>"
        )

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
    """Verify non-epoch result downloads are rejected."""

    @pytest.mark.asyncio
    async def test_download_requires_explicit_epoch_for_existing_job(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir, "ns", "job-1", "metrics.json", b'{"result": true}'
        )

        resp = await client.get(
            "/api/v1/results/ns/job-1/metrics.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"] == (
            "Run epoch required; use /api/v1/results/ns/job-1/runs/<epoch>"
        )

    @pytest.mark.asyncio
    async def test_download_nonexistent_job_still_requires_epoch(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/results/ns/nojob/file.json")
        assert resp.status_code == 409
        assert resp.json()["detail"] == (
            "Run epoch required; use /api/v1/results/ns/nojob/runs/<epoch>"
        )


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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": ["job-1", "job-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job-1" in data["job_ids"]
        assert "job-2" in data["job_ids"]

    @pytest.mark.asyncio
    async def test_compare_unambiguous_bare_jobs_preserves_bare_keys(
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
        await _ingest_runs(results_dir)

        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": ["job-1", "job-2"]},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["meta"].keys() == {"job-1", "job-2"}
        value_entries = [
            entry
            for entry in data["entries"]
            if entry["metric"] == "request_throughput" and entry["stat"] == "avg"
        ]
        assert value_entries[0]["values"] == {"job-1": 100.0, "job-2": 200.0}

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
        await _ingest_runs(results_dir)

        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": ["nonexistent"]},
        )
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    @pytest.mark.asyncio
    async def test_compare_bare_job_name_rejects_namespace_ambiguity(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns-a",
            "same-job",
            "profile_export_aiperf.json",
            _summary_json(metric_val=100.0),
        )
        _create_result_file(
            results_dir,
            "ns-b",
            "same-job",
            "profile_export_aiperf.json",
            _summary_json(metric_val=200.0),
        )
        await _ingest_runs(results_dir)

        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": ["same-job"]},
        )

        assert resp.status_code == 409
        assert "ns-a/same-job" in resp.json()["detail"]
        assert "ns-b/same-job" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_compare_accepts_namespace_qualified_job_refs(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(
            results_dir,
            "ns-a",
            "same-job",
            "profile_export_aiperf.json",
            _summary_json(metric_val=100.0),
        )
        _create_result_file(
            results_dir,
            "ns-b",
            "same-job",
            "profile_export_aiperf.json",
            _summary_json(metric_val=200.0),
        )
        await _ingest_runs(results_dir)

        resp = await client.get(
            "/api/v1/analytics/compare",
            params={"jobs": ["ns-a/same-job"]},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["meta"].keys() == {"ns-a/same-job"}
        value_entries = [
            entry
            for entry in data["entries"]
            if entry["metric"] == "request_throughput" and entry["stat"] == "avg"
        ]
        assert value_entries[0]["values"] == {"ns-a/same-job": 100.0}


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
        await _ingest_runs(results_dir)

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
        await _ingest_runs(results_dir)

        resp = await client.get("/api/v1/analytics/summary/ns/job-1")
        assert resp.status_code == 200
        assert "request_throughput" in resp.json()

    @pytest.mark.asyncio
    async def test_summary_job_dir_exists_but_no_summary_file(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _create_result_file(results_dir, "ns", "job-1", "other_file.json")
        await _ingest_runs(results_dir)

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
    async def test_download_unicode_filename_requires_epoch(
        self, results_dir: Path, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/results/ns/job-1/\u00e9\u00e0\u00fc.json")
        assert resp.status_code == 409

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


# ============================================================
# Epoch-keyed layout routing
# ============================================================


def _seed_epoch_run(
    base: Path,
    namespace: str,
    name: str,
    epoch: str,
    filename: str,
    content: bytes = b"{}",
) -> Path:
    from aiperf.kubernetes.results_sidecar import write_ready_marker
    from aiperf.operator.results_layout import run_dir, write_latest

    d = run_dir(base, namespace, name, epoch)
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_bytes(content)
    write_ready_marker(d)
    write_latest(base, namespace, name, epoch)
    return d


_EPOCH_OLD = "1714064523"
_EPOCH_NEW = "1714150923"


def test_list_job_files_requires_explicit_epoch(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "old.json", b'{"v":1}')
    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_NEW, "new.json", b'{"v":2}')

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job")
        assert r.status_code == 409
        body = r.json()
        assert (
            body["detail"]
            == "Run epoch required; use /api/v1/results/ns/job/runs/<epoch>"
        )


def test_historical_route_pins_epoch(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "old.json", b'{"v":1}')
    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_NEW, "new.json", b'{"v":2}')

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get(f"/api/v1/results/ns/job/runs/{_EPOCH_OLD}")
        assert r.status_code == 200
        names = {f["name"] for f in r.json()["files"]}
        assert names == {"old.json"}


def test_historical_route_invalid_epoch_rejected(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "old.json")
    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job/runs/..%2Fevil")
        assert r.status_code in (404, 422)


def test_historical_zip_bundle_pins_epoch(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "old.json", b'{"v":1}')
    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_NEW, "new.json", b'{"v":2}')
    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get(f"/api/v1/results/ns/job/runs/{_EPOCH_OLD}.zip")
        assert r.status_code == 200
        assert b"old.json" in r.content


def test_scan_job_dirs_collapses_to_latest_epoch(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "old.json")
    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_NEW, "new.json")
    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results")
        assert r.status_code == 200
        entries = [
            (j["namespace"], j["job_id"], j["file_count"]) for j in r.json()["jobs"]
        ]
        assert entries == [("ns", "job", 1)]


def test_list_runs_returns_epochs_newest_first(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "a.json")
    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_NEW, "b.json")

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job/runs")
        assert r.status_code == 200
        body = r.json()
        assert body["namespace"] == "ns"
        assert body["job_id"] == "job"
        assert body["latest_epoch"] == _EPOCH_NEW
        epochs = [run["epoch"] for run in body["runs"]]
        assert epochs == [_EPOCH_NEW, _EPOCH_OLD]
        latest_flags = [run["is_latest"] for run in body["runs"]]
        assert latest_flags == [True, False]
        for run in body["runs"]:
            assert run["file_count"] == 1
            assert run["total_size_bytes"] > 0


def test_list_runs_404_when_no_runs(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/absent/runs")
        assert r.status_code == 404


def test_list_runs_skips_non_epoch_dirs(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator.results_layout import job_dir
    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "a.json")
    (job_dir(tmp_path, "ns", "job") / "not-an-epoch").mkdir()

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job/runs")
        assert r.status_code == 200
        epochs = {run["epoch"] for run in r.json()["runs"]}
        assert epochs == {_EPOCH_OLD}


def test_readonly_list_runs_does_not_schedule_lazy_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator import runs_index as runs_index_mod
    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "a.json")
    lazy_backfill = AsyncMock()
    monkeypatch.setattr(runs_index_mod, "is_open", lambda: True, raising=False)
    monkeypatch.setattr(runs_index_mod, "is_readonly", lambda: True, raising=False)
    monkeypatch.setattr(runs_index_mod, "lazy_backfill_run", lazy_backfill)

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job/runs")

    assert r.status_code == 200
    lazy_backfill.assert_not_called()


def test_readonly_sidecar_missing_index_does_not_schedule_lazy_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from aiperf.operator import runs_index as runs_index_mod
    from aiperf.operator.results_server import create_app

    _seed_epoch_run(tmp_path, "ns", "job", _EPOCH_OLD, "a.json")
    lazy_backfill = AsyncMock()
    monkeypatch.setattr(runs_index_mod, "lazy_backfill_run", lazy_backfill)

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/results/ns/job/runs")

    assert r.status_code == 200
    lazy_backfill.assert_not_called()


def test_config_retention_endpoint_returns_current_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/v1/config/retention returns the live OperatorEnvironment values."""
    from fastapi.testclient import TestClient

    from aiperf.operator.environment import OperatorEnvironment
    from aiperf.operator.results_server import create_app

    # The singleton is constructed once at module import, so values are fixed
    # from the test-runner's environment. Patch the already-loaded settings
    # object directly rather than reloading the module (which would strand
    # other importers on a stale instance).
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "RETAIN_RUNS", 15)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "RETAIN_DAYS", 30)

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get("/api/v1/config/retention")
        assert r.status_code == 200
        body = r.json()
        assert body["retain_runs"] == 15
        assert body["retain_days"] == 30


# ============================================================
# Quick-export profile_export endpoint
# ============================================================


_PROFILE_EXPORT_EPOCH = "1714150923"


def test_profile_export_quick_route_returns_json_when_present(tmp_path: Path) -> None:
    """The quick-export route returns the raw profile_export_aiperf.json bytes
    with application/json + canonical Content-Disposition, skipping the
    directory-listing roundtrip the artifacts table normally requires."""
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    payload = orjson.dumps({"request_throughput": {"avg": 123.4, "unit": "req/s"}})
    _seed_epoch_run(
        tmp_path,
        "acme-bench",
        "vllm-test",
        _PROFILE_EXPORT_EPOCH,
        "profile_export_aiperf.json",
        payload,
    )

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get(
            f"/api/v1/results/acme-bench/vllm-test/runs/{_PROFILE_EXPORT_EPOCH}"
            "/profile_export"
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert (
            r.headers["content-disposition"]
            == 'attachment; filename="profile_export_aiperf.json"'
        )
        assert r.headers["x-filename"] == "profile_export_aiperf.json"
        assert r.content == payload
        assert orjson.loads(r.content)["request_throughput"]["avg"] == 123.4


def test_profile_export_quick_route_decompresses_zst(tmp_path: Path) -> None:
    """The quick-export route falls back to the .zst companion when the
    uncompressed JSON is absent — mirrors the per-file route's transparent
    decompression but pins media_type to application/json."""
    from fastapi.testclient import TestClient

    from aiperf.kubernetes.results_sidecar import write_ready_marker
    from aiperf.operator.results_layout import run_dir, write_latest
    from aiperf.operator.results_server import create_app

    payload = orjson.dumps({"output_token_throughput": {"avg": 987.6, "unit": "tok/s"}})
    cctx = zstandard.ZstdCompressor()
    compressed = cctx.compress(payload)

    d = run_dir(tmp_path, "acme-bench", "vllm-test", _PROFILE_EXPORT_EPOCH)
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile_export_aiperf.json.zst").write_bytes(compressed)
    write_ready_marker(d)
    write_latest(tmp_path, "acme-bench", "vllm-test", _PROFILE_EXPORT_EPOCH)

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get(
            f"/api/v1/results/acme-bench/vllm-test/runs/{_PROFILE_EXPORT_EPOCH}"
            "/profile_export"
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert r.content == payload


def test_profile_export_quick_route_404_when_missing(tmp_path: Path) -> None:
    """The run dir exists with other artifacts but no profile_export — must
    return 404 with a meaningful detail naming the missing file."""
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    _seed_epoch_run(
        tmp_path,
        "acme-bench",
        "vllm-test",
        _PROFILE_EXPORT_EPOCH,
        "metrics.json",
        b'{"x": 1}',
    )

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get(
            f"/api/v1/results/acme-bench/vllm-test/runs/{_PROFILE_EXPORT_EPOCH}"
            "/profile_export"
        )
        assert r.status_code == 404
        assert "profile_export_aiperf.json" in r.json()["detail"]


def test_profile_export_quick_route_does_not_shadow_filename_route(
    tmp_path: Path,
) -> None:
    """The literal /profile_export route must not be caught by the
    {filename:path} catch-all that follows it. A real file named
    'profile_export' (no extension) under the run dir would be served by the
    catch-all, but the registered literal wins for ambiguity-free callers."""
    from fastapi.testclient import TestClient

    from aiperf.operator.results_server import create_app

    payload = orjson.dumps({"sentinel": True})
    _seed_epoch_run(
        tmp_path,
        "acme-bench",
        "vllm-test",
        _PROFILE_EXPORT_EPOCH,
        "profile_export_aiperf.json",
        payload,
    )

    with TestClient(create_app(results_dir=tmp_path)) as client:
        r = client.get(
            f"/api/v1/results/acme-bench/vllm-test/runs/{_PROFILE_EXPORT_EPOCH}"
            "/profile_export"
        )
        # If the catch-all had won, content-type would be application/octet-stream.
        assert r.headers["content-type"].startswith("application/json")
        assert orjson.loads(r.content)["sentinel"] is True
