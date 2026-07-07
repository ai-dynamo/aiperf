# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the operator results-files router.

Focuses on:
- path traversal and symlink escape attempts at the filename trust boundary
- results-ready marker gating for final top-level artifacts
- checkpoint bypass semantics while a run is still processing
- stale or missing runs-index fallback to the PVC artifact tree
- malformed artifact trees and URL-encoded namespace/job identifiers

Out of scope: analytics SQL behavior, covered by sibling operator results tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import orjson
import pytest
from fastapi import FastAPI
from pytest import param

from aiperf.operator.results_layout import run_dir, write_latest
from aiperf.operator.routers.results_files import create_results_files_router
from aiperf.operator.routers.results_files_io import list_job_files_with_readiness

# ============================================================
# Helpers
# ============================================================

_EPOCH_OLD = "1714064523"
_EPOCH_NEW = "1714150923"
_READY_MARKER = ".aiperf_results_ready.json"


def _app_for_results_dir(results_dir: Path) -> FastAPI:
    """Build only the results-files router so tests avoid k8s lifespan side effects."""
    app = FastAPI()
    app.include_router(create_results_files_router(results_dir))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by a temp PVC-like results directory."""
    transport = httpx.ASGITransport(app=_app_for_results_dir(tmp_path))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.local"
    ) as c:
        yield c


def _seed_run(
    base_dir: Path,
    *,
    namespace: str = "bench-prod",
    job_id: str = "llama-3-8b-load",
    epoch: str = _EPOCH_NEW,
    files: dict[str, bytes] | None = None,
    ready: bool = True,
) -> Path:
    """Create one run directory and point latest.txt at it."""
    target = run_dir(base_dir, namespace, job_id, epoch)
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in (files or {"profile_export_aiperf.json": b"{}"}).items():
        file_path = target / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(payload)
    if ready:
        (target / _READY_MARKER).write_bytes(orjson.dumps({"ready": True}))
    write_latest(base_dir, namespace, job_id, epoch)
    return target


# ============================================================
# Path traversal / trust boundary
# ============================================================


class TestResultsFilesPathTraversal:
    """Filename and route parameters must not escape the run directory."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "encoded_filename",
        [
            param("..%2F..%2Fsecret-token.txt", id="encoded-parent-segments"),
            param("%2E%2E/%2E%2E/secret-token.txt", id="encoded-dot-segments"),
            param("checkpoints%2F..%2F..%2Fsecret-token.txt", id="checkpoint-prefix-smuggling"),
        ],
    )  # fmt: skip
    async def test_download_historical_file_path_traversal_returns_404_not_secret(
        self, tmp_path: Path, client: httpx.AsyncClient, encoded_filename: str
    ) -> None:
        _seed_run(tmp_path, files={"profile_export_aiperf.json": b'{"ok": true}'})
        (tmp_path / "secret-token.txt").write_text("operator-service-account-token")

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH_NEW}/"
            f"{encoded_filename}",
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 404
        assert b"operator-service-account-token" not in response.content
        assert "secret-token.txt" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_download_historical_file_symlink_escape_returns_404(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        run = _seed_run(tmp_path, files={"profile_export_aiperf.json": b"{}"})
        outside_secret = tmp_path / "external-token.txt"
        outside_secret.write_text("cluster-admin-token")
        (run / "token-link.txt").symlink_to(outside_secret)

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH_NEW}"
            "/token-link.txt",
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 404
        assert b"cluster-admin-token" not in response.content


# ============================================================
# Ready marker / checkpoint bypass
# ============================================================


class TestResultsFilesNamespaceJobTraversal:
    """namespace/job_id path parameters must not escape the results base dir."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "namespace",
        [
            param("%2E%2E", id="encoded-dot-namespace"),
            param("bench_prod", id="underscore-illegal-namespace"),
            param("Bench-Prod", id="uppercase-illegal-namespace"),
            param("team.alpha", id="dotted-namespace-not-a-label"),
        ],
    )  # fmt: skip
    async def test_list_historical_files_bad_namespace_returns_400_not_secret(
        self, tmp_path: Path, client: httpx.AsyncClient, namespace: str
    ) -> None:
        _seed_run(tmp_path, files={"profile_export_aiperf.json": b'{"ok": true}'})
        (tmp_path / "secret-token.txt").write_text("operator-service-account-token")

        response = await client.get(
            f"/api/v1/results/{namespace}/llama-3-8b-load/runs/{_EPOCH_NEW}"
        )

        assert response.status_code == 400
        assert b"operator-service-account-token" not in response.content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "job_id",
        [
            param("bad_job", id="underscore-illegal-job"),
            param("Bad-Job", id="uppercase-illegal-job"),
        ],
    )  # fmt: skip
    async def test_download_historical_file_bad_job_returns_400_not_secret(
        self, tmp_path: Path, client: httpx.AsyncClient, job_id: str
    ) -> None:
        _seed_run(tmp_path, files={"profile_export_aiperf.json": b'{"ok": true}'})
        (tmp_path / "secret-token.txt").write_text("operator-service-account-token")

        response = await client.get(
            f"/api/v1/results/bench-prod/{job_id}/runs/{_EPOCH_NEW}"
            "/profile_export_aiperf.json"
        )

        assert response.status_code == 400
        assert b"operator-service-account-token" not in response.content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "namespace,job_id",
        [
            param("..%2F..", "llama-3-8b-load", id="encoded-parent-namespace"),
            param("bench-prod", "..%2F..", id="encoded-parent-job"),
        ],
    )  # fmt: skip
    async def test_encoded_slash_traversal_never_returns_secret(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        namespace: str,
        job_id: str,
    ) -> None:
        """Encoded-slash traversal is always rejected and never leaks a secret.

        Depending on how the encoded ``%2F`` re-splits the path it may be
        caught by the path validator (400), by route matching (404), or by the
        epoch-required guard (409) — the invariant is a >= 400 rejection that
        never returns the operator token file.
        """
        _seed_run(tmp_path, files={"profile_export_aiperf.json": b'{"ok": true}'})
        (tmp_path / "secret-token.txt").write_text("operator-service-account-token")

        response = await client.get(
            f"/api/v1/results/{namespace}/{job_id}/runs/{_EPOCH_NEW}"
        )

        assert response.status_code >= 400
        assert b"operator-service-account-token" not in response.content

    @pytest.mark.asyncio
    async def test_list_runs_raw_dotdot_namespace_returns_400(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        """A single raw ``..`` segment (no slash) must be rejected before joins."""
        response = await client.get("/api/v1/results/../llama-3-8b-load/runs")
        assert response.status_code in (400, 404)
        assert b"operator-service-account-token" not in response.content


class TestResultsFilesReadyMarkerGate:
    """Top-level final artifacts are hidden until ready; checkpoints bypass."""

    def test_list_job_files_with_readiness_hides_final_files_without_marker(
        self, tmp_path: Path
    ) -> None:
        run = _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": b'{"partial": true}',
                "metrics.json": b'{"should": "stay hidden"}',
            },
            ready=False,
        )

        files, ready = list_job_files_with_readiness(run)

        assert ready is False
        assert files == []

    @pytest.mark.asyncio
    async def test_list_historical_files_missing_ready_marker_lists_only_checkpoints(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": b'{"partial": true}',
                "metrics.json": b'{"should": "stay hidden"}',
                "checkpoints/records-manager-0001.parquet": b"PAR1",
            },
            ready=False,
        )

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH_NEW}"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is False
        names = {entry["name"] for entry in body["files"]}
        assert names == {"checkpoints/records-manager-0001.parquet"}

    @pytest.mark.asyncio
    async def test_download_historical_file_missing_ready_marker_rejects_top_level_export(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json": b'{"partial": true}'},
            ready=False,
        )

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH_NEW}"
            "/profile_export_aiperf.json",
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 404
        assert _READY_MARKER in response.json()["detail"]
        assert b'"partial": true' not in response.content

    @pytest.mark.asyncio
    async def test_download_historical_file_missing_ready_marker_allows_checkpoint(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": b'{"partial": true}',
                "checkpoints/records-manager-0001.parquet": b"PAR1checkpoint",
            },
            ready=False,
        )

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH_NEW}"
            "/checkpoints/records-manager-0001.parquet",
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 200
        assert response.content == b"PAR1checkpoint"
        assert response.headers["x-filename"] == "records-manager-0001.parquet"

    @pytest.mark.asyncio
    async def test_profile_export_quick_missing_ready_marker_returns_not_ready(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json": b'{"partial": true}'},
            ready=False,
        )

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH_NEW}"
            "/profile_export"
        )

        assert response.status_code == 404
        assert _READY_MARKER in response.json()["detail"]
        assert response.content != b'{"partial": true}'


# ============================================================
# Index fallback / malformed artifact trees
# ============================================================


class TestResultsFilesIndexAndMalformedTrees:
    """The router treats SQLite as a cache and disk as authoritative truth."""

    @pytest.mark.asyncio
    async def test_list_runs_stale_index_merges_disk_epoch_and_corrects_latest(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from aiperf.operator.runs_index_models import RunIndexRow

        _seed_run(tmp_path, epoch=_EPOCH_OLD, files={"old.json": b"{}"})
        _seed_run(tmp_path, epoch=_EPOCH_NEW, files={"new.json": b"{}"})

        async def fake_index_rows(namespace: str, job_id: str) -> list[RunIndexRow]:
            return [
                RunIndexRow(
                    namespace=namespace,
                    job_id=job_id,
                    epoch=_EPOCH_OLD,
                    phase="Succeeded",
                    is_latest=True,
                    start_time=None,
                    end_time=None,
                    created_unix=int(_EPOCH_OLD),
                    mtime_epoch=1,
                    error=None,
                    model=None,
                    endpoint=None,
                    gpu_count=0,
                    gpu_name=None,
                    file_count=1,
                    total_size_bytes=2,
                    sweep_namespace=None,
                    sweep_name=None,
                    sweep_epoch=None,
                    sweep_variation_idx=None,
                )
            ]

        monkeypatch.setattr(
            "aiperf.operator.runs_index.list_runs_for_job", fake_index_rows
        )

        response = await client.get("/api/v1/results/bench-prod/llama-3-8b-load/runs")

        assert response.status_code == 200
        body = response.json()
        assert {entry["epoch"] for entry in body["runs"]} == {_EPOCH_OLD, _EPOCH_NEW}
        assert body["latest_epoch"] == _EPOCH_NEW
        assert {entry["epoch"] for entry in body["runs"] if entry["is_latest"]} == {
            _EPOCH_NEW
        }

    @pytest.mark.asyncio
    async def test_list_jobs_missing_latest_pointer_skips_malformed_job_tree(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        malformed = run_dir(tmp_path, "bench-prod", "orphaned-load", _EPOCH_NEW)
        malformed.mkdir(parents=True)
        (malformed / "profile_export_aiperf.json").write_bytes(b"{}")
        _seed_run(tmp_path, job_id="healthy-load", files={"metrics.json": b"{}"})

        response = await client.get("/api/v1/results")

        assert response.status_code == 200
        jobs = {
            (entry["namespace"], entry["job_id"]) for entry in response.json()["jobs"]
        }
        assert jobs == {("bench-prod", "healthy-load")}

    @pytest.mark.asyncio
    async def test_list_historical_files_skips_symlinks_and_reports_regular_files(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        run = _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json": b"{}", "metrics.json": b"{}"},
        )
        external = tmp_path / "outside-metrics.json"
        external.write_bytes(b'{"leaked": true}')
        (run / "outside-metrics.json").symlink_to(external)

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH_NEW}"
        )

        assert response.status_code == 200
        names = {entry["name"] for entry in response.json()["files"]}
        assert "outside-metrics.json" not in names
        assert {"profile_export_aiperf.json", "metrics.json", _READY_MARKER} >= names


# ============================================================
# URL encoding
# ============================================================


class TestResultsFilesUrlEncoding:
    """Encoded path segments should identify the same on-disk namespace/job."""

    @pytest.mark.asyncio
    async def test_list_historical_files_url_encoded_namespace_and_job_returns_files(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        """Percent-encoded legal characters decode to the same on-disk names."""
        _seed_run(
            tmp_path,
            namespace="team-alpha",
            job_id="llama-3.1-8b-load",
            files={
                "profile_export_aiperf.json": b'{"request_throughput": {"avg": 42}}'
            },
        )

        response = await client.get(
            f"/api/v1/results/team%2Dalpha/llama-3%2E1-8b-load/runs/{_EPOCH_NEW}"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["namespace"] == "team-alpha"
        assert body["job_id"] == "llama-3.1-8b-load"
        assert {entry["name"] for entry in body["files"]} >= {
            "profile_export_aiperf.json"
        }

    @pytest.mark.asyncio
    async def test_download_historical_file_url_encoded_filename_returns_exact_file(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={"metrics+summary.json": b'{"request_throughput": {"avg": 99}}'},
        )

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH_NEW}"
            "/metrics%2Bsummary.json",
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 200
        assert orjson.loads(response.content)["request_throughput"]["avg"] == 99
        assert response.headers["x-filename"] == "metrics+summary.json"
