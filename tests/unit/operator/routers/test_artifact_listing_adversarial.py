# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator artifact listing metadata.

Focuses on:
- file-list metadata correctness for size, mtime, stored name, and compression state
- internal ready-marker exclusion from list responses
- checkpoint visibility before final artifacts are ready
- deterministic sorting, directory handling, symlink exclusion, and URL-encoded identifiers
- empty and malformed run directories at the artifact-listing trust boundary

Out of scope: download header/media-type behavior, covered by sibling artifact-header tests.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path, PureWindowsPath
from urllib.parse import quote

import httpx
import orjson
import pytest
from aiperf.operator.results_layout import run_dir, write_latest
from aiperf.operator.routers import results_files_io
from aiperf.operator.routers.results_files import create_results_files_router
from fastapi import FastAPI

# ============================================================
# Helpers
# ============================================================

_EPOCH = "1714150923"
_READY_MARKER = ".aiperf_results_ready.json"


def test_artifact_display_name_uses_posix_separators_for_windows_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep API-visible artifact paths portable across host platforms."""
    monkeypatch.setattr(results_files_io, "Path", PureWindowsPath)

    assert (
        results_files_io._artifact_display_name("sweep_aggregate/sweep_results.csv")
        == "sweep_aggregate/sweep_results.csv"
    )


def _app_for_results_dir(results_dir: Path) -> FastAPI:
    """Build only the results-files router so tests avoid k8s lifespan side effects."""
    app = FastAPI()
    app.include_router(create_results_files_router(results_dir))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by a temp PVC-like results directory."""
    transport = httpx.ASGITransport(
        app=_app_for_results_dir(tmp_path), raise_app_exceptions=False
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.operator.local"
    ) as c:
        yield c


def _seed_run(
    base_dir: Path,
    *,
    namespace: str = "bench-prod",
    job_id: str = "llama-3-8b-load",
    epoch: str = _EPOCH,
    files: dict[str, bytes] | None = None,
    ready: bool = True,
) -> Path:
    """Create one epoch-pinned run directory with realistic artifact files."""
    target = run_dir(base_dir, namespace, job_id, epoch)
    target.mkdir(parents=True, exist_ok=True)
    seed_files = {"profile_export_aiperf.json": b"{}"} if files is None else files
    for name, payload in seed_files.items():
        file_path = target / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(payload)
    if ready:
        (target / _READY_MARKER).write_bytes(orjson.dumps({"ready": True}))
    write_latest(base_dir, namespace, job_id, epoch)
    return target


def _artifact_list_url(
    *,
    namespace: str = "bench-prod",
    job_id: str = "llama-3-8b-load",
    epoch: str = _EPOCH,
) -> str:
    """Return the epoch-pinned file-listing URL for a stored benchmark run."""
    ns_seg = quote(namespace, safe="")
    job_seg = quote(job_id, safe="")
    epoch_seg = quote(epoch, safe="")
    return f"/api/v1/results/{ns_seg}/{job_seg}/runs/{epoch_seg}"


def _entry_by_name(files: list[dict[str, object]], name: str) -> dict[str, object]:
    """Return one file-list entry by display name."""
    return next(entry for entry in files if entry["name"] == name)


def _set_mtime(path: Path, epoch: int) -> None:
    """Set atime and mtime to an exact epoch-second boundary."""
    os.utime(path, (epoch, epoch))


# ============================================================
# Metadata correctness
# ============================================================


class TestArtifactListingMetadata:
    """List responses should describe the bytes the API will serve."""

    @pytest.mark.asyncio
    async def test_list_historical_files_regular_file_reports_size_and_mtime_epoch(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        payload = b'{"request_throughput":{"avg":42.5}}'
        mtime_epoch = 1714151999
        run = _seed_run(tmp_path, files={"profile_export_aiperf.json": payload})
        _set_mtime(run / "profile_export_aiperf.json", mtime_epoch)

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        entry = _entry_by_name(response.json()["files"], "profile_export_aiperf.json")
        assert entry["stored_name"] == "profile_export_aiperf.json"
        assert entry["size_bytes"] == len(payload)
        assert entry["compressed"] is False
        assert entry["mtime_epoch"] == mtime_epoch

    @pytest.mark.asyncio
    async def test_list_historical_files_zstd_file_reports_stored_size_and_mtime_epoch(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        stored_payload = b"zstd-frame-bytes-for-profile-export"
        mtime_epoch = 1714152007
        run = _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json.zst": stored_payload},
        )
        _set_mtime(run / "profile_export_aiperf.json.zst", mtime_epoch)

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        entry = _entry_by_name(response.json()["files"], "profile_export_aiperf.json")
        assert entry["stored_name"] == "profile_export_aiperf.json.zst"
        assert entry["size_bytes"] == len(stored_payload)
        assert entry["compressed"] is True
        assert entry["mtime_epoch"] == mtime_epoch

    @pytest.mark.asyncio
    async def test_list_historical_files_excludes_internal_ready_marker(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": b'{"status":"Succeeded"}',
                "metrics.json": b'{"ttft_ms":{"p50":12}}',
            },
        )

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        names = {entry["name"] for entry in response.json()["files"]}
        assert _READY_MARKER not in names
        assert names == {"metrics.json", "profile_export_aiperf.json"}


# ============================================================
# Ready marker and checkpoint gates
# ============================================================


class TestArtifactListingReadyGate:
    """Final artifacts are gated by the marker; checkpoints stay inspectable."""

    @pytest.mark.asyncio
    async def test_list_historical_files_not_ready_lists_checkpoint_metadata_only(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        checkpoint_payload = b"PAR1checkpoint-record-batch"
        final_payload = b'{"partial":true}'
        mtime_epoch = 1714152011
        run = _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": final_payload,
                "metrics.json": b'{"hidden":"until-ready"}',
                "checkpoints/records-manager-0001.parquet": checkpoint_payload,
            },
            ready=False,
        )
        _set_mtime(run / "checkpoints" / "records-manager-0001.parquet", mtime_epoch)

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        files = response.json()["files"]
        assert [entry["name"] for entry in files] == [
            "checkpoints/records-manager-0001.parquet"
        ]
        assert files[0]["size_bytes"] == len(checkpoint_payload)
        assert files[0]["mtime_epoch"] == mtime_epoch

    @pytest.mark.asyncio
    async def test_list_historical_files_ready_lists_final_and_checkpoint_artifacts(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": b'{"status":"Succeeded"}',
                "checkpoints/records-manager-0001.parquet": b"PAR1checkpoint",
            },
            ready=True,
        )

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        assert [entry["name"] for entry in response.json()["files"]] == [
            "checkpoints/records-manager-0001.parquet",
            "profile_export_aiperf.json",
        ]


# ============================================================
# Directory, symlink, and sort stability
# ============================================================


class TestArtifactListingFilesystemEdges:
    """Directory-shaped and link-shaped artifacts should not leak into listings."""

    @pytest.mark.asyncio
    async def test_list_historical_files_ignores_non_checkpoint_directories(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        run = _seed_run(tmp_path, files={"profile_export_aiperf.json": b"{}"})
        (run / "nested-diagnostics" / "raw-events.jsonl").parent.mkdir()
        (run / "nested-diagnostics" / "raw-events.jsonl").write_bytes(b'{"idx":1}\n')
        (run / "checkpoints" / "empty-dir").mkdir(parents=True)

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        names = {entry["name"] for entry in response.json()["files"]}
        assert "nested-diagnostics/raw-events.jsonl" not in names
        assert "checkpoints/empty-dir" not in names
        assert names == {"profile_export_aiperf.json"}

    @pytest.mark.asyncio
    async def test_list_historical_files_skips_file_and_directory_symlinks(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        run = _seed_run(tmp_path, files={"profile_export_aiperf.json": b"{}"})
        external_file = tmp_path / "external-summary.json"
        external_file.write_bytes(b'{"cluster_token":"secret"}')
        external_dir = tmp_path / "external-checkpoints"
        external_dir.mkdir()
        (external_dir / "records-manager-0002.parquet").write_bytes(b"PAR1external")
        (run / "external-summary.json").symlink_to(external_file)
        (run / "checkpoints").mkdir()
        (run / "checkpoints" / "external-dir").symlink_to(
            external_dir, target_is_directory=True
        )

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        names = {entry["name"] for entry in response.json()["files"]}
        assert "external-summary.json" not in names
        assert "checkpoints/external-dir/records-manager-0002.parquet" not in names
        assert names == {"profile_export_aiperf.json"}

    @pytest.mark.asyncio
    async def test_list_historical_files_sorts_by_display_name_then_stored_name(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json.zst": b"compressed-profile-export",
                "profile_export_aiperf.json": b'{"status":"Succeeded"}',
                "metrics.json": b'{"request_throughput":{"avg":77}}',
            },
        )

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        assert [entry["stored_name"] for entry in response.json()["files"]] == [
            "metrics.json",
            "profile_export_aiperf.json",
            "profile_export_aiperf.json.zst",
        ]


# ============================================================
# URL encoding and malformed directories
# ============================================================


class TestArtifactListingUrlAndMalformedRuns:
    """Encoded identifiers and malformed run dirs stay inside the API contract."""

    @pytest.mark.asyncio
    async def test_list_historical_files_url_encoded_namespace_and_job_preserve_names(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        # Names are valid Kubernetes identifiers (DNS-1123); the dot in the job
        # name still exercises percent-decoding while passing path validation.
        _seed_run(
            tmp_path,
            namespace="bench-prod",
            job_id="llama-3.1-8b-load",
            files={"profile_export_aiperf.json": b'{"status":"Succeeded"}'},
        )

        response = await client.get(
            _artifact_list_url(namespace="bench-prod", job_id="llama-3.1-8b-load")
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["namespace"] == "bench-prod"
        assert body["job_id"] == "llama-3.1-8b-load"
        assert [entry["name"] for entry in body["files"]] == [
            "profile_export_aiperf.json"
        ]

    @pytest.mark.asyncio
    async def test_list_historical_files_empty_ready_run_returns_empty_file_list(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(tmp_path, files={}, ready=True)

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        assert response.json()["files"] == []

    @pytest.mark.asyncio
    async def test_list_historical_files_malformed_checkpoints_file_is_ignored(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        run = _seed_run(tmp_path, files={"profile_export_aiperf.json": b"{}"})
        (run / "checkpoints").write_bytes(b"not-a-directory")

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        assert [entry["name"] for entry in response.json()["files"]] == [
            "profile_export_aiperf.json"
        ]

    @pytest.mark.asyncio
    async def test_list_historical_files_missing_run_returns_contextual_404(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path, job_id="healthy-load", files={"profile_export_aiperf.json": b"{}"}
        )

        response = await client.get(
            _artifact_list_url(job_id="missing-load", epoch="1714237323")
        )

        assert response.status_code == 404
        assert (
            response.json()["detail"]
            == "No results for bench-prod/missing-load/runs/1714237323"
        )
