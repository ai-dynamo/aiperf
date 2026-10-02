# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator results zip bundles.

Focuses on:
- ready-marker gating for epoch-pinned bundle downloads
- internal marker and symlink exclusion at archive-construction boundaries
- stable archive filenames and deterministic member ordering
- binary payload preservation, including zstd-backed visible filenames
- empty ready run directories producing valid empty bundles
- checkpoint visibility parity with the file-listing route
- chunked bundle streaming for large archives

Out of scope: per-file content negotiation and run-index fallback, covered by sibling results-files tests.
"""

from __future__ import annotations

import zipfile
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path

import httpx
import orjson
import pytest
from fastapi import FastAPI

from aiperf.common.results_markers import CHECKPOINTS_DIR_NAME, READY_MARKER_NAME
from aiperf.operator.results_layout import run_dir, write_latest
from aiperf.operator.routers.results_files import create_results_files_router
from aiperf.operator.routers.results_files_io import CHUNK_SIZE, _stream_job_bundle

# ============================================================
# Helpers
# ============================================================

_EPOCH = "1714150923"
_NAMESPACE = "bench-prod"
_JOB_ID = "llama-3-8b-load"


def _app_for_results_dir(results_dir: Path) -> FastAPI:
    """Build the results-files router without full operator app side effects."""
    app = FastAPI()
    app.include_router(create_results_files_router(results_dir))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by a temp PVC-like results directory."""
    transport = httpx.ASGITransport(app=_app_for_results_dir(tmp_path))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.operator.local"
    ) as c:
        yield c


def _seed_run(
    base_dir: Path,
    *,
    namespace: str = _NAMESPACE,
    job_id: str = _JOB_ID,
    epoch: str = _EPOCH,
    files: dict[str, bytes] | None = None,
    ready: bool = True,
) -> Path:
    """Create one epoch-pinned results directory and update latest.txt."""
    target = run_dir(base_dir, namespace, job_id, epoch)
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in (files or {}).items():
        file_path = target / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(payload)
    if ready:
        (target / READY_MARKER_NAME).write_bytes(orjson.dumps({"ready": True}))
    write_latest(base_dir, namespace, job_id, epoch)
    return target


def _bundle_url(
    *, namespace: str = _NAMESPACE, job_id: str = _JOB_ID, epoch: str = _EPOCH
) -> str:
    """Return the epoch-pinned bundle URL for one run."""
    return f"/api/v1/results/{namespace}/{job_id}/runs/{epoch}.zip"


def _zip_entries(payload: bytes) -> dict[str, bytes]:
    """Return zip member payloads keyed by archive name."""
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _zip_names(payload: bytes) -> list[str]:
    """Return archive member names in stored order."""
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        return zf.namelist()


# ============================================================
# Ready marker gate / empty bundles
# ============================================================


class TestZipBundleReadyMarkerGate:
    """Bundles are final artifacts and require the results-ready marker."""

    @pytest.mark.asyncio
    async def test_download_historical_bundle_missing_ready_marker_returns_not_ready(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json": b'{"partial": true}'},
            ready=False,
        )

        response = await client.get(
            _bundle_url(), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 404
        assert READY_MARKER_NAME in response.json()["detail"]
        assert not response.content.startswith(b"PK")

    @pytest.mark.asyncio
    async def test_download_historical_bundle_ready_empty_run_returns_empty_zip(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(tmp_path, files={}, ready=True)

        response = await client.get(
            _bundle_url(), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        assert _zip_names(response.content) == []
        assert (
            response.headers["x-filename"] == f"{_NAMESPACE}__{_JOB_ID}__{_EPOCH}.zip"
        )


# ============================================================
# Archive trust boundary
# ============================================================


class TestZipBundleArchiveTrustBoundary:
    """Archive members must stay user-artifact-only and extraction-safe."""

    @pytest.mark.asyncio
    async def test_download_historical_bundle_excludes_ready_marker_and_symlink_escape(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        run = _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json": b'{"status": "complete"}'},
        )
        outside_secret = tmp_path / "operator-service-account-token.txt"
        outside_secret.write_bytes(b"cluster-admin-token")
        (run / "token-link.txt").symlink_to(outside_secret)

        response = await client.get(
            _bundle_url(), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        entries = _zip_entries(response.content)
        assert entries == {"profile_export_aiperf.json": b'{"status": "complete"}'}
        assert READY_MARKER_NAME not in entries
        assert "token-link.txt" not in entries
        assert b"cluster-admin-token" not in response.content

    @pytest.mark.asyncio
    async def test_download_historical_bundle_member_names_are_relative_and_sorted(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={
                "zeta_metrics.json": b'{"order": 3}',
                "alpha_summary.json": b'{"order": 1}',
                "middle.csv": b"metric,value\nttft_ms,12.5\n",
            },
        )

        response = await client.get(
            _bundle_url(), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        names = _zip_names(response.content)
        assert names == ["alpha_summary.json", "middle.csv", "zeta_metrics.json"]
        assert all(not Path(name).is_absolute() for name in names)
        assert all(".." not in Path(name).parts for name in names)


# ============================================================
# Payload preservation
# ============================================================


class TestZipBundlePayloadPreservation:
    """Zip bundle entries preserve artifact bytes and visible filenames."""

    @pytest.mark.asyncio
    async def test_download_historical_bundle_preserves_binary_file_bytes(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        png_payload = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 5 + b"IEND"
        parquet_payload = b"PAR1" + bytes(reversed(range(256))) * 3
        _seed_run(
            tmp_path,
            files={
                "latency_histogram.png": png_payload,
                "records.parquet": parquet_payload,
            },
        )

        response = await client.get(
            _bundle_url(), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        entries = _zip_entries(response.content)
        assert entries["latency_histogram.png"] == png_payload
        assert entries["records.parquet"] == parquet_payload

    @pytest.mark.asyncio
    async def test_download_historical_bundle_decompresses_zstd_member_to_stable_name(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        import zstandard

        summary_payload = b'{"request_throughput": {"avg": 101.5}}'
        compressed = zstandard.ZstdCompressor().compress(summary_payload)
        _seed_run(tmp_path, files={"profile_export_aiperf.json.zst": compressed})

        response = await client.get(
            _bundle_url(), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        entries = _zip_entries(response.content)
        assert entries == {"profile_export_aiperf.json": summary_payload}
        assert "profile_export_aiperf.json.zst" not in entries


# ============================================================
# Checkpoint bundle semantics
# ============================================================


class TestZipBundleCheckpointSemantics:
    """Bundle behavior must match the visible files users can download."""

    @pytest.mark.asyncio
    async def test_download_historical_bundle_missing_ready_marker_does_not_bypass_for_checkpoint(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={
                f"{CHECKPOINTS_DIR_NAME}/records-manager-0001.parquet": b"PAR1partial"
            },
            ready=False,
        )

        response = await client.get(
            _bundle_url(), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 404
        assert READY_MARKER_NAME in response.json()["detail"]
        assert b"PAR1partial" not in response.content

    @pytest.mark.asyncio
    async def test_download_historical_bundle_ready_run_includes_visible_checkpoint(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": b'{"complete": true}',
                f"{CHECKPOINTS_DIR_NAME}/records-manager-0001.parquet": b"PAR1checkpoint",
            },
            ready=True,
        )

        list_response = await client.get(
            f"/api/v1/results/{_NAMESPACE}/{_JOB_ID}/runs/{_EPOCH}"
        )
        bundle_response = await client.get(
            _bundle_url(), headers={"Accept-Encoding": "identity"}
        )

        assert list_response.status_code == 200, list_response.text
        visible_names = {entry["name"] for entry in list_response.json()["files"]}
        assert f"{CHECKPOINTS_DIR_NAME}/records-manager-0001.parquet" in visible_names
        assert bundle_response.status_code == 200, bundle_response.text
        entries = _zip_entries(bundle_response.content)
        assert (
            entries[f"{CHECKPOINTS_DIR_NAME}/records-manager-0001.parquet"]
            == b"PAR1checkpoint"
        )


# ============================================================
# Streaming boundaries
# ============================================================


class TestZipBundleStreamingBoundaries:
    """Bundle streaming yields fixed-size chunks without corrupting the archive."""

    @pytest.mark.asyncio
    async def test_stream_job_bundle_large_archive_yields_chunk_sized_segments(
        self, tmp_path: Path
    ) -> None:
        run = _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": b'{"complete": true}',
                "large_trace_records.jsonl": (b'{"token":"a"}\n' * (CHUNK_SIZE // 4)),
            },
        )

        chunks = [chunk async for chunk in _stream_job_bundle(run)]

        assert len(chunks) >= 2
        assert all(0 < len(chunk) <= CHUNK_SIZE for chunk in chunks)
        entries = _zip_entries(b"".join(chunks))
        assert entries["profile_export_aiperf.json"] == b'{"complete": true}'
        assert entries["large_trace_records.jsonl"].startswith(b'{"token":"a"}\n')
