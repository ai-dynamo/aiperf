# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator artifact download headers.

Focuses on:
- media-type contracts for downloadable JSON, JSONL, Parquet, CSV, PNG, and ZIP artifacts
- RFC 5987 filename encoding for non-ASCII download names
- reserved/internal artifact files remaining hidden even by direct download
- cache headers on immutable epoch-pinned downloads
- path traversal and binary streaming at chunk boundaries

Out of scope: run discovery/index fallback, covered by sibling results-files router tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote

import httpx
import orjson
import pytest
from aiperf.operator.results_layout import run_dir, write_latest
from aiperf.operator.routers.results_files import create_results_files_router
from aiperf.operator.routers.results_files_io import CHUNK_SIZE
from fastapi import FastAPI
from pytest import param

# ============================================================
# Helpers
# ============================================================

_EPOCH = "1714150923"
_READY_MARKER = ".aiperf_results_ready.json"


def _app_for_results_dir(results_dir: Path) -> FastAPI:
    """Build the results-files router without the full operator app lifespan."""
    app = FastAPI()
    app.include_router(create_results_files_router(results_dir))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by a temp PVC-like artifact directory."""
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
    """Create one epoch-pinned run directory with visible artifact files."""
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


def _run_artifact_url(filename: str) -> str:
    """Return the epoch-pinned artifact URL for the canonical benchmark run."""
    encoded_filename = quote(filename)
    return (
        f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH}/{encoded_filename}"
    )


# ============================================================
# Media types and cache headers
# ============================================================


class TestArtifactDownloadMediaTypes:
    """Artifact extensions define browser-visible media types, not octet-stream."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename,payload,expected_media_type",
        [
            ("profile_export_aiperf.json", b'{"status":"Succeeded"}', "application/json"),
            param("trace_records.jsonl", b'{"idx":1}\n', "application/x-ndjson", id="jsonl-ndjson"),
            param("records.parquet", b"PAR1\x00\x01", "application/vnd.apache.parquet", id="parquet"),
            param("metrics.csv", b"metric,value\nttft_ms,12.5\n", "text/csv", id="csv"),
            param("latency_histogram.png", b"\x89PNG\r\n\x1a\n", "image/png", id="png"),
        ],
    )  # fmt: skip
    async def test_download_historical_file_known_extension_sets_specific_content_type(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        filename: str,
        payload: bytes,
        expected_media_type: str,
    ) -> None:
        _seed_run(tmp_path, files={filename: payload})

        response = await client.get(
            _run_artifact_url(filename), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        content_type = response.headers["content-type"].split(";", maxsplit=1)[0]
        assert content_type == expected_media_type
        assert response.content == payload

    @pytest.mark.asyncio
    async def test_download_historical_bundle_zip_sets_zip_content_type_and_filename(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(tmp_path, files={"profile_export_aiperf.json": b'{"ok":true}'})

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH}.zip",
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 200, response.text
        assert (
            response.headers["content-type"].split(";", maxsplit=1)[0]
            == "application/zip"
        )
        assert response.headers["x-filename"] == (
            f"bench-prod__llama-3-8b-load__{_EPOCH}.zip"
        )
        assert response.headers["content-disposition"] == (
            f'attachment; filename="bench-prod__llama-3-8b-load__{_EPOCH}.zip"'
        )
        assert response.content.startswith(b"PK")

    @pytest.mark.asyncio
    async def test_download_historical_file_epoch_pinned_response_is_not_cached(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(tmp_path, files={"profile_export_aiperf.json": b'{"run":"complete"}'})

        response = await client.get(
            _run_artifact_url("profile_export_aiperf.json"),
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["pragma"] == "no-cache"


# ============================================================
# Filename encoding and hidden internals
# ============================================================


class TestArtifactDownloadHeaderTrustBoundary:
    """Download headers must not expose ambiguous or internal filenames."""

    @pytest.mark.asyncio
    async def test_download_historical_file_non_ascii_filename_uses_rfc5987_encoding(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        filename = "metrics café summary.json"
        _seed_run(tmp_path, files={filename: b'{"request_throughput":{"avg":100}}'})

        response = await client.get(
            _run_artifact_url(filename), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        disposition = response.headers["content-disposition"]
        assert "filename*=UTF-8''metrics%20caf%C3%A9%20summary.json" in disposition
        assert 'filename="metrics café summary.json"' not in disposition
        assert response.headers["x-filename"] == filename

    @pytest.mark.asyncio
    async def test_download_historical_file_reserved_ready_marker_returns_404(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(tmp_path, files={"profile_export_aiperf.json": b"{}"})

        response = await client.get(
            _run_artifact_url(_READY_MARKER), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 404
        assert b'"ready":true' not in response.content
        assert _READY_MARKER in response.json()["detail"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "encoded_filename",
        [
            param("..%2F..%2Foperator-token.txt", id="encoded-parent-segments"),
            param("%2Ftmp%2Foperator-token.txt", id="encoded-absolute-path"),
        ],
    )  # fmt: skip
    async def test_download_historical_file_traversal_attempt_returns_404_without_secret(
        self, tmp_path: Path, client: httpx.AsyncClient, encoded_filename: str
    ) -> None:
        _seed_run(tmp_path, files={"profile_export_aiperf.json": b"{}"})
        (tmp_path / "operator-token.txt").write_bytes(b"cluster-admin-service-account")

        response = await client.get(
            f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH}/"
            f"{encoded_filename}",
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 404
        assert b"cluster-admin-service-account" not in response.content


# ============================================================
# Streaming boundaries
# ============================================================


class TestArtifactDownloadStreamingBoundaries:
    """Large binary artifacts stream byte-for-byte across chunk boundaries."""

    @pytest.mark.asyncio
    async def test_download_historical_file_large_png_preserves_boundary_bytes(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        payload = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 257 + b"END"
        assert len(payload) > CHUNK_SIZE
        _seed_run(tmp_path, files={"large_latency_histogram.png": payload})

        response = await client.get(
            _run_artifact_url("large_latency_histogram.png"),
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 200, response.text
        content_type = response.headers["content-type"].split(";", maxsplit=1)[0]
        assert content_type == "image/png"
        assert response.content == payload
