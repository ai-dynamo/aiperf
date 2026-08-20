# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for compressed operator result artifacts.

Focuses on:
- .zst stored-name to display-name mapping in file-list responses
- content negotiation for zstd-stored JSON/JSONL/Parquet artifacts
- stored-size metadata rather than decompressed-size metadata
- malformed compressed profile exports failing closed instead of returning corrupt JSON
- raw plus compressed sibling preference for quick export and per-file download routes
- results-ready gating for final compressed artifacts and checkpoint bypass semantics

Out of scope: run discovery and path traversal, covered by sibling results-files tests.
"""

from __future__ import annotations

import gzip
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote

import httpx
import orjson
import pytest
import zstandard
from aiperf.operator.results_layout import run_dir, write_latest
from aiperf.operator.routers.results_files import create_results_files_router
from fastapi import FastAPI

# ============================================================
# Helpers
# ============================================================

_EPOCH = "1714150923"
_READY_MARKER = ".aiperf_results_ready.json"


def _app_for_results_dir(results_dir: Path) -> FastAPI:
    """Build only the results-files router so tests avoid k8s lifespan side effects."""
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


def _zstd(payload: bytes) -> bytes:
    """Return a valid zstd frame for a realistic stored artifact payload."""
    return zstandard.ZstdCompressor().compress(payload)


def _seed_run(
    base_dir: Path,
    *,
    namespace: str = "bench-prod",
    job_id: str = "llama-3-8b-load",
    epoch: str = _EPOCH,
    files: dict[str, bytes] | None = None,
    ready: bool = True,
) -> Path:
    """Create one epoch-pinned run directory with result artifacts."""
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


def _artifact_list_url() -> str:
    """Return the epoch-pinned file-listing URL for the canonical benchmark run."""
    return f"/api/v1/results/bench-prod/llama-3-8b-load/runs/{_EPOCH}"


def _artifact_url(filename: str) -> str:
    """Return the epoch-pinned artifact download URL for the canonical run."""
    return f"{_artifact_list_url()}/{quote(filename)}"


def _entry_by_name(files: list[dict[str, object]], name: str) -> dict[str, object]:
    """Return one file-list entry by display name."""
    return next(entry for entry in files if entry["name"] == name)


async def _read_raw_stream(
    client: httpx.AsyncClient, url: str, *, accept_encoding: str
) -> tuple[httpx.Response, bytes]:
    """Read wire bytes before httpx applies content-encoding decompression."""
    async with client.stream(
        "GET", url, headers={"Accept-Encoding": accept_encoding}
    ) as response:
        chunks = [chunk async for chunk in response.aiter_raw()]
        return response, b"".join(chunks)


# ============================================================
# Stored-name mapping and metadata
# ============================================================


class TestCompressedArtifactListingMetadata:
    """List responses describe the stored bytes while exposing display names."""

    @pytest.mark.asyncio
    async def test_list_historical_files_zst_json_maps_display_name_and_stored_size(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        payload = b'{"request_throughput":{"avg":42.5}}'
        stored_payload = _zstd(payload)
        _seed_run(tmp_path, files={"profile_export_aiperf.json.zst": stored_payload})

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        entry = _entry_by_name(response.json()["files"], "profile_export_aiperf.json")
        assert entry["name"] == "profile_export_aiperf.json"
        assert entry["stored_name"] == "profile_export_aiperf.json.zst"
        assert entry["size_bytes"] == len(stored_payload)
        assert entry["compressed"] is True

    @pytest.mark.asyncio
    async def test_list_historical_files_checkpoint_zst_preserves_relative_display_name(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        payload = b"PAR1checkpoint-record-batch"
        stored_payload = _zstd(payload)
        _seed_run(
            tmp_path,
            files={"checkpoints/records-manager-0001.parquet.zst": stored_payload},
        )

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        entry = _entry_by_name(
            response.json()["files"], "checkpoints/records-manager-0001.parquet"
        )
        assert entry["stored_name"] == "checkpoints/records-manager-0001.parquet.zst"
        assert entry["size_bytes"] == len(stored_payload)
        assert entry["compressed"] is True


# ============================================================
# Download content negotiation
# ============================================================


class TestCompressedArtifactDownloadNegotiation:
    """A .zst stored file is served as the logical artifact unless requested raw."""

    @pytest.mark.asyncio
    async def test_download_historical_file_zst_identity_decompresses_with_json_media_type(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        payload = b'{"status":"Succeeded","request_throughput":{"avg":77}}'
        _seed_run(tmp_path, files={"metrics.json.zst": _zstd(payload)})

        response = await client.get(
            _artifact_url("metrics.json"), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        assert response.content == payload
        assert "content-encoding" not in response.headers
        assert (
            response.headers["content-type"].split(";", maxsplit=1)[0]
            == "application/json"
        )
        assert response.headers["x-filename"] == "metrics.json"

    @pytest.mark.asyncio
    async def test_download_historical_file_zst_accept_zstd_returns_stored_frame(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        payload = b'{"idx":1,"latency_ms":12.5}\n'
        stored_payload = _zstd(payload)
        _seed_run(tmp_path, files={"trace_records.jsonl.zst": stored_payload})

        response, raw_body = await _read_raw_stream(
            client, _artifact_url("trace_records.jsonl"), accept_encoding="zstd"
        )

        assert response.status_code == 200, response.text
        assert raw_body == stored_payload
        assert response.headers["content-encoding"] == "zstd"
        assert (
            response.headers["content-type"].split(";", maxsplit=1)[0]
            == "application/x-ndjson"
        )
        assert response.headers["x-filename"] == "trace_records.jsonl"

    @pytest.mark.asyncio
    async def test_download_historical_file_zst_accept_gzip_transcodes_decompressed_payload(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        payload = b"PAR1" + bytes(range(64))
        _seed_run(tmp_path, files={"records.parquet.zst": _zstd(payload)})

        response, raw_body = await _read_raw_stream(
            client, _artifact_url("records.parquet"), accept_encoding="gzip"
        )

        assert response.status_code == 200, response.text
        assert raw_body.startswith(b"\x1f\x8b")
        assert gzip.decompress(raw_body) == payload
        assert response.headers["content-encoding"] == "gzip"
        assert (
            response.headers["content-type"].split(";", maxsplit=1)[0]
            == "application/vnd.apache.parquet"
        )


# ============================================================
# Sibling raw/compressed preference
# ============================================================


class TestCompressedArtifactSiblingPreference:
    """Raw and compressed siblings intentionally differ by route contract."""

    @pytest.mark.asyncio
    async def test_profile_export_quick_raw_and_zst_siblings_prefers_raw_json(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        raw_payload = b'{"source":"raw-profile-export"}'
        compressed_payload = b'{"source":"compressed-profile-export"}'
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json": raw_payload,
                "profile_export_aiperf.json.zst": _zstd(compressed_payload),
            },
        )

        response = await client.get(f"{_artifact_list_url()}/profile_export")

        assert response.status_code == 200, response.text
        assert response.content == raw_payload
        assert (
            response.headers["content-type"].split(";", maxsplit=1)[0]
            == "application/json"
        )
        assert response.headers["x-filename"] == "profile_export_aiperf.json"

    @pytest.mark.asyncio
    async def test_download_historical_file_raw_and_zst_siblings_prefers_compressed_store(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        raw_payload = b'{"source":"raw-metrics"}'
        compressed_payload = b'{"source":"compressed-metrics"}'
        _seed_run(
            tmp_path,
            files={
                "metrics.json": raw_payload,
                "metrics.json.zst": _zstd(compressed_payload),
            },
        )

        response = await client.get(
            _artifact_url("metrics.json"), headers={"Accept-Encoding": "identity"}
        )

        assert response.status_code == 200, response.text
        assert response.content == compressed_payload
        assert response.content != raw_payload
        assert response.headers["x-filename"] == "metrics.json"


# ============================================================
# Ready gating and malformed compressed payloads
# ============================================================


class TestCompressedArtifactReadyGateAndMalformedPayloads:
    """Final compressed artifacts stay gated; corrupt compressed exports fail closed."""

    @pytest.mark.asyncio
    async def test_list_historical_files_not_ready_hides_top_level_zst_artifact(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json.zst": _zstd(b'{"partial":true}')},
            ready=False,
        )

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        assert response.json()["files"] == []

    @pytest.mark.asyncio
    async def test_download_historical_file_not_ready_rejects_top_level_zst_artifact(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json.zst": _zstd(b'{"partial":true}')},
            ready=False,
        )

        response = await client.get(
            _artifact_url("profile_export_aiperf.json"),
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 404
        assert _READY_MARKER in response.json()["detail"]
        assert b'"partial":true' not in response.content

    @pytest.mark.asyncio
    async def test_profile_export_quick_malformed_zst_returns_server_error_not_json(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_run(
            tmp_path,
            files={"profile_export_aiperf.json.zst": b"not-a-zstd-frame"},
        )

        response = await client.get(f"{_artifact_list_url()}/profile_export")

        assert response.status_code == 500
        assert (
            response.headers["content-type"].split(";", maxsplit=1)[0] == "text/plain"
        )
        assert b"profile_export_aiperf" not in response.content


# ============================================================
# Checkpoint compressed artifacts
# ============================================================


class TestCompressedCheckpointArtifacts:
    """Compressed checkpoints are inspectable before final results are ready."""

    @pytest.mark.asyncio
    async def test_list_historical_files_not_ready_lists_checkpoint_zst_metadata_only(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        checkpoint_payload = b"PAR1checkpoint-record-batch"
        stored_payload = _zstd(checkpoint_payload)
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json.zst": _zstd(b'{"hidden":true}'),
                "checkpoints/records-manager-0001.parquet.zst": stored_payload,
            },
            ready=False,
        )

        response = await client.get(_artifact_list_url())

        assert response.status_code == 200, response.text
        files = response.json()["files"]
        assert len(files) == 1
        assert files[0]["name"] == "checkpoints/records-manager-0001.parquet"
        assert files[0]["stored_name"] == "checkpoints/records-manager-0001.parquet.zst"
        assert files[0]["size_bytes"] == len(stored_payload)
        assert files[0]["compressed"] is True

    @pytest.mark.asyncio
    async def test_download_historical_file_not_ready_allows_checkpoint_zst_decompression(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        checkpoint_payload = b"PAR1checkpoint-record-batch"
        _seed_run(
            tmp_path,
            files={
                "profile_export_aiperf.json.zst": _zstd(b'{"hidden":true}'),
                "checkpoints/records-manager-0001.parquet.zst": _zstd(
                    checkpoint_payload
                ),
            },
            ready=False,
        )

        response = await client.get(
            _artifact_url("checkpoints/records-manager-0001.parquet"),
            headers={"Accept-Encoding": "identity"},
        )

        assert response.status_code == 200, response.text
        assert response.content == checkpoint_payload
        assert (
            response.headers["content-type"].split(";", maxsplit=1)[0]
            == "application/vnd.apache.parquet"
        )
        assert response.headers["x-filename"] == "records-manager-0001.parquet"
