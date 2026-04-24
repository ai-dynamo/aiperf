# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ProgressClient._download_response with COMPRESS_ON_DISK support.

Focuses on:
- COMPRESS_ON_DISK=True: zstd passthrough, gzip->zstd transcoding, identity->zstd
- COMPRESS_ON_DISK=False: original behavior (decompress to raw files)
- Edge cases: empty responses, multi-chunk responses
"""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import patch

import pytest
import zstandard as zstd

from aiperf.operator.progress_client import ProgressClient

# ============================================================
# Helpers
# ============================================================


class FakeStreamContent:
    """Fake aiohttp response content with iter_chunked support."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class FakeResponse:
    """Fake aiohttp.ClientResponse with content streaming."""

    def __init__(
        self, chunks: list[bytes], headers: dict[str, str] | None = None
    ) -> None:
        self.content = FakeStreamContent(chunks)
        self.headers = headers or {}
        self.status = 200


def _compress_zstd(data: bytes) -> bytes:
    """Compress data with zstd."""
    return zstd.ZstdCompressor().compress(data)


def _decompress_zstd(data: bytes) -> bytes:
    """Decompress zstd data (handles streaming frames without content size)."""
    dctx = zstd.ZstdDecompressor()
    # Use stream_reader for frames that may lack content size header
    return dctx.decompress(data, max_output_size=10 * 1024 * 1024)


# ============================================================
# COMPRESS_ON_DISK=False (original behavior)
# ============================================================


class TestDownloadResponseCompressOff:
    """Verify _download_response with COMPRESS_ON_DISK=False preserves original behavior."""

    @pytest.mark.asyncio
    async def test_identity_encoding_saves_raw(self, tmp_path: Path) -> None:
        content = b'{"metric": "value"}'
        response = FakeResponse([content])
        dest = tmp_path / "metrics.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = False
            await client._download_response(response, dest, "identity")

        assert dest.exists()
        assert dest.read_bytes() == content

    @pytest.mark.asyncio
    async def test_gzip_encoding_decompresses_to_raw(self, tmp_path: Path) -> None:
        original = b'{"metric": "gzipped"}'
        compressed = gzip.compress(original)
        response = FakeResponse([compressed])
        dest = tmp_path / "metrics.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = False
            await client._download_response(response, dest, "gzip")

        assert dest.exists()
        assert dest.read_bytes() == original

    @pytest.mark.asyncio
    async def test_zstd_encoding_decompresses_to_raw(self, tmp_path: Path) -> None:
        original = b'{"metric": "zstd_value"}'
        compressed = _compress_zstd(original)
        response = FakeResponse([compressed])
        dest = tmp_path / "metrics.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = False
            await client._download_response(response, dest, "zstd")

        assert dest.exists()
        assert dest.read_bytes() == original


# ============================================================
# COMPRESS_ON_DISK=True (new behavior)
# ============================================================


class TestDownloadResponseCompressOn:
    """Verify _download_response with COMPRESS_ON_DISK=True stores as .zst."""

    @pytest.mark.asyncio
    async def test_zstd_encoding_passthrough_to_zst(self, tmp_path: Path) -> None:
        original = b'{"metric": "zstd_passthrough"}'
        compressed = _compress_zstd(original)
        response = FakeResponse([compressed])
        dest = tmp_path / "metrics.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = True
            await client._download_response(response, dest, "zstd")

        zst_path = tmp_path / "metrics.json.zst"
        assert zst_path.exists()
        assert not dest.exists()
        # Verify the passthrough preserved valid zstd data
        assert _decompress_zstd(zst_path.read_bytes()) == original

    @pytest.mark.asyncio
    async def test_gzip_encoding_transcodes_to_zst(self, tmp_path: Path) -> None:
        original = b'{"metric": "gzip_transcoded"}'
        compressed = gzip.compress(original)
        response = FakeResponse([compressed])
        dest = tmp_path / "metrics.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = True
            await client._download_response(response, dest, "gzip")

        zst_path = tmp_path / "metrics.json.zst"
        assert zst_path.exists()
        assert not dest.exists()
        assert _decompress_zstd(zst_path.read_bytes()) == original

    @pytest.mark.asyncio
    async def test_identity_encoding_compresses_to_zst(self, tmp_path: Path) -> None:
        original = b'{"metric": "identity_compressed"}'
        response = FakeResponse([original])
        dest = tmp_path / "metrics.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = True
            await client._download_response(response, dest, "identity")

        zst_path = tmp_path / "metrics.json.zst"
        assert zst_path.exists()
        assert not dest.exists()
        assert _decompress_zstd(zst_path.read_bytes()) == original


# ============================================================
# Edge Cases
# ============================================================


class TestDownloadResponseEdgeCases:
    """Verify edge cases for _download_response."""

    @pytest.mark.asyncio
    async def test_empty_response_identity_compress_off(self, tmp_path: Path) -> None:
        response = FakeResponse([])
        dest = tmp_path / "empty.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = False
            await client._download_response(response, dest, "identity")

        assert dest.exists()
        assert dest.read_bytes() == b""

    @pytest.mark.asyncio
    async def test_multi_chunk_identity_compress_off(self, tmp_path: Path) -> None:
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        response = FakeResponse(chunks)
        dest = tmp_path / "multi.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = False
            await client._download_response(response, dest, "identity")

        assert dest.read_bytes() == b"chunk1chunk2chunk3"

    @pytest.mark.asyncio
    async def test_multi_chunk_identity_compress_on(self, tmp_path: Path) -> None:
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        response = FakeResponse(chunks)
        dest = tmp_path / "multi.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = True
            await client._download_response(response, dest, "identity")

        zst_path = tmp_path / "multi.json.zst"
        assert zst_path.exists()
        assert _decompress_zstd(zst_path.read_bytes()) == b"chunk1chunk2chunk3"

    @pytest.mark.asyncio
    async def test_large_payload_compress_on_zstd(self, tmp_path: Path) -> None:
        original = b"x" * 1_000_000
        compressed = _compress_zstd(original)
        # Split into multiple chunks
        chunk_size = 64 * 1024
        chunks = [
            compressed[i : i + chunk_size]
            for i in range(0, len(compressed), chunk_size)
        ]
        response = FakeResponse(chunks)
        dest = tmp_path / "large.json"

        client = ProgressClient()
        with patch("aiperf.operator.environment.OperatorEnvironment") as mock_env:
            mock_env.RESULTS.COMPRESS_ON_DISK = True
            await client._download_response(response, dest, "zstd")

        zst_path = tmp_path / "large.json.zst"
        assert zst_path.exists()
        # Passthrough: bytes should be the same as the compressed input
        assert _decompress_zstd(zst_path.read_bytes()) == original
