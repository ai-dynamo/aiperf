# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.operator.progress_download``.

Covers the streaming download helpers that ``ProgressClient`` delegates to:
decompressor factory, zstd passthrough write, transcode-to-zstd write, and
plain decompressed write. Uses an async-iterator stand-in for
``aiohttp.ClientResponse.content.iter_chunked``.
"""

from __future__ import annotations

import gzip
import io
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import zstandard as zstd
from pytest import param

from aiperf.operator.progress_download import (
    make_decompressor,
    save_decompressed,
    save_transcoded_zstd,
    save_zstd_passthrough,
)


def _zstd_decode_stream(data: bytes) -> bytes:
    """Decode a zstd frame produced by a streaming compressor (no content size)."""
    with zstd.ZstdDecompressor().stream_reader(io.BytesIO(data)) as r:
        return r.read()


def _fake_response(chunks: list[bytes]) -> MagicMock:
    """Build a stand-in response whose ``content.iter_chunked`` yields ``chunks``."""

    async def _iter(_size: int):
        for c in chunks:
            yield c

    response = MagicMock()
    response.content.iter_chunked = _iter
    return response


class TestMakeDecompressor:
    """Tests for ``make_decompressor``."""

    @pytest.mark.parametrize(
        "encoding,is_none",
        [
            param("zstd", False, id="zstd"),
            param("gzip", False, id="gzip"),
            param("identity", True, id="identity"),
            param("", True, id="empty"),
            param("br", True, id="unknown"),
        ],
    )  # fmt: skip
    def test_returns_decompressor_for_known_encoding(
        self, encoding: str, is_none: bool
    ) -> None:
        """Verify the factory returns a decompressor for zstd/gzip and None otherwise."""
        result = make_decompressor(encoding)
        if is_none:
            assert result is None
        else:
            assert result is not None
            # Either returns a decompressor with .decompress()
            assert hasattr(result, "decompress")

    def test_zstd_decompressor_roundtrip(self) -> None:
        """Verify the zstd decompressor correctly reverses zstd-compressed bytes."""
        payload = b"hello-zstd-payload" * 20
        compressed = zstd.ZstdCompressor().compress(payload)

        decomp = make_decompressor("zstd")
        assert decomp is not None
        assert decomp.decompress(compressed) == payload

    def test_gzip_decompressor_roundtrip(self) -> None:
        """Verify the gzip decompressor correctly reverses gzip-compressed bytes."""
        payload = b"hello-gzip-payload" * 20
        compressed = gzip.compress(payload)

        decomp = make_decompressor("gzip")
        assert decomp is not None
        assert decomp.decompress(compressed) == payload


class TestSaveZstdPassthrough:
    """Tests for ``save_zstd_passthrough``."""

    @pytest.mark.asyncio
    async def test_writes_bytes_verbatim_to_zst_suffix(self, tmp_path: Path) -> None:
        """Verify the response body is written byte-for-byte to ``<name>.zst``."""
        payload = b"already-zstd-compressed-bytes"
        response = _fake_response([payload[:10], payload[10:]])
        dest = tmp_path / "results.json"

        await save_zstd_passthrough(response, dest)

        zst_path = tmp_path / "results.json.zst"
        assert zst_path.exists()
        assert zst_path.read_bytes() == payload

    @pytest.mark.asyncio
    async def test_empty_chunks_are_skipped(self, tmp_path: Path) -> None:
        """Verify empty chunks are skipped (no zero-byte write calls)."""
        response = _fake_response([b"", b"abc", b""])
        dest = tmp_path / "x.bin"

        await save_zstd_passthrough(response, dest)

        assert (tmp_path / "x.bin.zst").read_bytes() == b"abc"


class TestSaveTranscodedZstd:
    """Tests for ``save_transcoded_zstd``."""

    @pytest.mark.asyncio
    async def test_transcodes_gzip_to_zstd(self, tmp_path: Path) -> None:
        """Verify a gzip-wire body is re-compressed as zstd on disk."""
        payload = b"the quick brown fox jumps over the lazy dog" * 50
        compressed = gzip.compress(payload)
        # Split mid-stream so the decompressor has to buffer across chunks.
        mid = len(compressed) // 2
        response = _fake_response([compressed[:mid], compressed[mid:]])
        dest = tmp_path / "data.json"

        decomp = zlib.decompressobj(wbits=31)
        await save_transcoded_zstd(response, dest, decomp)

        zst_bytes = (tmp_path / "data.json.zst").read_bytes()
        # Decompressing the stored zstd must round-trip to the original payload.
        assert _zstd_decode_stream(zst_bytes) == payload

    @pytest.mark.asyncio
    async def test_no_decompressor_recompresses_as_zstd(self, tmp_path: Path) -> None:
        """Verify identity-wire bodies are compressed as zstd on disk."""
        payload = b"raw-bytes-no-wire-compression" * 30
        response = _fake_response([payload])
        dest = tmp_path / "raw.json"

        await save_transcoded_zstd(response, dest, decompressor=None)

        zst_bytes = (tmp_path / "raw.json.zst").read_bytes()
        assert _zstd_decode_stream(zst_bytes) == payload


class TestSaveDecompressed:
    """Tests for ``save_decompressed``."""

    @pytest.mark.asyncio
    async def test_writes_decompressed_bytes_to_dest(self, tmp_path: Path) -> None:
        """Verify a zstd-wire body is decoded and written raw to ``dest_path``."""
        payload = b"decoded-output-bytes" * 20
        compressed = zstd.ZstdCompressor().compress(payload)
        mid = len(compressed) // 2
        response = _fake_response([compressed[:mid], compressed[mid:]])
        dest = tmp_path / "raw.json"

        decomp = zstd.ZstdDecompressor().decompressobj()
        await save_decompressed(response, dest, decomp)

        assert dest.read_bytes() == payload

    @pytest.mark.asyncio
    async def test_no_decompressor_writes_bytes_verbatim(self, tmp_path: Path) -> None:
        """Verify identity wire encoding (decompressor=None) writes raw bytes."""
        payload = b"plain-body"
        response = _fake_response([payload])
        dest = tmp_path / "plain.json"

        await save_decompressed(response, dest, decompressor=None)

        assert dest.read_bytes() == payload
