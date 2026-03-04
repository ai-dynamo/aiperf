# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compression module."""

from __future__ import annotations

import gzip
import io
import pathlib

import pytest
from pytest import param

from aiperf.common.compression import (
    CompressionEncoding,
    is_zstd_available,
    select_encoding,
    stream_file_compressed,
)
from aiperf.common.environment import Environment


class TestCompressionAvailability:
    """Test compression library availability checks."""

    def test_is_zstd_available_returns_bool(self) -> None:
        """Test that is_zstd_available returns a boolean."""
        result = is_zstd_available()
        assert isinstance(result, bool)

    def test_availability_caching(self) -> None:
        """Test that availability checks are cached."""
        # Call twice - should return same result (tests caching)
        result1 = is_zstd_available()
        result2 = is_zstd_available()
        assert result1 == result2


class TestSelectEncoding:
    """Test encoding selection logic."""

    @pytest.mark.parametrize(
        "accept_encoding,expected",
        [
            param("zstd, gzip", CompressionEncoding.ZSTD, id="prefers-zstd"),
            param("gzip", CompressionEncoding.GZIP, id="gzip-only"),
            param("deflate, br", CompressionEncoding.IDENTITY, id="unknown-identity-fallback"),
            param(None, CompressionEncoding.GZIP, id="none-fallback"),
            param("", CompressionEncoding.GZIP, id="empty-fallback"),
            param("ZSTD", CompressionEncoding.ZSTD, id="case-insensitive-zstd"),
            param("GZIP", CompressionEncoding.GZIP, id="case-insensitive-gzip"),
        ],
    )  # fmt: skip
    def test_select_encoding(
        self, accept_encoding: str | None, expected: CompressionEncoding
    ) -> None:
        """Test encoding selection based on Accept-Encoding header."""
        result = select_encoding(accept_encoding)
        # If zstd not available, result may differ
        if expected == CompressionEncoding.ZSTD and not is_zstd_available():
            assert result == CompressionEncoding.GZIP
        else:
            assert result == expected

    def test_select_encoding_custom_default(self) -> None:
        """Test that custom default is used when no encoding matches."""
        result = select_encoding("br, deflate", default=CompressionEncoding.IDENTITY)
        assert result == CompressionEncoding.IDENTITY


class TestFileStreaming:
    """Test file streaming compression functions."""

    @pytest.fixture
    def temp_file(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Create a temporary test file."""
        file_path = tmp_path / "test.dat"
        test_data = b"Hello World! " * 1000
        file_path.write_bytes(test_data)
        return file_path

    async def test_stream_file_compressed_gzip(self, temp_file: pathlib.Path) -> None:
        """Test gzip file compression streaming."""
        chunks = []
        async for chunk in stream_file_compressed(temp_file, CompressionEncoding.GZIP):
            chunks.append(chunk)

        compressed = b"".join(chunks)
        # Verify it's valid gzip by decompressing
        decompressed = gzip.decompress(compressed)
        assert decompressed == temp_file.read_bytes()

    @pytest.mark.skipif(not is_zstd_available(), reason="zstandard not installed")
    async def test_stream_file_compressed_zstd(self, temp_file: pathlib.Path) -> None:
        """Test zstd file compression streaming."""
        import zstandard

        chunks = []
        async for chunk in stream_file_compressed(temp_file, CompressionEncoding.ZSTD):
            chunks.append(chunk)

        compressed = b"".join(chunks)
        # Streaming compressor omits content size — use stream_reader
        decompressor = zstandard.ZstdDecompressor()
        decompressed = decompressor.stream_reader(io.BytesIO(compressed)).read()
        assert decompressed == temp_file.read_bytes()

    async def test_stream_file_compressed_identity(
        self, temp_file: pathlib.Path
    ) -> None:
        """Test identity (no compression) streaming."""
        chunks = []
        async for chunk in stream_file_compressed(
            temp_file, CompressionEncoding.IDENTITY
        ):
            chunks.append(chunk)

        result = b"".join(chunks)
        assert result == temp_file.read_bytes()


class TestQualityValueParsing:
    """Test Accept-Encoding quality value (q=) parsing."""

    @pytest.mark.parametrize(
        "accept_encoding,expected",
        [
            param("gzip;q=1.0, zstd;q=0.5", CompressionEncoding.ZSTD, id="zstd-lower-quality-still-preferred"),
            param("gzip;q=0.8", CompressionEncoding.GZIP, id="gzip-with-quality"),
            param("zstd;q=0, gzip;q=1.0", CompressionEncoding.GZIP, id="zstd-rejected-q0"),
            param("gzip;q=0, zstd;q=0", CompressionEncoding.IDENTITY, id="all-rejected-identity-fallback"),
            param("gzip;q=0, zstd;q=0, identity;q=0", CompressionEncoding.GZIP, id="all-rejected-falls-to-default"),
            param("identity;q=1.0", CompressionEncoding.IDENTITY, id="identity-only"),
            param("zstd;q=0, gzip;q=0, identity;q=1.0", CompressionEncoding.IDENTITY, id="only-identity-accepted"),
        ],
    )  # fmt: skip
    def test_quality_value_encoding_selection(
        self, accept_encoding: str, expected: CompressionEncoding
    ) -> None:
        """Test encoding selection respects quality values."""
        result = select_encoding(accept_encoding)
        if expected == CompressionEncoding.ZSTD and not is_zstd_available():
            assert result == CompressionEncoding.GZIP
        else:
            assert result == expected

    def test_q_zero_explicit_rejection(self) -> None:
        """Test that q=0 explicitly rejects an encoding."""
        result = select_encoding("gzip;q=0", default=CompressionEncoding.IDENTITY)
        assert result == CompressionEncoding.IDENTITY


class TestEmptyFileStreaming:
    """Test streaming of empty files."""

    async def test_stream_empty_file_identity(self, tmp_path: pathlib.Path) -> None:
        """Test streaming an empty file with identity encoding."""
        empty_file = tmp_path / "empty.dat"
        empty_file.write_bytes(b"")

        chunks = []
        async for chunk in stream_file_compressed(
            empty_file, CompressionEncoding.IDENTITY
        ):
            chunks.append(chunk)

        assert b"".join(chunks) == b""

    async def test_stream_empty_file_gzip(self, tmp_path: pathlib.Path) -> None:
        """Test streaming an empty file with gzip produces valid gzip output."""
        empty_file = tmp_path / "empty.dat"
        empty_file.write_bytes(b"")

        chunks = []
        async for chunk in stream_file_compressed(empty_file, CompressionEncoding.GZIP):
            chunks.append(chunk)

        compressed = b"".join(chunks)
        decompressed = gzip.decompress(compressed)
        assert decompressed == b""


class TestConstants:
    """Test module constants."""

    def test_chunk_size_is_reasonable(self) -> None:
        """Test that chunk size is a reasonable value."""
        assert Environment.COMPRESSION.CHUNK_SIZE > 0
        assert Environment.COMPRESSION.CHUNK_SIZE == 64 * 1024  # 64KB
