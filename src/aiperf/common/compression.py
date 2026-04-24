# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable compression utilities for streaming file compression.

Supports compression algorithms with content negotiation:
- zstd: Best compression ratio, fast decompression (preferred)
- gzip: Universal fallback

Usage::
    from aiperf.common.compression import (
        select_encoding,
        stream_file_compressed,
    )

    # Select encoding based on Accept-Encoding header
    encoding = select_encoding(request.headers.get("accept-encoding"))

    # Stream a compressed file
    async for chunk in stream_file_compressed(file_path, encoding):
        yield chunk
"""

from __future__ import annotations

import functools
import pathlib
import re
import zlib
from collections.abc import AsyncIterator
from typing import Any

import aiofiles

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.environment import Environment

_logger = AIPerfLogger(__name__)

_ACCEPT_ENCODING_REGEX = re.compile(r"([a-z*]+)\s*(?:;\s*q\s*=\s*([0-9.]+))?")


class CompressionEncoding(CaseInsensitiveStrEnum):
    """Supported compression encodings."""

    ZSTD = "zstd"
    """Zstandard compression. Best compression ratio, fast decompression. Requires zstandard library."""

    GZIP = "gzip"
    """Gzip compression. Universal fallback. Requires zlib library."""

    IDENTITY = "identity"
    """No compression."""


@functools.lru_cache(maxsize=1)
def is_zstd_available() -> bool:
    """Check if zstandard library is available."""
    try:
        import zstandard as zstandard  # noqa: F811 — availability check

        return True
    except ImportError as e:
        _logger.warning(
            f"zstandard library not installed: {e!r}. Please install `uv add zstandard` in order to use zstd compression."
        )
        return False


def _parse_accept_encoding(header: str) -> dict[str, float]:
    """Parse Accept-Encoding header into {encoding: quality} mapping."""
    return {
        m.group(1): float(m.group(2)) if m.group(2) else 1.0
        for m in _ACCEPT_ENCODING_REGEX.finditer(header.lower())
    }


def select_encoding(
    accept_encoding: str | None,
    default: CompressionEncoding = CompressionEncoding.GZIP,
) -> CompressionEncoding:
    """Select best compression based on Accept-Encoding header.

    Priority: zstd > gzip (if available and accepted by client).
    Respects quality values: q=0 means the encoding is explicitly rejected.

    Args:
        accept_encoding: The Accept-Encoding header value from HTTP request.
        default: Fallback encoding if no preferred encoding is accepted.

    Returns:
        Selected compression encoding.
    """
    if not accept_encoding:
        return default

    accepted = _parse_accept_encoding(accept_encoding)

    if accepted.get("zstd", 0) > 0 and is_zstd_available():
        return CompressionEncoding.ZSTD
    if accepted.get("gzip", 0) > 0:
        return CompressionEncoding.GZIP
    if accepted.get("identity", 1.0) > 0:
        return CompressionEncoding.IDENTITY

    return default


def _make_compressobj(encoding: CompressionEncoding) -> Any | None:
    """Create a streaming compressor for the given encoding, or None for identity."""
    match encoding:
        case CompressionEncoding.ZSTD:
            try:
                import zstandard
            except ImportError as e:
                raise ValueError(
                    f"zstd encoding requested but zstandard not installed: {e!r}. Please install `uv add zstandard` in order to use zstd compression."
                ) from e
            return zstandard.ZstdCompressor(
                level=Environment.COMPRESSION.ZSTD_LEVEL
            ).compressobj()
        case CompressionEncoding.GZIP:
            return zlib.compressobj(level=Environment.COMPRESSION.GZIP_LEVEL, wbits=31)
        case CompressionEncoding.IDENTITY:
            return None
        case _:
            raise ValueError(f"Unsupported encoding: {encoding}")


async def stream_file_compressed(
    file_path: pathlib.Path,
    encoding: CompressionEncoding,
    chunk_size: int = Environment.COMPRESSION.CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """Stream a file with the specified compression encoding.

    Args:
        file_path: Path to the file to compress.
        encoding: Compression encoding to use.
        chunk_size: Size of output chunks.

    Yields:
        Compressed data chunks.

    Raises:
        ValueError: If the encoding is not supported or library unavailable.
    """
    comp_obj = _make_compressobj(encoding)

    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(chunk_size):
            if comp_obj is not None:
                chunk = comp_obj.compress(chunk)
            if chunk:
                yield chunk

    if comp_obj is not None:
        final = comp_obj.flush()
        if final:
            yield final
