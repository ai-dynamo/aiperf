# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming download helpers for :class:`ProgressClient`.

Split out of ``progress_client.py`` to keep that module under the ergonomics
file-size limit. These helpers stream aiohttp response bodies to disk with
optional zstd passthrough / transcoding.
"""

import zlib
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import zstandard as zstd

CHUNK_SIZE = 64 * 1024


def make_decompressor(content_encoding: str) -> Any:
    """Build a streaming decompressor for the given HTTP ``Content-Encoding``.

    Returns ``None`` for identity/unknown encodings so callers can skip the
    decompress step without branching on the encoding string.
    """
    if content_encoding == "zstd":
        return zstd.ZstdDecompressor().decompressobj()
    if content_encoding == "gzip":
        return zlib.decompressobj(wbits=31)
    return None


async def save_zstd_passthrough(
    response: aiohttp.ClientResponse, dest_path: Path
) -> None:
    """Stream a zstd-encoded response body verbatim to ``<name>.zst``."""
    zst_path = dest_path.parent / (dest_path.name + ".zst")
    async with aiofiles.open(zst_path, "wb") as f:
        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
            if chunk:
                await f.write(chunk)


async def save_transcoded_zstd(
    response: aiohttp.ClientResponse,
    dest_path: Path,
    decompressor: Any,
) -> None:
    """Decompress from wire encoding then re-compress as zstd for storage."""
    zst_path = dest_path.parent / (dest_path.name + ".zst")
    cctx = zstd.ZstdCompressor(level=3)
    compressor = cctx.compressobj()

    async with aiofiles.open(zst_path, "wb") as f:
        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
            if decompressor is not None:
                chunk = decompressor.decompress(chunk)
            if chunk:
                compressed = compressor.compress(chunk)
                if compressed:
                    await f.write(compressed)
        if decompressor is not None:
            remaining = decompressor.flush()
            if remaining:
                compressed = compressor.compress(remaining)
                if compressed:
                    await f.write(compressed)
        final = compressor.flush()
        if final:
            await f.write(final)


async def save_decompressed(
    response: aiohttp.ClientResponse,
    dest_path: Path,
    decompressor: Any,
) -> None:
    """Decompress wire encoding and save the raw bytes to ``dest_path``."""
    async with aiofiles.open(dest_path, "wb") as f:
        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
            if decompressor is not None:
                chunk = decompressor.decompress(chunk)
            if chunk:
                await f.write(chunk)
        if decompressor is not None:
            remaining = decompressor.flush()
            if remaining:
                await f.write(remaining)
