# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming download helpers for :class:`ProgressClient`.

Split out of ``progress_client.py`` to keep that module under the ergonomics
file-size limit. These helpers stream aiohttp response bodies to disk with
optional zstd passthrough / transcoding.
"""

import os
import zlib
from pathlib import Path
from typing import Protocol

import aiofiles
import aiohttp
import zstandard as zstd

CHUNK_SIZE = 64 * 1024


class StreamingDecompressor(Protocol):
    """Minimal protocol for streaming decompressors used in response download.

    Satisfied by ``zstd.ZstdDecompressionObj`` and ``zlib.decompressobj()`` — both
    accept a ``bytes`` chunk and return the decompressed ``bytes``; both expose
    ``flush()`` to drain any buffered output after the last chunk. ``None`` is
    used for identity/unknown encodings (no decompression applied).
    """

    def decompress(self, chunk: bytes) -> bytes: ...
    def flush(self) -> bytes: ...


def make_decompressor(content_encoding: str) -> StreamingDecompressor | None:
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
    """Atomically stream a zstd-encoded response body to ``<name>.zst``."""
    zst_path = dest_path.parent / (dest_path.name + ".zst")
    part_path = zst_path.parent / (zst_path.name + ".part")
    try:
        async with aiofiles.open(part_path, "wb") as f:
            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                if chunk:
                    await f.write(chunk)
        os.replace(part_path, zst_path)
    finally:
        part_path.unlink(missing_ok=True)


async def save_transcoded_zstd(
    response: aiohttp.ClientResponse,
    dest_path: Path,
    decompressor: StreamingDecompressor | None,
) -> None:
    """Atomically decompress the wire body and re-compress it as zstd."""
    zst_path = dest_path.parent / (dest_path.name + ".zst")
    part_path = zst_path.parent / (zst_path.name + ".part")
    cctx = zstd.ZstdCompressor(level=3)
    compressor = cctx.compressobj()

    try:
        async with aiofiles.open(part_path, "wb") as f:
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
        os.replace(part_path, zst_path)
    finally:
        part_path.unlink(missing_ok=True)


async def save_decompressed(
    response: aiohttp.ClientResponse,
    dest_path: Path,
    decompressor: StreamingDecompressor | None,
) -> None:
    """Decompress wire encoding and save the raw bytes to ``dest_path``.

    Streams to a ``<name>.part`` sibling and ``os.replace``s onto the final
    name only after the body is fully written, so readers gated on file
    existence (e.g. the sweep-aggregate harvest checking ``aggregate.json``)
    can never observe a truncated file. On failure the partial ``.part`` file
    is removed and ``dest_path`` is left untouched.
    """
    part_path = dest_path.parent / (dest_path.name + ".part")
    try:
        async with aiofiles.open(part_path, "wb") as f:
            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                if decompressor is not None:
                    chunk = decompressor.decompress(chunk)
                if chunk:
                    await f.write(chunk)
            if decompressor is not None:
                remaining = decompressor.flush()
                if remaining:
                    await f.write(remaining)
        os.replace(part_path, dest_path)
    finally:
        part_path.unlink(missing_ok=True)
