# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer router -- streams tar+zstd of HF snapshot dirs to worker pods.

Mirrors ``DatasetRouter`` in shape: a single GET endpoint that streams a
compressed binary representation of the artefact. The tar uses
``dereference=True`` so HF snapshot symlinks (snapshot-file -> blob) become
real files in the bundle.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import AsyncIterator
from pathlib import Path

import zstandard
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aiperf.common.environment import Environment
from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry

_CHUNK_SIZE = 1 << 16  # 64 KiB


def _stream_tar_zstd(snapshot_dir: Path) -> AsyncIterator[bytes]:
    """Yield zstd-compressed tar chunks of ``snapshot_dir`` contents."""

    async def _iter() -> AsyncIterator[bytes]:
        cctx = zstandard.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
        buf = io.BytesIO()
        with (
            cctx.stream_writer(buf, closefd=False) as zwriter,
            tarfile.open(fileobj=zwriter, mode="w|", dereference=True) as tar,
        ):
            for entry in sorted(snapshot_dir.iterdir()):
                tar.add(entry, arcname=entry.name)
                while True:
                    data = buf.getvalue()
                    if not data:
                        break
                    buf.seek(0)
                    buf.truncate(0)
                    for i in range(0, len(data), _CHUNK_SIZE):
                        yield data[i : i + _CHUNK_SIZE]
        # Flush any tail bytes that landed after the last yield.
        tail = buf.getvalue()
        if tail:
            for i in range(0, len(tail), _CHUNK_SIZE):
                yield tail[i : i + _CHUNK_SIZE]

    return _iter()


def build_tokenizer_router(registry: TokenizerBundleRegistry) -> APIRouter:
    """Return an APIRouter exposing ``GET /api/tokenizer/{name:path}/bundle``."""
    router = APIRouter(
        prefix="/api/tokenizer", tags=["Tokenizer"], include_in_schema=False
    )

    @router.get("/{name:path}/bundle")
    async def get_tokenizer_bundle(name: str) -> StreamingResponse:
        entry = registry.get(name)
        if entry is None:
            raise HTTPException(
                status_code=404, detail=f"tokenizer '{name}' not registered"
            )
        snapshot_dir, ready = entry
        if not ready.is_set() or snapshot_dir is None:
            raise HTTPException(
                status_code=503,
                detail=f"tokenizer '{name}' not yet ready",
                headers={"Retry-After": "1"},
            )
        return StreamingResponse(
            _stream_tar_zstd(snapshot_dir), media_type="application/zstd"
        )

    return router
