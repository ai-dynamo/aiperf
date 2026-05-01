# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer router -- serves tar+zstd of HF snapshot dirs from the shared cache.

The api container has zero HF egress at request time. Snapshots are populated
by the api container's own ``_prewarm_tokenizers`` (which runs before uvicorn
binds), writing into the shared ``tokenizer-cache`` emptyDir mounted at
``HF_HOME``. This router calls ``snapshot_download(local_files_only=True)``
against that shared cache and streams the resulting directory back as a single
``application/zstd`` payload.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import zstandard
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aiperf.common.environment import Environment

_CHUNK_SIZE = 1 << 16  # 64 KiB


def _materialize_bundle(snapshot_dir: Path) -> bytes:
    """Build the full tar+zstd payload for ``snapshot_dir`` once."""
    import io as _io
    import tarfile as _tarfile

    cctx = zstandard.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
    with _io.BytesIO() as raw_tar:
        with _tarfile.open(fileobj=raw_tar, mode="w", dereference=True) as tar:
            for entry in sorted(snapshot_dir.iterdir()):
                tar.add(entry, arcname=entry.name)
        return cctx.compress(raw_tar.getvalue())


def _stream_bytes(payload: bytes) -> AsyncIterator[bytes]:
    async def _iter() -> AsyncIterator[bytes]:
        for i in range(0, len(payload), _CHUNK_SIZE):
            yield payload[i : i + _CHUNK_SIZE]

    return _iter()


async def _resolve_snapshot_dir(name: str) -> Path:
    """Return the local snapshot dir for ``name`` from the shared HF cache.

    Returns 503 when the cache is cold (worker pods retry through this) and
    404 when HF Hub doesn't recognise the name. Never reaches the network at
    request time.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        path = await asyncio.to_thread(
            snapshot_download,
            repo_id=name,
            repo_type="model",
            local_files_only=True,
        )
    except LocalEntryNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"tokenizer '{name}' not yet warmed in shared HF cache",
            headers={"Retry-After": "1"},
        ) from exc
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"tokenizer '{name}' not configured for this run",
        ) from exc
    return Path(path)


def build_tokenizer_router() -> APIRouter:
    """Return an APIRouter exposing ``GET /api/tokenizer/{name:path}/bundle``."""
    router = APIRouter(
        prefix="/api/tokenizer", tags=["Tokenizer"], include_in_schema=False
    )
    bundle_cache: dict[str, bytes] = {}
    cache_lock = asyncio.Lock()

    async def _get_bundle_bytes(name: str) -> bytes:
        cached = bundle_cache.get(name)
        if cached is not None:
            return cached
        async with cache_lock:
            cached = bundle_cache.get(name)
            if cached is not None:
                return cached
            snapshot_dir = await _resolve_snapshot_dir(name)
            payload = await asyncio.to_thread(_materialize_bundle, snapshot_dir)
            bundle_cache[name] = payload
            return payload

    @router.get("/{name:path}/bundle")
    async def get_tokenizer_bundle(name: str) -> StreamingResponse:
        payload = await _get_bundle_bytes(name)
        return StreamingResponse(_stream_bytes(payload), media_type="application/zstd")

    return router
