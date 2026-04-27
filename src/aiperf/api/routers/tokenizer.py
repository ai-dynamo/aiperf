# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer router -- serves tar+zstd of HF snapshot dirs from the shared cache.

The api container has zero HF egress. Snapshots are populated by the
controller-pod warmer (``tokenizer_validator._prefetch_tokenizers``) writing
to the shared ``tokenizer-cache`` emptyDir volume mounted at ``HF_HOME``.
This router calls ``snapshot_download(local_files_only=True)`` against that
shared cache and streams the resulting directory back as a single
``application/zstd`` payload. Worker pods retry through 503s while the
warmer is still running.
"""

from __future__ import annotations

import asyncio
import io
import tarfile
from collections.abc import AsyncIterator
from pathlib import Path

import zstandard
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aiperf.common.environment import Environment

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


async def _resolve_snapshot_dir(name: str) -> Path:
    """Return the local snapshot dir for ``name`` from the shared HF cache.

    Returns 503 when the warmer hasn't populated the cache yet (worker pods
    retry through this) and 404 when the warmer asked for a name HF doesn't
    know about. Never reaches the network.
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
            detail=f"tokenizer '{name}' not found on HuggingFace Hub: {exc}",
        ) from exc
    return Path(path)


def build_tokenizer_router() -> APIRouter:
    """Return an APIRouter exposing ``GET /api/tokenizer/{name:path}/bundle``."""
    router = APIRouter(
        prefix="/api/tokenizer", tags=["Tokenizer"], include_in_schema=False
    )

    @router.get("/{name:path}/bundle")
    async def get_tokenizer_bundle(name: str) -> StreamingResponse:
        snapshot_dir = await _resolve_snapshot_dir(name)
        return StreamingResponse(
            _stream_tar_zstd(snapshot_dir), media_type="application/zstd"
        )

    return router
