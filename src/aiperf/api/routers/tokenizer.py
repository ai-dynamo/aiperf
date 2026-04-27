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
from collections.abc import AsyncIterator
from pathlib import Path

import zstandard
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aiperf.common.environment import Environment

_CHUNK_SIZE = 1 << 16  # 64 KiB


def _materialize_bundle(snapshot_dir: Path) -> bytes:
    """Build the full tar+zstd payload for ``snapshot_dir`` once.

    The bundles are small (tokenizer files only — single-digit MB
    compressed), so it's cheaper to materialize once into RAM and serve
    every subsequent worker-pod request from the cached bytes than to
    re-walk the directory and re-tar/re-compress per request.
    """
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
    # Per-name bundle cache: materialize tar+zstd once per tokenizer, serve
    # subsequent worker-pod requests from RAM. Bounded by the number of
    # distinct tokenizers in the run config (typically 1, never more than a
    # handful), so no eviction policy is needed.
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
