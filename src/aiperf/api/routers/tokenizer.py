# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer router -- on-demand HF snapshot fetch + tar+zstd streaming.

The api container is the only network egress for tokenizers; worker pods
must never hit huggingface.co. When a worker requests a bundle, this router
calls ``snapshot_download`` with a tokenizer-only ``allow_patterns`` filter,
tars the resulting snapshot directory, and streams it back as
``application/zstd``.

We do NOT coordinate with the controller-pod warmer. The api container
runs in its own process and would not see a module-level registry that the
control-plane container populated. Going straight to HF on demand keeps the
two halves decoupled and avoids cross-container synchronization. The HF
on-disk cache amortizes repeat requests within a single api-container
lifetime.
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

# Tokenizer-only file patterns: tokenizer.json, vocab.{json,txt}, merges.txt,
# special_tokens_map.json, added_tokens.json, tokenizer_config.json,
# chat_template.jinja, sentencepiece *.model, tiktoken *.tiktoken, and *.py
# for trust_remote_code modules. Excludes weights.
_TOKENIZER_ALLOW_PATTERNS: list[str] = [
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.jinja",
    "*.py",
]


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
    """Return a local snapshot dir for ``name``, downloading from HF if needed.

    Runs the blocking ``snapshot_download`` in a worker thread.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        path = await asyncio.to_thread(
            snapshot_download,
            repo_id=name,
            repo_type="model",
            allow_patterns=_TOKENIZER_ALLOW_PATTERNS,
        )
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
