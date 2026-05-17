# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer router -- serves tar+zstd of HF snapshot dirs from the shared cache.

The api container has zero HF egress at request time. Snapshots are populated
by the api container's own ``_prewarm_tokenizers`` (which runs before uvicorn
binds), writing into the shared ``tokenizer-cache`` emptyDir mounted at
``HF_HOME``. ``_prewarm_tokenizers`` then calls :func:`prewarm_bundle` for
every configured name so the tar+zstd payload is fully materialised in this
process's RAM before the port binds. The bundle endpoint then serves
synchronously out of the module-level cache -- no per-request file IO,
tarring, or compression in the hot path.

If a request arrives for a name that prewarm did not cover (e.g. an out-of-band
tokenizer the operator did not pre-register), the endpoint falls back to an
on-demand build under a per-name lock so concurrent requests for the same name
do not stampede. The per-request build keeps the same on-disk semantics as
prewarm (resolve via ``snapshot_download(local_files_only=True)``, materialise
via :func:`_materialize_bundle`), so cold-cache requests still succeed -- they
just pay the materialisation cost on the request path instead of at startup.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import zstandard
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aiperf.common.environment import Environment

_CHUNK_SIZE = 1 << 16  # 64 KiB

# Allowlist of tokenizer-related filenames in an HF snapshot dir. Anything
# outside this set (notably ``*.safetensors`` / ``*.bin`` weight shards) is
# excluded from the bundle. The HF cache layout stores the *real* bytes in
# ``../../blobs/<sha>`` and exposes them as symlinks inside the snapshot dir,
# so a naive ``tar(..., dereference=True)`` over the whole snapshot inflates
# from a few hundred KB of symlinks to the full model size (>100 GB for
# Nemotron-3-Super-120B). The allowlist keeps the bundle bounded to actual
# tokenizer artefacts.
_TOKENIZER_ALLOW_NAMES: frozenset[str] = frozenset(
    {
        # Core HF tokenizer artefacts
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",  # SentencePiece
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "added_tokens.json",
        # Chat / generation behaviour the tokenizer surfaces via apply_chat_template
        "chat_template.jinja",
        "generation_config.json",
        # AutoTokenizer needs ``config.json`` to resolve the tokenizer class
        # (and any ``auto_map`` pointing at custom Python modules).
        "config.json",
    }
)
# Suffix allowlist for variable-name files: custom tokenizer Python (loaded
# under ``trust_remote_code`` / ``auto_map``) and tiktoken vocab dumps.
_TOKENIZER_ALLOW_SUFFIXES: frozenset[str] = frozenset({".py", ".tiktoken"})

# Hard cap on the uncompressed tar size. Real tokenizers are well under 30 MB;
# a bundle larger than this almost certainly means a non-tokenizer artefact
# slipped past the allowlist. We raise rather than silently truncate so the
# allowlist gets fixed instead of users getting a corrupt half-bundle.
_TOKENIZER_BUNDLE_MAX_BYTES: int = 50 * 1024 * 1024  # 50 MiB


def _is_tokenizer_file(entry: Path) -> bool:
    """Return True iff ``entry`` is on the tokenizer allowlist (case-insensitive)."""
    name_lower = entry.name.lower()
    if name_lower in _TOKENIZER_ALLOW_NAMES:
        return True
    return entry.suffix.lower() in _TOKENIZER_ALLOW_SUFFIXES


class _BundleStore:
    """Process-wide bundle state shared by prewarm and the route handler.

    Keyed by tokenizer ``name`` exactly as it appears on the URL.
    Populated by :func:`prewarm_bundle` (called from
    ``api_service._prewarm_tokenizers``) and read by the bundle endpoint.
    Held as a singleton so the prewarm path and the route handler share
    the same bytes regardless of how many ``build_tokenizer_router()``
    instances exist (in production there is exactly one; tests create more).
    """

    __slots__ = ("bundles", "locks", "locks_guard")

    def __init__(self) -> None:
        self.bundles: dict[str, bytes] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.locks_guard: asyncio.Lock = asyncio.Lock()

    async def lock_for(self, name: str) -> asyncio.Lock:
        """Return the per-name lock for ``name``, creating it under a guard.

        Per-name locking ensures one concurrent build per tokenizer; unrelated
        tokenizers do not serialise behind each other (which a single global
        cache lock would have caused).
        """
        async with self.locks_guard:
            lock = self.locks.get(name)
            if lock is None:
                lock = asyncio.Lock()
                self.locks[name] = lock
            return lock

    def reset(self) -> None:
        """Drop all cached bundles and per-name locks. Tests only."""
        self.bundles.clear()
        self.locks.clear()


_store = _BundleStore()


def _materialize_bundle(snapshot_dir: Path) -> bytes:
    """Build the tar+zstd payload for ``snapshot_dir``, tokenizer files only.

    Walks ``snapshot_dir`` and adds only entries on the tokenizer allowlist
    (see :data:`_TOKENIZER_ALLOW_NAMES` / :data:`_TOKENIZER_ALLOW_SUFFIXES`).
    Weight shards (``*.safetensors``, ``*.bin``, ``*.pth``, ``*.gguf``,
    ``*.onnx``) are intentionally excluded -- in production the snapshot dir
    is a HuggingFace cache mount where these are symlinks into ``../../blobs/``,
    so ``dereference=True`` over the whole dir would balloon the bundle to the
    full model size. Allowlisted files are still dereferenced (HF cache stores
    even ``tokenizer.json`` as a blob symlink, so we need the real bytes).

    Raises ``ValueError`` if the assembled tar exceeds
    :data:`_TOKENIZER_BUNDLE_MAX_BYTES` -- a guard against a non-tokenizer
    artefact slipping past the allowlist.
    """
    import io as _io
    import tarfile as _tarfile

    cctx = zstandard.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
    with _io.BytesIO() as raw_tar:
        with _tarfile.open(fileobj=raw_tar, mode="w", dereference=True) as tar:
            for entry in sorted(snapshot_dir.iterdir()):
                if not _is_tokenizer_file(entry):
                    continue
                tar.add(entry, arcname=entry.name)
        raw_size = raw_tar.tell()
        if raw_size > _TOKENIZER_BUNDLE_MAX_BYTES:
            raise ValueError(
                f"tokenizer bundle for '{snapshot_dir}' is {raw_size} bytes, "
                f"exceeds cap of {_TOKENIZER_BUNDLE_MAX_BYTES} bytes; "
                f"a non-tokenizer artefact likely slipped past the allowlist "
                f"in aiperf.api.routers.tokenizer._TOKENIZER_ALLOW_NAMES / "
                f"_TOKENIZER_ALLOW_SUFFIXES"
            )
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


async def prewarm_bundle(name: str) -> None:
    """Materialise ``name``'s tar+zstd payload into the module-level cache.

    Called by ``api_service._prewarm_tokenizers`` after
    ``AutoTokenizer.from_pretrained`` has populated HF_HOME for ``name``.
    Idempotent: a second call for the same name is a no-op once cached.

    Failures here are intentionally caught and logged by the caller -- a
    failed prewarm leaves the slot empty, and the request path will rebuild
    on demand (or surface a clear 503/404). This guarantees the api server
    can still bind even when one tokenizer's snapshot is malformed.
    """
    if name in _store.bundles:
        return
    lock = await _store.lock_for(name)
    async with lock:
        if name in _store.bundles:
            return
        snapshot_dir = await _resolve_snapshot_dir(name)
        payload = await asyncio.to_thread(_materialize_bundle, snapshot_dir)
        _store.bundles[name] = payload


def reset_cache_for_tests() -> None:
    """Drop the module-level bundle cache. Tests only.

    Production code never calls this -- the cache is bounded by the number
    of distinct tokenizer names in the run config (typically one) and lives
    until process exit. Tests need a clean slate between cases because the
    cache is module-level.
    """
    _store.reset()


def build_tokenizer_router() -> APIRouter:
    """Return an APIRouter exposing ``GET /api/tokenizer/{name:path}/bundle``."""
    router = APIRouter(
        prefix="/api/tokenizer", tags=["Tokenizer"], include_in_schema=False
    )

    async def _get_bundle_bytes(name: str) -> bytes:
        cached = _store.bundles.get(name)
        if cached is not None:
            return cached
        # Cold-cache fallback: a name that prewarm didn't cover, or arrived
        # before prewarm finished. Serialise concurrent requesters per-name
        # so we don't tar+compress N times in parallel.
        lock = await _store.lock_for(name)
        t0 = time.monotonic()
        async with lock:
            cached = _store.bundles.get(name)
            if cached is not None:
                return cached
            snapshot_dir = await _resolve_snapshot_dir(name)
            payload = await asyncio.to_thread(_materialize_bundle, snapshot_dir)
            _store.bundles[name] = payload
            elapsed = time.monotonic() - t0
            if elapsed > 5.0:
                # Surface slow on-demand builds in the api log; production
                # should always hit the prewarmed fast path, so a slow
                # request-path build is a signal that prewarm missed this
                # name (or failed silently).
                import logging

                logging.getLogger("aiperf.api.tokenizer").warning(
                    "tokenizer bundle '%s' built on request path in %.1fs "
                    "(%d bytes); prewarm should have covered this name",
                    name,
                    elapsed,
                    len(payload),
                )
            return payload

    @router.get("/{name:path}/bundle")
    async def get_tokenizer_bundle(name: str) -> StreamingResponse:
        payload = await _get_bundle_bytes(name)
        return StreamingResponse(_stream_bytes(payload), media_type="application/zstd")

    return router
