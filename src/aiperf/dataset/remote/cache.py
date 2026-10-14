# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-disk cache layout and crash-safe writers for remote dataset downloads."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import os
import re
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from pathlib import Path

import aiofiles
from filelock import FileLock

from aiperf.common.environment import Environment
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError, RemoteStage

_LOCK_TIMEOUT_S = 1800.0
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_VERSION_NAME_MAX_CHARS = 48
# Local filesystem failures a retry cannot fix (and re-downloading GBs would not help).
_LOCAL_FS_ERRNOS = frozenset(
    {errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES, errno.EPERM}
)


def remote_cache_dir() -> Path:
    """Root of the remote-dataset cache (``AIPERF_DATASET_REMOTE_CACHE_DIR`` or the default)."""
    configured = Environment.DATASET.REMOTE_CACHE_DIR
    if configured is not None:
        return Path(configured)
    return Path.home() / ".cache" / "aiperf" / "remote"


def local_fs_error(
    exc: BaseException,
    *,
    ref_display: str,
    stage: RemoteStage,
    cache: str | None = None,
    env_var: str = "AIPERF_DATASET_REMOTE_CACHE_DIR",
) -> RemoteSourceError | None:
    """The non-retried error for a cache that cannot be written; None for any other failure.

    ``cache`` describes the cache being written (default: AIPerf's remote cache) and
    ``env_var`` names the variable that moves it.
    """
    if not isinstance(exc, OSError) or exc.errno not in _LOCAL_FS_ERRNOS:
        return None
    cache = cache or f"the remote dataset cache at {remote_cache_dir()}"
    return RemoteSourceError(
        ref=ref_display, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
        what=f"{cache} cannot be written ({exc.strerror})",
        hint=f"Free disk space or fix permissions there, or set {env_var} to another "
        "location, then retry",
    )  # fmt: skip


def digest(value: str, length: int) -> str:
    """First ``length`` hex chars of the SHA-256 of ``value``; stable cache-key material."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _version_dir_name(version: str) -> str:
    """Filesystem-safe name for a version or ETag, including on Windows.

    ETags carry quotes, slashes and colons. The readable prefix is only a debugging
    aid; the digest of the raw value keeps distinct versions from colliding after
    sanitizing (``a:b`` and ``a/b`` both become ``a_b``).
    """
    readable = _UNSAFE_NAME_CHARS.sub("_", version)[:_VERSION_NAME_MAX_CHARS]
    return f"{readable}-{digest(version, 16)}"


def entry_dir(ref: RemoteRef) -> Path:
    """Directory holding the downloaded bytes of one exact version of ``ref``."""
    if ref.resolved_version is None:
        raise ValueError(f"{ref.display} has no resolved_version; probe it first")
    return (
        remote_cache_dir()
        / ref.scheme
        / digest(ref.canonical, 32)
        / _version_dir_name(ref.resolved_version)
    )


@contextlib.asynccontextmanager
async def entry_lock(entry: Path) -> AsyncIterator[None]:
    """Serialize concurrent downloads of the same cache entry across processes."""
    await asyncio.to_thread(entry.parent.mkdir, parents=True, exist_ok=True)
    # thread_local=False: acquire and release run on different to_thread workers.
    lock = FileLock(f"{entry}.lock", thread_local=False)
    await asyncio.to_thread(lock.acquire, timeout=_LOCK_TIMEOUT_S)
    try:
        yield
    finally:
        await asyncio.to_thread(lock.release)


def _partial_path(dest: Path) -> Path:
    return dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.partial")


async def stream_to_file(chunks: AsyncIterable[bytes], dest: Path) -> tuple[str, int]:
    """Write ``chunks`` to ``dest`` atomically; return (sha256 hex, size).

    An exception or cancellation mid-stream removes the partial file and leaves
    ``dest`` untouched, so a retry re-downloads instead of serving a truncated file.
    """
    await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
    tmp = _partial_path(dest)
    hasher = hashlib.sha256()
    size = 0
    try:
        async with aiofiles.open(tmp, "wb") as f:
            async for chunk in chunks:
                hasher.update(chunk)
                size += len(chunk)
                await f.write(chunk)
        await asyncio.to_thread(os.replace, tmp, dest)
    finally:
        # Synchronous on purpose: this also runs while handling CancelledError, and
        # an awaited cleanup could itself be cancelled and strand the partial file.
        # After a successful replace the temp name no longer exists.
        tmp.unlink(missing_ok=True)
    return hasher.hexdigest(), size


def write_atomically(chunks: Iterable[bytes], dest: Path) -> tuple[str, int]:
    """Blocking twin of ``stream_to_file`` for libraries with sync bodies (botocore)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial_path(dest)
    hasher = hashlib.sha256()
    size = 0
    try:
        with tmp.open("wb") as f:
            for chunk in chunks:
                hasher.update(chunk)
                size += len(chunk)
                f.write(chunk)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return hasher.hexdigest(), size
