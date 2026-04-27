# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP download helper -- pulls tokenizer bundles from the operator API.

Mirrors ``worker_pod_dataset_download.download_dataset`` but for the tokenizer
endpoint. Each bundle is a single tar+zstd stream that gets decompressed and
untarred into ``{dest_root}/{slug(name)}/``. The slug is URL-quoted so on-disk
layout is debuggable from a shell into the pod.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import tarfile
from pathlib import Path
from urllib.parse import quote

import aiohttp
import zstandard

from aiperf.transports.aiohttp_client import create_tcp_connector

_INITIAL_BACKOFF_S = 0.5
_MAX_BACKOFF_S = 8.0


def slug_for_tokenizer(name: str) -> str:
    """URL-quote a tokenizer name into a single safe path segment."""
    return quote(name, safe="")


async def download_tokenizer(
    *,
    api_base_url: str,
    name: str,
    dest_root: Path,
    max_retries: int,
    logger: logging.Logger,
) -> Path:
    """Download and extract one tokenizer bundle. Returns the snapshot dir.

    Raises:
        RuntimeError: 404 from server, or retries exhausted.
    """
    base = api_base_url.rstrip("/")
    slug = slug_for_tokenizer(name)
    dest = dest_root / slug
    dest.mkdir(parents=True, exist_ok=True)

    # Per-bundle lock: first arrival downloads, others wait then read.
    lock_path = dest_root / f"{slug}.lock"
    sentinel = dest / ".ready"
    if sentinel.exists():
        return dest

    # Cooperative async lock; cross-container coordination uses fcntl below.
    async with _bundle_lock(lock_path):
        if sentinel.exists():
            return dest

        url = f"{base}/api/tokenizer/{name}/bundle"
        backoff = _INITIAL_BACKOFF_S
        last_exc: Exception | None = None
        async with aiohttp.ClientSession(connector=create_tcp_connector()) as session:
            for attempt in range(1, max_retries + 1):
                try:
                    async with session.get(url) as resp:
                        if resp.status == 404:
                            raise RuntimeError(
                                f"tokenizer '{name}' not registered on operator API "
                                f"(HTTP 404 from {url})"
                            )
                        if resp.status == 503:
                            logger.info(
                                f"tokenizer '{name}' not ready (503), "
                                f"attempt {attempt}/{max_retries}"
                            )
                            await asyncio.sleep(min(backoff, _MAX_BACKOFF_S))
                            backoff *= 2
                            continue
                        resp.raise_for_status()
                        compressed = await resp.read()
                    _extract_bundle(compressed, dest)
                    sentinel.write_text("ok")
                    return dest
                except aiohttp.ClientError as exc:
                    last_exc = exc
                    logger.warning(
                        f"transient error downloading tokenizer '{name}' "
                        f"({type(exc).__name__}: {exc}); attempt {attempt}/{max_retries}"
                    )
                    await asyncio.sleep(min(backoff, _MAX_BACKOFF_S))
                    backoff *= 2

        raise RuntimeError(
            f"failed to download tokenizer '{name}' after {max_retries} attempts: {last_exc}"
        )


def _extract_bundle(compressed: bytes, dest: Path) -> None:
    """Decompress zstd, untar in-memory into ``dest``.

    Uses ``stream_reader`` rather than ``decompress(buf)`` because the
    server-side compressor (``ZstdCompressor.stream_writer``) does not
    embed a content size in the frame header; ``decompress(buf)`` then
    raises "could not determine content size in frame header".
    """
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
        tar_bytes = reader.read()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        # ``filter="data"`` was added in Python 3.12 and becomes mandatory in
        # 3.14 (PEP 706). Pass it conditionally so we still run on 3.10/3.11.
        if sys.version_info >= (3, 12):
            tf.extractall(path=dest, filter="data")
        else:
            tf.extractall(path=dest)


class _bundle_lock:
    """Cross-container file lock + asyncio-friendly entry."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    async def __aenter__(self) -> _bundle_lock:
        import fcntl
        import os

        self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        # Acquire the lock in a worker thread so we don't block the loop.
        await asyncio.to_thread(fcntl.flock, self._fd, fcntl.LOCK_EX)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        import fcntl
        import os

        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
