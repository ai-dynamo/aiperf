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
import os
import shutil
import tarfile
from pathlib import Path
from urllib.parse import quote

import aiohttp
import zstandard

from aiperf.common.constants import IS_WINDOWS
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

    Extraction is crash-atomic: the tar is unpacked into ``{slug}.tmp/``,
    a ``.ready`` sentinel is written inside, then the directory is renamed
    to ``{slug}/`` via ``os.replace``. A crash mid-extraction leaves the
    tmp dir behind (cleaned up on next retry) but no half-populated final
    dir; readers always see a fully-extracted bundle or nothing.

    Raises:
        RuntimeError: 404 from server, or retries exhausted.
    """
    base = api_base_url.rstrip("/")
    url = f"{base}/api/tokenizer/{name}/bundle"
    logger.info(f"download_tokenizer: starting for '{name}' from {url}")
    slug = slug_for_tokenizer(name)
    dest = dest_root / slug
    dest_root.mkdir(parents=True, exist_ok=True)
    sentinel = dest / ".ready"
    if sentinel.exists():
        logger.info(f"download_tokenizer: '{name}' already extracted at {dest}")
        return dest

    lock_path = dest_root / f"{slug}.lock"
    logger.info(f"download_tokenizer: acquiring bundle lock at {lock_path}")
    async with _bundle_lock(lock_path):
        logger.info(f"download_tokenizer: lock acquired for '{name}'")
        if sentinel.exists():
            return dest

        backoff = _INITIAL_BACKOFF_S
        last_exc: Exception | None = None
        # 5-minute hard ceiling per request so we never hang forever on a
        # broken connection — backoff loop still handles 503/transient
        # failures with retries.
        request_timeout = aiohttp.ClientTimeout(total=300.0)
        async with aiohttp.ClientSession(
            connector=create_tcp_connector(), timeout=request_timeout
        ) as session:
            for attempt in range(1, max_retries + 1):
                try:
                    compressed = await _fetch_bundle(
                        session,
                        url,
                        name=name,
                        attempt=attempt,
                        max_retries=max_retries,
                        logger=logger,
                    )
                    if compressed is None:
                        # Bundle not ready yet (503) — back off and retry.
                        await asyncio.sleep(min(backoff, _MAX_BACKOFF_S))
                        backoff *= 2
                        continue
                    logger.info(
                        f"download_tokenizer: '{name}' fetched "
                        f"({len(compressed)} bytes), extracting atomically"
                    )
                    tmp_dest = dest_root / f"{slug}.tmp"
                    if tmp_dest.exists():
                        shutil.rmtree(tmp_dest)
                    tmp_dest.mkdir(parents=True)
                    try:
                        _extract_bundle(compressed, tmp_dest)
                        (tmp_dest / ".ready").write_text("ok")
                    except BaseException:
                        # Clean up the partial tmp dir; final dest is untouched.
                        shutil.rmtree(tmp_dest, ignore_errors=True)
                        raise
                    # Atomic swap; survives crashes on either side of the rename.
                    if dest.exists():
                        shutil.rmtree(dest)
                    os.replace(tmp_dest, dest)
                    logger.info(f"download_tokenizer: '{name}' ready at {dest}")
                    return dest
                except (TimeoutError, aiohttp.ClientError) as exc:
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


async def _fetch_bundle(
    session: aiohttp.ClientSession,
    url: str,
    *,
    name: str,
    attempt: int,
    max_retries: int,
    logger: logging.Logger,
) -> bytes | None:
    """Fetch the compressed bundle bytes for one attempt.

    Returns None when the server replies 503 (bundle not ready yet) so the
    caller can back off and retry.

    Raises:
        RuntimeError: 404 from server (tokenizer not registered).
        aiohttp.ClientError: Non-OK status or transport failure.
    """
    async with session.get(url) as resp:
        if resp.status == 404:
            raise RuntimeError(
                f"tokenizer '{name}' not registered on operator API "
                f"(HTTP 404 from {url})"
            )
        if resp.status == 503:
            logger.info(
                f"tokenizer '{name}' not ready (503), attempt {attempt}/{max_retries}"
            )
            return None
        resp.raise_for_status()
        return await resp.read()


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
        _safe_extractall(tf, dest)


def _safe_extractall(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract every member into ``dest``, rejecting path-traversal escapes.

    A malicious tokenizer bundle could carry a member named ``../evil`` (or an
    absolute path, or an escaping symlink) to write outside ``dest`` — the
    classic tar-slip (CWE-22). PEP 706's ``data`` filter blocks all of these;
    it was backported to 3.11.4+/3.12+ and becomes the default in 3.14, so on
    every currently-shipped interpreter (requires-python >=3.11) we extract
    through ``filter="data"``.

    The handful of 3.11.0-3.11.3 patch releases predate the backport
    (``tarfile.data_filter`` absent). Rather than fall back to an unfiltered
    ``extractall`` there, validate each member's resolved path stays under
    ``dest`` first, so no supported interpreter ever runs an unguarded extract.
    """
    if hasattr(tarfile, "data_filter"):
        tf.extractall(path=dest, filter="data")
        return

    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise tarfile.TarError(
                f"tar member '{member.name}' escapes extraction dir {dest}"
            )
        if member.issym() or member.islnk():
            link_target = (
                (dest / member.name).parent.joinpath(member.linkname).resolve()
            )
            if (
                link_target != dest_resolved
                and dest_resolved not in link_target.parents
            ):
                raise tarfile.TarError(
                    f"tar link member '{member.name}' -> '{member.linkname}' "
                    f"escapes extraction dir {dest}"
                )
    tf.extractall(path=dest)


class _bundle_lock:
    """Cross-container file lock + asyncio-friendly entry.

    On Windows this is a no-op: the flock exists only to serialize bundle
    extraction across containers sharing a volume inside Linux worker pods,
    a topology that cannot occur on Windows (dev/CI runs only), and the
    ``fcntl`` module does not exist there.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    async def __aenter__(self) -> _bundle_lock:
        if IS_WINDOWS:
            return self
        import fcntl

        self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        # Acquire the lock in a worker thread so we don't block the loop.
        await asyncio.to_thread(fcntl.flock, self._fd, fcntl.LOCK_EX)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._fd is not None:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
