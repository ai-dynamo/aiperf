# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset download helpers for the WorkerGroupManager.

Extracted from ``worker_pod_manager`` to keep that module within the
ergonomics file-size limit. The functions here speak only to ``aiohttp`` and
the local filesystem — they have no knowledge of the pod-lifecycle router or
worker coordination, and are invoked from ``WorkerGroupManagerBase`` when a
dataset-configured notification arrives.
"""

from __future__ import annotations

import asyncio
import tempfile
import zlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import aiofiles
import aiohttp
import zstandard

from aiperf.common.environment import Environment
from aiperf.transports.aiohttp_client import create_tcp_connector

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class _DownloadLogger(Protocol):
    """Structural protocol matching logging methods used on BaseComponentService."""

    def info(self, msg: str) -> None: ...
    def debug(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...


DownloadFileCallable = Callable[[aiohttp.ClientSession, str, Path], Awaitable[None]]


def placeholder_local_paths(run: BenchmarkRun) -> tuple[Path, Path]:
    """Return the placeholder local dataset paths used on download failure."""
    cfg = run.cfg
    mmap_base = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
    local_dir = mmap_base / f"aiperf_mmap_{cfg.benchmark_id}"
    return local_dir / "dataset.dat", local_dir / "index.dat"


async def download_dataset(
    run: BenchmarkRun,
    logger: _DownloadLogger,
    download_file: DownloadFileCallable | None = None,
) -> tuple[Path, Path]:
    """Download the dataset from the control-plane API with retry.

    The dataset is downloaded once and saved to local storage (emptyDir volume).
    Workers then mmap the file for fast access. Retries with exponential
    backoff on transient network failures.

    Returns:
        Tuple of (data_path, index_path) where files were saved.

    Raises:
        RuntimeError: If download fails after all retries or dataset_api_base_url is not set.
    """
    cfg = run.cfg
    if not cfg.runtime.dataset_api_base_url:
        raise RuntimeError(
            "No dataset_api_base_url configured. "
            "WorkerGroupManager requires this to download the dataset."
        )

    base_url = cfg.runtime.dataset_api_base_url.rstrip("/")
    logger.info(f"Downloading dataset from {base_url}")

    mmap_base = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
    local_dir = mmap_base / f"aiperf_mmap_{cfg.benchmark_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    data_path = local_dir / "dataset.dat"
    index_path = local_dir / "index.dat"
    logger.info(f"Saving dataset to {local_dir}")

    max_retries = Environment.DATASET.DOWNLOAD_MAX_RETRIES
    retry_delay = Environment.DATASET.DOWNLOAD_RETRY_DELAY
    last_error: Exception | None = None

    # Allow callers (e.g. the WorkerGroupManager) to inject a bound
    # ``_download_file`` so per-instance patching in tests still runs.
    async def _call_download_file(
        session: aiohttp.ClientSession, url: str, dest_path: Path
    ) -> None:
        if download_file is not None:
            await download_file(session, url, dest_path)
        else:
            await _download_file(session, url, dest_path, logger)

    for attempt in range(max_retries + 1):
        try:
            connector = create_tcp_connector()
            async with aiohttp.ClientSession(connector=connector) as session:
                await asyncio.gather(
                    _call_download_file(session, f"{base_url}/data", data_path),
                    _call_download_file(session, f"{base_url}/index", index_path),
                )
            logger.info(
                f"Dataset download complete: data={data_path.stat().st_size} bytes, "
                f"index={index_path.stat().st_size} bytes"
            )
            return data_path, index_path
        except (aiohttp.ClientError, RuntimeError) as e:
            last_error = e
            if attempt < max_retries:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"Dataset download attempt {attempt + 1}/{max_retries + 1} failed: {e!r}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

    raise RuntimeError(
        f"Dataset download failed after {max_retries + 1} attempts"
    ) from last_error


async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest_path: Path,
    logger: _DownloadLogger,
) -> None:
    """Download a file from HTTP to local path with compression support.

    Requests compressed transfer via Accept-Encoding. aiohttp auto-decompresses
    gzip; zstd is handled manually.
    """
    logger.debug(f"Downloading {url} -> {dest_path}")
    headers = {"Accept-Encoding": "zstd, gzip"}
    try:
        async with session.get(url, headers=headers, auto_decompress=False) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to download {url}: HTTP {response.status}")
            content_encoding = response.headers.get("Content-Encoding", "").lower()
            logger.debug(f"Response encoding: {content_encoding or 'none'}")
            await _stream_response(response, dest_path, content_encoding)
        logger.debug(f"Downloaded {dest_path.stat().st_size} bytes to {dest_path}")
    except aiohttp.ClientError as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e


async def _stream_response(
    response: aiohttp.ClientResponse,
    dest_path: Path,
    content_encoding: str,
) -> None:
    """Stream response to file, decompressing if needed."""
    if content_encoding == "zstd":
        dctx = zstandard.ZstdDecompressor()
        decompressor = dctx.decompressobj()
    elif content_encoding == "gzip":
        decompressor = zlib.decompressobj(wbits=31)
    else:
        decompressor = None

    async with aiofiles.open(dest_path, "wb") as f:
        async for chunk in response.content.iter_chunked(
            Environment.COMPRESSION.CHUNK_SIZE
        ):
            if decompressor is not None:
                chunk = decompressor.decompress(chunk)
            if chunk:
                await f.write(chunk)
        # zlib decompressobj has flush(); zstandard decompressobj does not
        if decompressor is not None and hasattr(decompressor, "flush"):
            remaining = decompressor.flush()
            if remaining:
                await f.write(remaining)
