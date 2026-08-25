# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared download primitives for the operator results-server clients.

A leaf module: ``results_operator`` imports ``results_operator_sweeps``, so
anything both need lives here rather than in either of them.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import aiofiles
import aiohttp

from aiperf.common.environment import Environment
from aiperf.kubernetes.console import print_error

__all__ = [
    "RESULTS_SERVER_PORT",
    "_REDIRECT_STATUSES",
    "_JobDownloadOutcome",
    "_download_and_decompress",
    "_get_no_redirects",
    "_is_refused_name",
    "_results_server_port",
    "_verify_operator_health",
]


def _results_server_port() -> int:
    """Read the configured operator results-server port."""
    return int(os.environ.get("AIPERF_RESULTS_SERVER_PORT", "8081"))


RESULTS_SERVER_PORT = _results_server_port()
_REDIRECT_STATUSES = {301, 302, 307, 308}


@dataclass(frozen=True, slots=True)
class _JobDownloadOutcome:
    """Result of downloading every advertised file for one run."""

    downloaded: list[tuple[str, int]]
    """(display name, size in bytes) for each file that landed on disk."""

    failed: list[str]
    """Display names the server advertised but did not deliver."""

    @property
    def complete(self) -> bool:
        """True when every advertised file was retrieved."""
        return not self.failed


def _is_refused_name(display_name: str) -> bool:
    """True when we decline to write this name, regardless of the server.

    Dot-files (the results-ready marker among them), absolute paths and
    parent traversals are refused by policy. They are advertised in listings
    but never downloaded, so they are skips rather than failures.
    """
    normalized = Path(display_name)
    leaf = normalized.name
    return (
        not leaf
        or leaf.startswith(".")
        or normalized.is_absolute()
        or ".." in normalized.parts
    )


def _get_no_redirects(
    session: aiohttp.ClientSession,
    url: str,
    **kwargs: object,
) -> object:
    """Start a GET request without redirects, including test-double fallback."""
    try:
        return session.get(url, allow_redirects=False, **kwargs)
    except TypeError as e:
        if "allow_redirects" not in str(e):
            raise
        return session.get(url, **kwargs)


async def _download_and_decompress(
    response: aiohttp.ClientResponse,
    dest_path: Path,
    content_encoding: str,
) -> None:
    """Stream an optionally encoded response to ``dest_path`` atomically."""
    import zlib

    if content_encoding == "zstd":
        import zstandard

        decompressor = zstandard.ZstdDecompressor().decompressobj()
    elif content_encoding == "gzip":
        decompressor = zlib.decompressobj(wbits=31)
    else:
        decompressor = None

    temp_path = dest_path.with_name(f".{dest_path.name}.{uuid.uuid4().hex}.tmp")
    replaced = False
    try:
        async with aiofiles.open(temp_path, "wb") as file:
            async for chunk in response.content.iter_chunked(
                Environment.COMPRESSION.CHUNK_SIZE
            ):
                if decompressor is not None:
                    chunk = decompressor.decompress(chunk)
                if chunk:
                    await file.write(chunk)
            if decompressor is not None:
                remaining = decompressor.flush()
                if remaining:
                    await file.write(remaining)
        await asyncio.to_thread(os.replace, temp_path, dest_path)
        replaced = True
    finally:
        if not replaced:
            await asyncio.to_thread(temp_path.unlink, missing_ok=True)


async def _verify_operator_health(api_base: str, timeout_seconds: float = 10) -> bool:
    """Return whether the operator results server passes its health endpoint."""
    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        try:
            async with session.get(f"{api_base}/healthz") as response:
                if response.status != 200:
                    print_error("Operator results server not healthy")
                    return False
        except aiohttp.ClientError as e:
            print_error(f"Could not connect to operator results server: {e}")
            return False
    return True
