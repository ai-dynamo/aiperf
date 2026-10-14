# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming HTTP(S) helpers for remote dataset sources, with origin-checked redirects."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path

import aiohttp
from yarl import URL

from aiperf.common.url_origin import same_origin, strip_credentials_for_foreign_origin
from aiperf.dataset.remote.cache import local_fs_error, stream_to_file
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError, RemoteStage
from aiperf.transports.aiohttp_client import create_tcp_connector

MAX_REDIRECTS = 10
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# Credential-free headers a redirected request still needs (HF resolve URLs redirect to a
# CDN; the token must not go with them but the byte range must).
_CROSS_ORIGIN_EXTRA_HEADERS = frozenset({"range", "accept", "accept-encoding"})
_HEAD_UNSUPPORTED_STATUSES = frozenset({401, 403, 405, 501})
_PREFIX_CHUNK = 64 * 1024
_DOWNLOAD_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class HttpMetadata:
    """Version-relevant response headers of a remote object."""

    etag: str | None
    """The ETag header, quotes included."""
    last_modified: str | None
    """The Last-Modified header."""
    content_length: int | None
    """The Content-Length header, when the server sent one."""


@contextlib.asynccontextmanager
async def open_session() -> AsyncIterator[aiohttp.ClientSession]:
    """Yield a session that reads bodies byte-for-byte as published, with no total timeout."""
    # No total timeout: multi-GB downloads legitimately run long; stalls trip sock_read.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    # Object stores send a stored Content-Encoding whatever the client accepts; decoding
    # it would cache bytes that no longer match the manifest's sha256 and size.
    async with aiohttp.ClientSession(
        connector=create_tcp_connector(),
        timeout=timeout,
        trust_env=True,
        headers={"Accept-Encoding": "identity"},
        auto_decompress=False,
    ) as session:
        yield session


def _transport_error(
    exc: BaseException, *, ref_display: str, stage: RemoteStage
) -> RemoteSourceError:
    return RemoteSourceError(
        ref=ref_display,
        kind=RemoteErrorKind.TRANSIENT,
        stage=stage,
        what=f"the connection failed ({type(exc).__name__}: {exc})",
        hint="Check network access and retry",
    )


def status_error(
    status: int, *, ref_display: str, stage: RemoteStage
) -> RemoteSourceError | None:
    """Map an HTTP status to the error for it, or None for a 2xx."""
    if 200 <= status < 300:
        return None
    if status in (401, 403):
        return RemoteSourceError(
            ref=ref_display, kind=RemoteErrorKind.AUTH_DENIED, stage=stage,
            what=f"the server denied access (HTTP {status})",
            hint="If this is a presigned URL it may have expired; generate a new one",
        )  # fmt: skip
    if status == 429 or status >= 500:
        return RemoteSourceError(
            ref=ref_display, kind=RemoteErrorKind.TRANSIENT, stage=stage,
            what=f"the server is unavailable (HTTP {status})", hint="Retry later",
        )  # fmt: skip
    return RemoteSourceError(
        ref=ref_display, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
        what=f"the server has no such file (HTTP {status})", hint="Check the URL",
    )  # fmt: skip


def content_range_total(value: str | None) -> int | None:
    """Total object size from a ``Content-Range`` value, None when unknown (``*``) or absent."""
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1]
    return int(total) if total.isdigit() else None


def _metadata(response: aiohttp.ClientResponse) -> HttpMetadata:
    length = response.headers.get("Content-Length")
    return HttpMetadata(
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        content_length=int(length) if length and length.isdigit() else None,
    )


def _redirect_target(
    response: aiohttp.ClientResponse,
    location: str,
    *,
    ref_display: str,
    stage: RemoteStage,
) -> str:
    """Resolve a ``Location`` header, rejecting unparseable and non-HTTP targets.

    The target is left out of the errors: it can be a presigned URL.
    """
    try:
        target = response.url.join(URL(location))
        scheme = target.scheme
    except ValueError:
        scheme = None
    if scheme is None:
        what = "it redirected to an invalid URL"
    elif scheme not in ("http", "https"):
        what = "it redirected to a non-HTTP URL"
    else:
        return str(target)
    raise RemoteSourceError(
        ref=ref_display,
        kind=RemoteErrorKind.NOT_FOUND,
        stage=stage,
        what=what,
        hint="Check the URL",
    )


@contextlib.asynccontextmanager
async def safe_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    ref_display: str,
    stage: RemoteStage,
) -> AsyncIterator[aiohttp.ClientResponse]:
    """Follow redirects by hand, dropping credential headers whenever the origin changes."""
    current, current_method, current_headers = url, method, dict(headers)
    for _ in range(MAX_REDIRECTS + 1):
        try:
            response = await session.request(
                current_method, current, headers=current_headers, allow_redirects=False
            )
        except (TimeoutError, aiohttp.ClientError) as e:
            raise _transport_error(e, ref_display=ref_display, stage=stage) from e
        location = response.headers.get("Location")
        if response.status in _REDIRECT_STATUSES and location:
            response.release()
            target = _redirect_target(
                response, location, ref_display=ref_display, stage=stage
            )
            if not same_origin(target, current):
                current_headers = strip_credentials_for_foreign_origin(
                    current_headers, extra_allowed=_CROSS_ORIGIN_EXTRA_HEADERS
                )
            if response.status == 303:
                current_method = "GET"
            current = target
            continue
        try:
            yield response
        finally:
            response.release()
        return
    raise RemoteSourceError(
        ref=ref_display, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
        what=f"it redirected more than {MAX_REDIRECTS} times", hint="Check the URL",
    )  # fmt: skip


async def fetch_metadata(
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: dict[str, str],
    ref_display: str,
) -> HttpMetadata:
    """HEAD the object and return its version headers.

    Presigned URLs (SigV4, GCS V4) sign the HTTP method, so a URL presigned for GET
    answers HEAD with 401/403; some servers answer 405/501. In those cases one
    ``Range: bytes=0-0`` GET supplies the same headers, and its body is never read.
    """
    stage = RemoteStage.ACCESS
    async with safe_request(
        session, "HEAD", url, headers=headers, ref_display=ref_display, stage=stage
    ) as resp:
        if resp.status not in _HEAD_UNSUPPORTED_STATUSES:
            if (
                err := status_error(resp.status, ref_display=ref_display, stage=stage)
            ) is not None:
                raise err
            return _metadata(resp)
    async with safe_request(
        session,
        "GET",
        url,
        headers={**headers, "Range": "bytes=0-0"},
        ref_display=ref_display,
        stage=stage,
    ) as resp:
        if (
            err := status_error(resp.status, ref_display=ref_display, stage=stage)
        ) is not None:
            raise err
        meta = _metadata(resp)
        if resp.status == 206:
            # Content-Length on a 206 is the slice length, not the object size.
            meta = replace(
                meta,
                content_length=content_range_total(resp.headers.get("Content-Range")),
            )
        return meta


async def read_prefix(
    session: aiohttp.ClientSession,
    url: str,
    nbytes: int,
    *,
    headers: dict[str, str],
    ref_display: str,
) -> tuple[bytes, bool]:
    """Read up to ``nbytes`` from the start; the bool is True when that is the whole object."""
    stage = RemoteStage.ACCESS
    request_headers = {**headers, "Range": f"bytes=0-{nbytes - 1}"}
    async with safe_request(
        session,
        "GET",
        url,
        headers=request_headers,
        ref_display=ref_display,
        stage=stage,
    ) as resp:
        if resp.status == 416:  # Range not satisfiable: the object is empty.
            return b"", True
        if (
            err := status_error(resp.status, ref_display=ref_display, stage=stage)
        ) is not None:
            raise err
        buf = bytearray()
        try:
            async for chunk in resp.content.iter_chunked(_PREFIX_CHUNK):
                buf += chunk
                if len(buf) > nbytes:
                    break
        except (TimeoutError, aiohttp.ClientError) as e:
            raise _transport_error(e, ref_display=ref_display, stage=stage) from e
        if resp.status == 206:
            total = content_range_total(resp.headers.get("Content-Range"))
            return bytes(buf[:nbytes]), total is not None and total <= nbytes
        # Server ignored Range: we stopped early unless the body ended within nbytes.
        return bytes(buf[:nbytes]), len(buf) <= nbytes


async def download_to(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    *,
    headers: dict[str, str],
    ref_display: str,
) -> tuple[str, int, HttpMetadata]:
    """Stream the object to ``dest`` atomically; return (sha256, size, metadata)."""
    stage = RemoteStage.MATERIALIZE
    async with safe_request(
        session, "GET", url, headers=headers, ref_display=ref_display, stage=stage
    ) as resp:
        if (
            err := status_error(resp.status, ref_display=ref_display, stage=stage)
        ) is not None:
            raise err
        try:
            sha, size = await stream_to_file(
                resp.content.iter_chunked(_DOWNLOAD_CHUNK), dest
            )
        # aiohttp.ClientOSError is also an OSError, so transport errors go first.
        except (TimeoutError, aiohttp.ClientError) as e:
            raise _transport_error(e, ref_display=ref_display, stage=stage) from e
        except OSError as e:
            local = local_fs_error(e, ref_display=ref_display, stage=stage)
            if local is None:
                raise
            raise local from e
        return sha, size, _metadata(resp)
