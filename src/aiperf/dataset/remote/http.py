# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolver for https:// and http:// dataset URLs (always a single file)."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit

import aiohttp

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.cache import entry_dir
from aiperf.dataset.remote.errors import (
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
    changed_after_pin,
)
from aiperf.dataset.remote.http_fetch import (
    HttpMetadata,
    download_to,
    fetch_metadata,
    open_session,
    read_prefix,
    safe_request,
    status_error,
)
from aiperf.dataset.remote.manifest import (
    SIDECAR_SUFFIX,
    AIPerfManifest,
    parse_manifest,
)
from aiperf.dataset.remote.models import (
    UNVERSIONED,
    ProbeResult,
    RemoteFile,
    ResolvedSource,
)

_logger = AIPerfLogger(__name__)

# Object stores answer 403 for a missing key in a public bucket.
_NO_SIDECAR_STATUSES = frozenset({401, 403, 404, 410})
_SIDECAR_MAX_BYTES = 1 << 20
_FALLBACK_FILENAME = "download"


def _version_of(meta: HttpMetadata) -> str:
    if meta.etag:
        return f"etag:{meta.etag}"
    if meta.last_modified and meta.content_length is not None:
        return f"lm:{meta.last_modified}|{meta.content_length}"
    return UNVERSIONED


def _weak(version: str) -> str:
    """An ``etag:`` version with any weak-validator prefix dropped (RFC 9110 weak comparison)."""
    if version.startswith("etag:W/"):
        return "etag:" + version[len("etag:W/") :]
    return version


def _same_version(current: str, pinned: str) -> bool:
    # nginx weakens a strong ETag when it gzips the body, so HEAD and a
    # compressed GET of the same bytes report "x" and W/"x".
    return _weak(current) == _weak(pinned)


def _still_pinned(meta: HttpMetadata, pinned: str) -> bool:
    """Whether the headers of a full GET still match the version the probe pinned.

    A ``lm:`` pin also records the probe's Content-Length, which a chunked or
    compressed GET may omit; only a Content-Length that is present and different
    counts as a change.
    """
    if pinned.startswith("lm:") and not meta.etag and meta.last_modified:
        last_modified, _, length = pinned[len("lm:") :].rpartition("|")
        return meta.last_modified == last_modified and (
            meta.content_length is None or str(meta.content_length) == length
        )
    return _same_version(_version_of(meta), pinned)


def _filename_of(url: str) -> str:
    """Cache file name for a URL; never a name that could leave the cache entry."""
    name = PurePosixPath(unquote(urlsplit(url).path)).name
    if name in ("", ".", "..") or "\\" in name or os.sep in name:
        return _FALLBACK_FILENAME
    return name


class HTTPSourceResolver:
    """Resolve https:// and http:// dataset URLs."""

    async def probe(self, ref: RemoteRef) -> ProbeResult:
        async with open_session() as session:
            meta = await fetch_metadata(
                session, ref.ref, headers={}, ref_display=ref.display
            )
            manifest, manifest_sha = await self._sidecar(session, ref)
        current = _version_of(meta)
        if ref.resolved_version not in (None, UNVERSIONED) and not _same_version(
            current, ref.resolved_version
        ):
            raise changed_after_pin(ref.display, RemoteStage.ACCESS, "the file")
        if current == UNVERSIONED:
            _logger.warning(
                f"{ref.display} sends no ETag or Last-Modified header; it will be "
                "downloaded again on every run"
            )
        pinned = ref.with_version(current)
        return ProbeResult(
            ref=pinned,
            pinned_ref=pinned.display,
            data_relpath=None,
            data_is_dir=False,
            filename=_filename_of(ref.ref),
            manifest=manifest,
            manifest_sha256=manifest_sha,
            needs_auth=False,
        )

    async def _sidecar(
        self, session: aiohttp.ClientSession, ref: RemoteRef
    ) -> tuple[AIPerfManifest | None, str | None]:
        parts = urlsplit(ref.ref)
        if parts.query:
            # A presigned signature covers only the named object; a sidecar is unreachable.
            return None, None
        sidecar = urlunsplit(parts._replace(path=parts.path + SIDECAR_SUFFIX))
        stage = RemoteStage.ACCESS
        async with safe_request(
            session, "GET", sidecar, headers={}, ref_display=ref.display, stage=stage
        ) as resp:
            if resp.status in _NO_SIDECAR_STATUSES:
                return None, None
            if (
                err := status_error(resp.status, ref_display=ref.display, stage=stage)
            ) is not None:
                raise err
            # SPA catch-alls and SSO landing pages answer any path with 200 HTML.
            if "html" in resp.headers.get("Content-Type", "").lower():
                return None, None
            try:
                raw = await self._read_capped(resp)
            except (TimeoutError, aiohttp.ClientError) as e:
                raise RemoteSourceError(
                    ref=ref.display,
                    kind=RemoteErrorKind.TRANSIENT,
                    stage=stage,
                    what=f"reading its {SIDECAR_SUFFIX} sidecar failed ({type(e).__name__}: {e})",
                    hint="Check network access and retry",
                ) from e
        if len(raw) > _SIDECAR_MAX_BYTES:
            raise RemoteSourceError(
                ref=ref.display,
                kind=RemoteErrorKind.MANIFEST_INVALID,
                stage=stage,
                what=f"its {SIDECAR_SUFFIX} sidecar is larger than 1 MiB",
                hint="A manifest is a small JSON document; check what the server returns at the sidecar URL",
            )
        manifest = parse_manifest(raw, ref_display=ref.display, kind="sidecar")
        return manifest, hashlib.sha256(raw).hexdigest()

    @staticmethod
    async def _read_capped(resp: aiohttp.ClientResponse) -> bytes:
        """Up to ``_SIDECAR_MAX_BYTES + 1`` bytes of the body; more than the cap means oversize."""
        buf = bytearray()
        # StreamReader.read(n) returns whatever is buffered, often a single chunk.
        while len(buf) <= _SIDECAR_MAX_BYTES:
            chunk = await resp.content.read(_SIDECAR_MAX_BYTES + 1 - len(buf))
            if not chunk:
                break
            buf += chunk
        return bytes(buf)

    async def read_prefix(self, probe: ProbeResult, nbytes: int) -> tuple[bytes, bool]:
        async with open_session() as session:
            return await read_prefix(
                session,
                probe.ref.ref,
                nbytes,
                headers={},
                ref_display=probe.ref.display,
            )

    async def materialize(self, probe: ProbeResult) -> ResolvedSource:
        dest = entry_dir(probe.ref) / probe.filename
        async with open_session() as session:
            sha, size, meta = await download_to(
                session, probe.ref.ref, dest, headers={}, ref_display=probe.ref.display
            )
        pinned = probe.ref.resolved_version
        if pinned not in (None, UNVERSIONED) and not _still_pinned(meta, pinned):
            await asyncio.to_thread(dest.unlink, missing_ok=True)
            raise changed_after_pin(
                probe.ref.display, RemoteStage.MATERIALIZE, "the file"
            )
        return ResolvedSource(
            probe=probe,
            local_path=dest,
            files=[RemoteFile(path=probe.filename, sha256=sha, size_bytes=size)],
        )
