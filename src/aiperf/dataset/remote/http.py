# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolver for https:// and http:// dataset URLs (always a single file)."""

from __future__ import annotations

import asyncio
import hashlib
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


def _version_of(meta: HttpMetadata) -> str:
    if meta.etag:
        return f"etag:{meta.etag}"
    if meta.last_modified and meta.content_length is not None:
        return f"lm:{meta.last_modified}|{meta.content_length}"
    return UNVERSIONED


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
    return _version_of(meta) == pinned


def _filename_of(url: str) -> str:
    return PurePosixPath(unquote(urlsplit(url).path)).name or "download"


class HTTPSourceResolver:
    """Resolve https:// and http:// dataset URLs."""

    async def probe(self, ref: RemoteRef) -> ProbeResult:
        async with open_session() as session:
            meta = await fetch_metadata(
                session, ref.ref, headers={}, ref_display=ref.display
            )
            manifest, manifest_sha = await self._sidecar(session, ref)
        current = _version_of(meta)
        if (
            ref.resolved_version not in (None, UNVERSIONED)
            and current != ref.resolved_version
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
            try:
                raw = await resp.read()
            except (TimeoutError, aiohttp.ClientError) as e:
                raise RemoteSourceError(
                    ref=ref.display,
                    kind=RemoteErrorKind.TRANSIENT,
                    stage=stage,
                    what=f"reading its {SIDECAR_SUFFIX} sidecar failed ({type(e).__name__}: {e})",
                    hint="Check network access and retry",
                ) from e
        manifest = parse_manifest(raw, ref_display=ref.display, kind="sidecar")
        return manifest, hashlib.sha256(raw).hexdigest()

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
