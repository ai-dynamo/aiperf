# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheme-agnostic resolution: retries, offline mode, caching, and local-path lookup."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import TypeVar

from filelock import Timeout

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.cache import (
    entry_dir,
    entry_lock,
    file_sha256,
    local_fs_error,
    lock_timeout_error,
)
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError, RemoteStage
from aiperf.dataset.remote.index import latest_indexed_version, read_index, write_index
from aiperf.dataset.remote.models import (
    UNVERSIONED,
    ProbeResult,
    RemoteFile,
    ResolvedSource,
)
from aiperf.dataset.remote.protocols import SourceResolverProtocol

_logger = AIPerfLogger(__name__)
T = TypeVar("T")

RETRY_ATTEMPTS = 3
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def get_resolver(scheme: str) -> SourceResolverProtocol:
    """Instantiate the ``dataset_source_resolver`` plugin registered for ``scheme``."""
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    return plugins.get_class(PluginType.DATASET_SOURCE_RESOLVER, scheme)()


def is_remote_offline() -> bool:
    """True when remote sources must be served from the cache with no network calls."""
    return bool(Environment.DATASET.REMOTE_OFFLINE) or (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUTHY
    )


async def with_transient_retry(operation: Callable[[], Awaitable[T]]) -> T:
    """Run ``operation``, retrying TRANSIENT failures with exponential backoff."""
    delay = 1.0
    for attempt in range(1, RETRY_ATTEMPTS):
        try:
            return await operation()
        except RemoteSourceError as e:
            if e.kind != RemoteErrorKind.TRANSIENT:
                raise
            _logger.warning(
                f"{e} (attempt {attempt}/{RETRY_ATTEMPTS}; retrying in {delay:.0f}s)"
            )
            await asyncio.sleep(delay)
            delay *= 2
    return await operation()


def _offline_error(ref: RemoteRef, stage: RemoteStage) -> RemoteSourceError:
    return RemoteSourceError(
        ref=ref.display, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
        what="offline mode is on and it is not in the local cache",
        hint="Run once with network access, or unset AIPERF_DATASET_REMOTE_OFFLINE / HF_HUB_OFFLINE",
    )  # fmt: skip


def _disk_path(source: ResolvedSource, file: RemoteFile) -> Path:
    """Where a recorded file lives on disk; the inverse of ``index.describe_files``.

    Raises ``ValueError`` when the recorded path is not under ``data_relpath``.
    """
    if not source.probe.data_is_dir:
        return source.local_path
    rel = PurePosixPath(file.path)
    if source.probe.data_relpath:
        rel = rel.relative_to(source.probe.data_relpath)
    return source.local_path.joinpath(*rel.parts)


def _is_intact(source: ResolvedSource) -> bool:
    """Every recorded file is on disk with its recorded size.

    Cache writers do not fsync, so after an OS crash a file can survive truncated
    while its index entry still claims it; a size mismatch is treated as a miss.
    """
    if not source.local_path.exists():
        return False
    try:
        return all(
            _disk_path(source, f).stat().st_size == f.size_bytes for f in source.files
        )
    except (OSError, ValueError):
        return False


def _usable_entry(ref: RemoteRef) -> ResolvedSource | None:
    source = read_index(ref)
    if source is None:
        return None
    if not _is_intact(source):
        _logger.warning(
            f"Cached copy of {ref.display} is missing or incomplete; ignoring it"
        )
        return None
    return source


def indexed_file(ref: RemoteRef, relpath: str) -> Path | None:
    """Local path of ``relpath`` in the indexed, intact cache entry of ``ref``; None otherwise.

    Reads only the local cache, and logs nothing on a miss: the caller fetches instead.
    """
    source = read_index(ref)
    if source is None or not _intact(source):
        return None
    file = next((f for f in source.files if f.path == relpath), None)
    return None if file is None else _disk_path(source, file)


async def probe_ref(ref: RemoteRef) -> ProbeResult:
    """Check access and pin the version of ``ref`` (from the cache when offline)."""
    if is_remote_offline():
        probe = await asyncio.to_thread(_offline_probe, ref)
    else:
        resolver = get_resolver(ref.scheme)
        probe = await with_transient_retry(lambda: resolver.probe(ref))
    if probe.data_is_dir and probe.manifest is not None and probe.manifest.sha256:
        _logger.warning(
            f"The manifest of {ref.display} declares a sha256, but its data is a folder; "
            "sha256 is checked for single files only, so these files are not verified"
        )
    return probe


def _offline_probe(ref: RemoteRef) -> ProbeResult:
    version = ref.resolved_version or latest_indexed_version(ref)
    source = _usable_entry(ref.with_version(version)) if version else None
    if source is None:
        raise _offline_error(ref, RemoteStage.ACCESS)
    if ref.resolved_version is None and ref.parsed.revision is None:
        _logger.warning(
            f"Offline mode: using cached {source.probe.pinned_ref} for {ref.display} "
            "without checking for a newer version"
        )
    # Keep the caller's ref: the indexed one was serialized redacted.
    return source.probe.model_copy(update={"ref": ref.with_version(version)})


def _declared_sha256(probe: ProbeResult) -> str | None:
    if probe.manifest is None or probe.data_is_dir:
        return None
    return probe.manifest.sha256


def _verify_checksum(source: ResolvedSource) -> None:
    declared = _declared_sha256(source.probe)
    actual = source.files[0].sha256
    if declared is not None and actual != declared:
        raise RemoteSourceError(
            ref=source.probe.ref.display, kind=RemoteErrorKind.CHECKSUM_MISMATCH,
            stage=RemoteStage.MATERIALIZE,
            what=f"the downloaded file's sha256 is {actual}, but its manifest declares {declared}",
            hint="Retry; if it persists, the publisher's file or manifest is wrong",
        )  # fmt: skip


def _cached_for(probe: ProbeResult) -> ResolvedSource | None:
    """The intact cached copy of ``probe``, recorded with the manifest accepted now.

    A version pins only the data file for HTTP and single S3 objects, so a
    sidecar can change (its sha256 included) while the cache key stays the same.
    A declared sha256 is checked against the file's bytes, not only the digest
    the index recorded at download time.
    """
    source = _usable_entry(probe.ref)
    if source is None:
        return None
    declared = _declared_sha256(probe)
    if declared is not None and (
        source.files[0].sha256 != declared or file_sha256(source.local_path) != declared
    ):
        _logger.warning(
            f"Cached copy of {probe.ref.display} does not match the sha256 its "
            "manifest declares; downloading it again"
        )
        return None
    if source.probe.manifest_sha256 != probe.manifest_sha256:
        # Provenance reads the index, so record the sidecar this run accepted.
        source = source.model_copy(update={"probe": probe})
        write_index(source)
    return source


async def materialize_probe(probe: ProbeResult) -> ResolvedSource:
    """Return the cached download for ``probe``, downloading it first if needed.

    The index entry is written only after a successful, verified download, so a
    failure leaves no entry that later stages could mistake for a usable cache.
    """
    ref = probe.ref
    offline = is_remote_offline()
    # An unversioned URL can change without notice, so online it is always
    # re-downloaded; offline, the last copy is the best there is.
    reusable = offline or ref.resolved_version != UNVERSIONED
    if reusable and (cached := await asyncio.to_thread(_cached_for, probe)):
        return cached
    if offline:
        raise _offline_error(ref, RemoteStage.MATERIALIZE)
    try:
        return await _download(probe, reusable=reusable)
    except OSError as e:
        local = local_fs_error(
            e, ref_display=ref.display, stage=RemoteStage.MATERIALIZE
        )
        if local is None:
            raise
        raise local from e


async def _download(probe: ProbeResult, *, reusable: bool) -> ResolvedSource:
    """Download ``probe`` under the entry lock and record it in the index."""
    ref = probe.ref
    resolver = get_resolver(ref.scheme)
    entry = entry_dir(ref)
    async with contextlib.AsyncExitStack() as stack:
        try:
            await stack.enter_async_context(entry_lock(entry))
        except Timeout as e:
            raise lock_timeout_error(
                entry, ref_display=ref.display, stage=RemoteStage.MATERIALIZE
            ) from e
        # Another process may have finished this download while we waited.
        if reusable and (cached := await asyncio.to_thread(_cached_for, probe)):
            return cached
        source = await with_transient_retry(lambda: resolver.materialize(probe))
        _verify_checksum(source)
        await asyncio.to_thread(write_index, source)
    return source


def cached_source(ref: RemoteRef) -> ResolvedSource:
    """Return the materialized source for a pinned ref without any network access."""
    if ref.resolved_version is None:
        raise RuntimeError(
            f"{ref.display} was never resolved; the dataset access preflight must run first"
        )
    source = _usable_entry(ref)
    if source is None:
        raise RuntimeError(
            f"{ref.display}@{ref.resolved_version} is not in the remote dataset cache; "
            "the dataset materialize preflight must run first"
        )
    return source


def local_path_for(path: Path | RemoteRef) -> Path:
    """Local filesystem path for a dataset path, materialized remote refs included."""
    if isinstance(path, RemoteRef):
        return cached_source(path).local_path
    return Path(path)
