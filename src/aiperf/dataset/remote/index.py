# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Index of materialized remote sources, so later stages resolve paths offline."""

from __future__ import annotations

import os
import uuid
from pathlib import Path, PurePosixPath

import orjson

from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.mmap_cache import hash_file_bytes
from aiperf.dataset.remote.cache import digest, remote_cache_dir
from aiperf.dataset.remote.models import RemoteFile, ResolvedSource


def _ref_dir(ref: RemoteRef) -> Path:
    return remote_cache_dir() / "index" / digest(ref.canonical, 32)


def index_path(ref: RemoteRef) -> Path:
    """Index file for one exact version of ``ref``; the ref must be pinned."""
    if ref.resolved_version is None:
        raise ValueError(f"{ref.display} has no resolved_version")
    return _ref_dir(ref) / f"{digest(ref.resolved_version, 16)}.json"


def write_index(source: ResolvedSource) -> None:
    """Atomically record ``source`` (redacted on serialization) in the index."""
    path = index_path(source.probe.ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(orjson.dumps(source.model_dump(mode="json")))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_index(ref: RemoteRef) -> ResolvedSource | None:
    """Return the indexed source for a pinned ``ref``, or None if absent or unpinned."""
    if ref.resolved_version is None:
        return None
    path = index_path(ref)
    if not path.is_file():
        return None
    return ResolvedSource.model_validate(orjson.loads(path.read_bytes()))


def latest_indexed_version(ref: RemoteRef) -> str | None:
    """Version of the most recently written index entry for ``ref``, if any."""
    directory = _ref_dir(ref)
    if not directory.is_dir():
        return None
    entries = sorted(
        directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not entries:
        return None
    source = ResolvedSource.model_validate(orjson.loads(entries[0].read_bytes()))
    return source.probe.ref.resolved_version


def describe_files(local: Path, relpath: str) -> list[RemoteFile]:
    """Digest a downloaded file, or every file under a downloaded directory."""
    if local.is_file():
        return [
            RemoteFile(
                path=relpath,
                sha256=hash_file_bytes(local),
                size_bytes=local.stat().st_size,
            )
        ]
    return [
        RemoteFile(
            path=str(PurePosixPath(relpath) / f.relative_to(local).as_posix()),
            sha256=hash_file_bytes(f),
            size_bytes=f.stat().st_size,
        )
        for f in sorted(local.rglob("*"))
        if f.is_file()
    ]
