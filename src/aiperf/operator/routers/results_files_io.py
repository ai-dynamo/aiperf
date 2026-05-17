# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""I/O helpers backing :mod:`aiperf.operator.routers.results_files`.

Pure file-system + streaming utilities — path traversal guards, zstd/gzip
content-negotiation streamers, on-disk job-dir scanning, and zip-bundle
construction. Routing concerns (route registration, request parsing) live
in the sibling ``results_files`` module so each file stays single-purpose.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from aiperf.operator.results_layout import resolve_run_dir
from aiperf.operator.routers.results_schemas import FileEntry, JobEntry

CHUNK_SIZE = 64 * 1024
PROFILE_EXPORT_FILENAME = "profile_export_aiperf.json"


@dataclass(frozen=True, slots=True)
class FileArtifact:
    """A file under an artifact root plus its API-visible relative name."""

    path: Path
    name: str


def _safe_resolve(base: Path, *parts: str) -> Path | None:
    """Resolve path parts under base, returning None on traversal attempts."""
    try:
        resolved = (base / Path(*parts)).resolve()
        resolved.relative_to(base.resolve())
        return resolved
    except (ValueError, OSError):
        return None


def _display_name(path: Path) -> str:
    """Strip .zst suffix for display."""
    if path.suffix == ".zst":
        return path.stem
    return path.name


def _artifact_display_name(name: str) -> str:
    """Strip .zst from an API-visible relative artifact name."""
    path = Path(name)
    display_leaf = _display_name(path)
    return str(path.with_name(display_leaf))


def _artifact_entry(artifact: FileArtifact) -> FileEntry:
    return FileEntry(
        name=_artifact_display_name(artifact.name),
        stored_name=artifact.name,
        size_bytes=artifact.path.stat().st_size,
        compressed=artifact.path.suffix == ".zst",
    )


def _list_file_artifacts(
    root: Path, relative_dirs: tuple[str, ...] = ()
) -> list[FileArtifact]:
    artifacts = [
        FileArtifact(path=f, name=f.name)
        for f in root.iterdir()
        if not f.is_symlink() and f.is_file()
    ]
    for rel_dir in relative_dirs:
        child = _safe_resolve(root, rel_dir)
        if child is None or not child.is_dir():
            continue
        artifacts.extend(
            FileArtifact(path=f, name=f"{rel_dir}/{f.name}")
            for f in child.iterdir()
            if not f.is_symlink() and f.is_file()
        )
    return sorted(artifacts, key=lambda item: _artifact_display_name(item.name))


def _list_artifact_files(
    root: Path, relative_dirs: tuple[str, ...] = ()
) -> list[FileEntry]:
    return [
        _artifact_entry(artifact)
        for artifact in _list_file_artifacts(root, relative_dirs)
    ]


async def _build_artifact_bundle(
    root: Path, relative_dirs: tuple[str, ...] = ()
) -> bytes:
    """Build an in-memory zip of scoped artifact files."""
    import io
    import zipfile

    import zstandard

    dctx = zstandard.ZstdDecompressor()
    artifacts = _list_file_artifacts(root, relative_dirs)

    def _build() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(
            buf, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zf:
            for artifact in artifacts:
                arcname = _artifact_display_name(artifact.name)
                if artifact.path.suffix == ".zst":
                    with (
                        artifact.path.open("rb") as fh,
                        dctx.stream_reader(fh) as reader,
                    ):
                        payload = reader.read()
                else:
                    payload = artifact.path.read_bytes()
                zf.writestr(arcname, payload)
        return buf.getvalue()

    return await asyncio.to_thread(_build)


async def _build_job_bundle(job_dir: Path) -> bytes:
    """Build an in-memory zip of every direct file in ``job_dir``."""
    return await _build_artifact_bundle(job_dir)


async def _stream_artifact_bundle(
    root: Path, relative_dirs: tuple[str, ...] = ()
) -> AsyncIterator[bytes]:
    """Yield a prebuilt artifact bundle in fixed-size chunks."""
    data = await _build_artifact_bundle(root, relative_dirs)
    for i in range(0, len(data), CHUNK_SIZE):
        yield data[i : i + CHUNK_SIZE]


async def _stream_job_bundle(job_dir: Path) -> AsyncIterator[bytes]:
    """Yield a prebuilt job bundle in fixed-size chunks."""
    async for chunk in _stream_artifact_bundle(job_dir):
        yield chunk


async def _stream_zstd_raw(file_path: Path) -> AsyncIterator[bytes]:
    """Stream a .zst file directly as raw bytes."""
    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(CHUNK_SIZE):
            yield chunk


async def _stream_zstd_to_gzip(file_path: Path) -> AsyncIterator[bytes]:
    """Decompress zstd, recompress as gzip (streaming)."""
    import zlib

    import zstandard

    gzip_obj = zlib.compressobj(level=6, wbits=31)
    dctx = zstandard.ZstdDecompressor()

    with open(file_path, "rb") as f, dctx.stream_reader(f) as reader:
        while chunk := await asyncio.to_thread(reader.read, CHUNK_SIZE):
            gzip_chunk = gzip_obj.compress(chunk)
            if gzip_chunk:
                yield gzip_chunk

    final = gzip_obj.flush()
    if final:
        yield final


async def _stream_zstd_decompress(file_path: Path) -> AsyncIterator[bytes]:
    """Decompress zstd on the fly."""
    import zstandard

    dctx = zstandard.ZstdDecompressor()

    with open(file_path, "rb") as f, dctx.stream_reader(f) as reader:
        while chunk := await asyncio.to_thread(reader.read, CHUNK_SIZE):
            yield chunk


def _serve_zst_file(
    request: Request, zst_path: Path, display_name: str
) -> StreamingResponse:
    """Serve a .zst file with content negotiation."""
    accept = (request.headers.get("accept-encoding") or "").lower()

    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="{display_name}"',
        "X-Filename": display_name,
    }

    if "zstd" in accept:
        headers["Content-Encoding"] = "zstd"
        return StreamingResponse(
            _stream_zstd_raw(zst_path),
            media_type="application/octet-stream",
            headers=headers,
        )

    if "gzip" in accept:
        headers["Content-Encoding"] = "gzip"
        return StreamingResponse(
            _stream_zstd_to_gzip(zst_path),
            media_type="application/octet-stream",
            headers=headers,
        )

    return StreamingResponse(
        _stream_zstd_decompress(zst_path),
        media_type="application/octet-stream",
        headers=headers,
    )


def _serve_raw_file(request: Request, file_path: Path) -> StreamingResponse:
    """Serve an uncompressed file, optionally compressing on the fly."""
    from aiperf.common.compression import (
        CompressionEncoding,
        select_encoding,
        stream_file_compressed,
    )

    accept = request.headers.get("accept-encoding")
    encoding = select_encoding(accept, default=CompressionEncoding.IDENTITY)

    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="{file_path.name}"',
        "X-Filename": file_path.name,
    }
    if encoding != CompressionEncoding.IDENTITY:
        headers["Content-Encoding"] = encoding

    return StreamingResponse(
        stream_file_compressed(file_path, encoding),
        media_type="application/octet-stream",
        headers=headers,
    )


def _model_name_from_benchmark(benchmark: dict[str, Any]) -> str | None:
    """Extract the first model name from a benchmark dict, tolerating shapes.

    ``models`` can be ``{"items": [{"name": "x"}]}``, ``{"modelNames": ["x"]}``,
    or just ``["x"]`` — match the shape-tolerance in ``operator/runs_index.py``.
    """
    models_cfg = benchmark.get("models", {})
    if isinstance(models_cfg, list):
        items = models_cfg
    elif isinstance(models_cfg, dict):
        items = models_cfg.get("items", models_cfg.get("modelNames", []))
    else:
        return None
    if not isinstance(items, list) or not items:
        return None
    first = items[0]
    if isinstance(first, dict):
        return first.get("name")
    return str(first) if first is not None else None


def _endpoint_url_from_benchmark(benchmark: dict[str, Any]) -> str | None:
    """Extract the first endpoint URL from a benchmark dict, tolerating shapes."""
    endpoint_cfg = benchmark.get("endpoint", {})
    if not isinstance(endpoint_cfg, dict):
        return None
    urls = endpoint_cfg.get("urls", endpoint_cfg.get("url", []))
    if isinstance(urls, str):
        return urls
    if isinstance(urls, list) and urls and urls[0] is not None:
        return str(urls[0])
    return None


def _extract_model_endpoint(latest_dir: Path) -> tuple[str | None, str | None]:
    """Read ``job_spec.json`` from the run dir and extract (model, endpoint).

    Mirrors the extraction in ``operator/runs_index.py`` so the SQLite
    index and the on-disk fallback agree on what a job's "model" is.
    Returns ``(None, None)`` for any failure — older jobs predate
    ``job_spec.json`` and we don't want to fail the entire ``/results``
    listing if one of them is unparsable.
    """
    spec_path = latest_dir / "job_spec.json"
    if not spec_path.exists():
        return None, None
    try:
        import orjson

        spec = orjson.loads(spec_path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None, None
    if not isinstance(spec, dict):
        return None, None
    benchmark = spec.get("benchmark", spec)
    if not isinstance(benchmark, dict):
        return None, None
    return _model_name_from_benchmark(benchmark), _endpoint_url_from_benchmark(
        benchmark
    )


def _scan_job_dirs(base_dir: Path) -> list[JobEntry]:
    """Walk ``<namespace>/<job_id>/<epoch>/`` under ``base_dir``.

    Yields one :class:`JobEntry` per ``<ns>/<name>`` using the run pointed
    to by latest.txt. Jobs whose pointer is missing or targets a vanished
    epoch are skipped silently. ``model`` and ``endpoint`` are populated
    from the run dir's ``job_spec.json`` so the UI can filter clusters of
    "similar runs" without round-tripping to the live AIPerfJob CR list.
    """
    found: list[JobEntry] = []
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        for name_dir in sorted(ns_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            latest_dir = resolve_run_dir(base_dir, ns_dir.name, name_dir.name)
            if latest_dir is None:
                continue
            files = [f for f in latest_dir.iterdir() if f.is_file()]
            if not files:
                continue
            model, endpoint = _extract_model_endpoint(latest_dir)
            found.append(
                JobEntry(
                    namespace=ns_dir.name,
                    job_id=name_dir.name,
                    file_count=len(files),
                    total_size_bytes=sum(f.stat().st_size for f in files),
                    model=model,
                    endpoint=endpoint,
                )
            )
    return found


def _list_job_files(job_dir: Path) -> list[FileEntry]:
    """List all regular files in a single job directory, sorted by display name."""
    return _list_artifact_files(job_dir)


def _read_profile_export_bytes(job_dir: Path) -> bytes:
    """Return the raw JSON bytes of ``profile_export_aiperf.json`` in ``job_dir``.

    Prefers the uncompressed file when present, then falls back to the
    ``.zst`` companion (decompressed in-memory). Raises ``FileNotFoundError``
    if neither exists so the caller can map it to a 404. The whole file is
    read into memory rather than streamed because typical profile exports
    are small (sub-MB) and callers (the dashboard quick-export button)
    expect a single ``application/json`` payload, not a streaming download.
    """
    raw_path = _safe_resolve(job_dir, PROFILE_EXPORT_FILENAME)
    if raw_path is not None and raw_path.is_file():
        return raw_path.read_bytes()
    zst_path = _safe_resolve(job_dir, PROFILE_EXPORT_FILENAME + ".zst")
    if zst_path is not None and zst_path.is_file():
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with zst_path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            return reader.read()
    raise FileNotFoundError(PROFILE_EXPORT_FILENAME)


def _serve_artifact_file(
    request: Request,
    root: Path,
    filename: str,
    *,
    allowed_relative_dirs: tuple[str, ...] = (),
) -> StreamingResponse:
    """Serve a scoped artifact file, preferring .zst + content negotiation."""
    zst_path = _safe_resolve(root, filename + ".zst")
    raw_path = _safe_resolve(root, filename)
    allowed_roots = [root.resolve()]
    for rel_dir in allowed_relative_dirs:
        child = _safe_resolve(root, rel_dir)
        if child is not None:
            allowed_roots.append(child.resolve())

    def _is_allowed(path: Path | None) -> bool:
        if path is None:
            return False
        resolved = path.resolve()
        if not allowed_relative_dirs:
            return resolved.is_relative_to(root.resolve())
        return any(resolved.parent == allowed for allowed in allowed_roots)

    display_name = Path(filename).name
    if _is_allowed(zst_path) and zst_path is not None and zst_path.is_file():
        return _serve_zst_file(request, zst_path, display_name)
    if _is_allowed(raw_path) and raw_path is not None and raw_path.is_file():
        return _serve_raw_file(request, raw_path)
    raise HTTPException(404, f"File not found: {filename}")


def _serve_job_file(
    request: Request, job_dir: Path, filename: str
) -> StreamingResponse:
    """Serve ``filename`` from ``job_dir``, preferring .zst + content negotiation."""
    return _serve_artifact_file(request, job_dir, filename)
