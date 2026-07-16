# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""I/O helpers backing :mod:`aiperf.operator.routers.results_files`."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiofiles
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from aiperf.kubernetes.results_sidecar import (
    CHECKPOINTS_DIR_NAME,
    READY_MARKER_NAME,
    ready_marker_path,
)
from aiperf.operator.results_layout import resolve_run_dir
from aiperf.operator.routers.results_schemas import FileEntry, JobEntry

CHUNK_SIZE = 64 * 1024
PROFILE_EXPORT_FILENAME = "profile_export_aiperf.json"
_ARTIFACT_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".parquet": "application/vnd.apache.parquet",
    ".png": "image/png",
}
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
}


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
    stat = artifact.path.stat()
    return FileEntry(
        name=_artifact_display_name(artifact.name),
        stored_name=artifact.name,
        size_bytes=stat.st_size,
        compressed=artifact.path.suffix == ".zst",
        mtime_epoch=int(stat.st_mtime),
    )


def _artifact_media_type(display_name: str) -> str:
    """Return the browser-visible media type for a downloadable artifact."""
    return _ARTIFACT_MEDIA_TYPES.get(
        Path(display_name).suffix, "application/octet-stream"
    )


def _download_headers(display_name: str) -> dict[str, str]:
    """Build download headers for an artifact filename."""
    try:
        display_name.encode("ascii")
    except UnicodeEncodeError:
        disposition = f"attachment; filename*=UTF-8''{quote(display_name)}"
    else:
        disposition = f'attachment; filename="{display_name}"'
    return {
        "Content-Disposition": disposition,
        "X-Filename": display_name,
        **_NO_CACHE_HEADERS,
    }


def _list_file_artifacts(
    root: Path,
    relative_dirs: tuple[str, ...] = (),
    *,
    include_root: bool = True,
) -> list[FileArtifact]:
    artifacts: list[FileArtifact] = []
    if include_root:
        artifacts.extend(
            FileArtifact(path=f, name=f.name)
            for f in root.iterdir()
            if (
                not f.is_symlink()
                and f.is_file()
                and f.name not in {READY_MARKER_NAME, CHECKPOINTS_DIR_NAME}
            )
        )
    for rel_dir in relative_dirs:
        child = _safe_resolve(root, rel_dir)
        if child is None or not child.is_dir():
            continue
        artifacts.extend(
            FileArtifact(path=f, name=f.relative_to(root).as_posix())
            for f in child.rglob("*")
            if not f.is_symlink() and f.is_file()
        )
    return sorted(
        artifacts, key=lambda item: (_artifact_display_name(item.name), item.name)
    )


def _list_artifact_files(
    root: Path,
    relative_dirs: tuple[str, ...] = (),
    *,
    include_root: bool = True,
) -> list[FileEntry]:
    return [
        _artifact_entry(artifact)
        for artifact in _list_file_artifacts(
            root, relative_dirs, include_root=include_root
        )
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
    """Build an in-memory zip of a run dir's visible artifacts.

    Mirrors what ``list_job_files_with_readiness`` exposes: the direct files
    plus the ``checkpoints/`` subtree. Without the explicit ``relative_dirs``,
    ``_list_file_artifacts`` skips ``CHECKPOINTS_DIR_NAME`` during the root
    walk, so the bundle would omit checkpoint parquet files that the run-detail
    listing shows as downloadable — a visible-but-unbundlable mismatch.
    """
    return await _build_artifact_bundle(job_dir, (CHECKPOINTS_DIR_NAME,))


async def _stream_artifact_bundle(
    root: Path, relative_dirs: tuple[str, ...] = ()
) -> AsyncIterator[bytes]:
    """Yield a prebuilt artifact bundle in fixed-size chunks."""
    data = await _build_artifact_bundle(root, relative_dirs)
    for i in range(0, len(data), CHUNK_SIZE):
        yield data[i : i + CHUNK_SIZE]


async def _stream_job_bundle(job_dir: Path) -> AsyncIterator[bytes]:
    """Yield a prebuilt job bundle (direct files + ``checkpoints/`` subtree).

    Passes ``CHECKPOINTS_DIR_NAME`` so the streamed ``.zip`` matches
    ``list_job_files_with_readiness``: checkpoint parquet files shown as
    downloadable in the run-detail listing are also present in the archive,
    rather than silently dropped during the root-only walk.
    """
    async for chunk in _stream_artifact_bundle(job_dir, (CHECKPOINTS_DIR_NAME,)):
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

    headers = _download_headers(display_name)

    if "zstd" in accept:
        headers["Content-Encoding"] = "zstd"
        return StreamingResponse(
            _stream_zstd_raw(zst_path),
            media_type=_artifact_media_type(display_name),
            headers=headers,
        )

    if "gzip" in accept:
        headers["Content-Encoding"] = "gzip"
        return StreamingResponse(
            _stream_zstd_to_gzip(zst_path),
            media_type=_artifact_media_type(display_name),
            headers=headers,
        )

    return StreamingResponse(
        _stream_zstd_decompress(zst_path),
        media_type=_artifact_media_type(display_name),
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

    headers = _download_headers(file_path.name)
    if encoding != CompressionEncoding.IDENTITY:
        headers["Content-Encoding"] = encoding

    return StreamingResponse(
        stream_file_compressed(file_path, encoding),
        media_type=_artifact_media_type(file_path.name),
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
    """Walk ``<namespace>/<job_id>/<epoch>/`` under ``base_dir``."""
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
            try:
                files = [
                    f
                    for f in latest_dir.iterdir()
                    if f.is_file() and f.name != READY_MARKER_NAME
                ]
                total_size_bytes = sum(f.stat().st_size for f in files)
            except OSError:
                continue
            if not files:
                continue
            model, endpoint = _extract_model_endpoint(latest_dir)
            found.append(
                JobEntry(
                    namespace=ns_dir.name,
                    job_id=name_dir.name,
                    file_count=len(files),
                    total_size_bytes=total_size_bytes,
                    model=model,
                    endpoint=endpoint,
                )
            )
    return found


def list_job_files_with_readiness(run_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    """List visible run artifacts and whether final files are download-ready."""
    ready = ready_marker_path(run_dir).is_file()
    entries = _list_artifact_files(
        run_dir,
        (CHECKPOINTS_DIR_NAME,),
        include_root=ready,
    )
    return [entry.model_dump() for entry in entries], ready


def _list_job_files(job_dir: Path) -> list[FileEntry]:
    """List visible artifacts in a run directory, sorted by display name."""
    files, _ready = list_job_files_with_readiness(job_dir)
    return [FileEntry.model_validate(file_info) for file_info in files]


def _read_profile_export_bytes(
    job_dir: Path, filename: str = PROFILE_EXPORT_FILENAME
) -> bytes:
    """Return the raw JSON bytes of ``filename`` in ``job_dir``.

    Defaults to the per-run ``profile_export_aiperf.json``; sweep callers pass
    the mirrored aggregate name (``profile_export_aiperf_aggregate.json``).
    Prefers the uncompressed file when present, then falls back to the
    ``.zst`` companion (decompressed in-memory). Raises ``FileNotFoundError``
    if neither exists so the caller can map it to a 404. The whole file is
    read into memory rather than streamed because typical profile exports
    are small (sub-MB) and callers (the dashboard quick-export button)
    expect a single ``application/json`` payload, not a streaming download.
    """
    raw_path = _safe_resolve(job_dir, filename)
    if raw_path is not None and raw_path.is_file():
        return raw_path.read_bytes()
    zst_path = _safe_resolve(job_dir, filename + ".zst")
    if zst_path is not None and zst_path.is_file():
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with zst_path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            return reader.read()
    raise FileNotFoundError(filename)


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


def _is_checkpoint_artifact(root: Path, path: Path) -> bool:
    """Return whether ``path`` resolves under the checkpoint subtree."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == CHECKPOINTS_DIR_NAME


def _not_ready_error(job_dir: Path) -> HTTPException:
    message = (
        f"Results not ready for {job_dir.name}; marker file "
        f"{READY_MARKER_NAME} not present — retry after completion"
    )
    return HTTPException(404, message)


def _serve_job_file(
    request: Request, job_dir: Path, filename: str
) -> StreamingResponse:
    """Serve ``filename`` from ``job_dir``, preferring .zst + content negotiation."""
    if Path(filename).name == READY_MARKER_NAME:
        raise HTTPException(404, f"File not found: {filename}")
    raw_path = _safe_resolve(job_dir, filename)
    zst_path = _safe_resolve(job_dir, filename + ".zst")
    candidate = raw_path or zst_path
    if (
        candidate is not None
        and not ready_marker_path(job_dir).is_file()
        and not _is_checkpoint_artifact(job_dir, candidate)
    ):
        raise _not_ready_error(job_dir)
    return _serve_artifact_file(request, job_dir, filename)
