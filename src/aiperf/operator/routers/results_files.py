# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File-serving routes for the operator results API.

Lists and downloads raw benchmark result files from the operator PVC,
with zstd/gzip/identity content negotiation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from aiperf.operator.routers.results_schemas import (
    FileEntry,
    FileListResponse,
    JobEntry,
    ResultsHistoryListResponse,
)

CHUNK_SIZE = 64 * 1024


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


async def _build_job_bundle(job_dir: Path) -> bytes:
    """Build an in-memory zip of every file in ``job_dir``, transparently
    decompressing ``.zst`` entries back to their original names.

    One-shot construction (return the full bytes) rather than streaming,
    because ``zipfile.ZipFile`` tracks its own byte offsets and doesn't
    compose with draining the underlying ``BytesIO`` between writes — the
    central-directory offsets get confused and readers see stray bytes.
    For the typical benchmark bundle (a few MiB, single-digit files) the
    whole thing fits in memory easily; streaming buys nothing.
    """
    import io
    import zipfile
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    files = sorted([f for f in job_dir.iterdir() if f.is_file()], key=lambda p: p.name)

    def _build() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for f in files:
                arcname = _display_name(f)
                if f.suffix == ".zst":
                    payload = dctx.decompress(f.read_bytes())
                else:
                    payload = f.read_bytes()
                zf.writestr(arcname, payload)
        return buf.getvalue()

    return await asyncio.to_thread(_build)


async def _stream_job_bundle(job_dir: Path) -> AsyncIterator[bytes]:
    """Yield a prebuilt job bundle in fixed-size chunks so the FastAPI
    ``StreamingResponse`` can flush progressively to the client even though
    the zip itself is constructed in one shot by :func:`_build_job_bundle`."""
    data = await _build_job_bundle(job_dir)
    for i in range(0, len(data), CHUNK_SIZE):
        yield data[i:i + CHUNK_SIZE]


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


def _scan_job_dirs(base_dir: Path) -> list[JobEntry]:
    """Walk ``<namespace>/<job_id>/`` under ``base_dir`` and summarize each job."""
    found: list[JobEntry] = []
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        for job_dir in sorted(ns_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            files = [f for f in job_dir.iterdir() if f.is_file()]
            if files:
                found.append(
                    JobEntry(
                        namespace=ns_dir.name,
                        job_id=job_dir.name,
                        file_count=len(files),
                        total_size_bytes=sum(f.stat().st_size for f in files),
                    )
                )
    return found


def _list_job_files(job_dir: Path) -> list[FileEntry]:
    """List all regular files in a single job directory, sorted by display name."""
    return sorted(
        [
            FileEntry(
                name=_display_name(f),
                stored_name=f.name,
                size_bytes=f.stat().st_size,
                compressed=f.suffix == ".zst",
            )
            for f in job_dir.iterdir()
            if f.is_file()
        ],
        key=lambda x: x.name,
    )


def _resolve_job_dir(base_dir: Path, namespace: str, job_id: str) -> Path:
    """Resolve ``base_dir/namespace/job_id`` or raise 404 if missing/unsafe."""
    job_dir = _safe_resolve(base_dir, namespace, job_id)
    if job_dir is None or not job_dir.is_dir():
        raise HTTPException(404, f"No results for {namespace}/{job_id}")
    return job_dir


def create_results_files_router(base_dir: Path) -> APIRouter:
    """Create the router for file listing/download endpoints.

    Args:
        base_dir: Base directory containing ``<namespace>/<job_id>/`` result files.
    """
    router = APIRouter(prefix="/api/v1", tags=["results-files"])

    @router.get("/results", response_model=ResultsHistoryListResponse)
    async def list_jobs() -> ResultsHistoryListResponse:
        """List all namespaces and jobs with stored results."""
        if not base_dir.exists():
            return ResultsHistoryListResponse()
        jobs = await asyncio.to_thread(_scan_job_dirs, base_dir)
        return ResultsHistoryListResponse(jobs=jobs)

    @router.get("/results/{namespace}/{job_id}.zip")
    async def download_bundle(namespace: str, job_id: str) -> StreamingResponse:
        """Download every result file for a job as one uncompressed zip.

        Entries stored in zstd-compressed form on the PVC are decompressed back
        to their original names inside the archive, so the download is usable
        without extra tooling.

        This route is registered before ``list_job_files`` so the FastAPI
        matcher sees ``{job_id}.zip`` first — otherwise the bare
        ``/results/{ns}/{job_id}`` pattern would capture the ``.zip`` as part
        of the job_id and fall through to a 404.
        """
        job_dir = _resolve_job_dir(base_dir, namespace, job_id)
        bundle_name = f"{namespace}__{job_id}.zip"
        return StreamingResponse(
            _stream_job_bundle(job_dir),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{bundle_name}"',
                "X-Filename": bundle_name,
            },
        )

    @router.get("/results/{namespace}/{job_id}", response_model=FileListResponse)
    async def list_job_files(namespace: str, job_id: str) -> FileListResponse:
        """List files for a specific job."""
        job_dir = _resolve_job_dir(base_dir, namespace, job_id)
        files = await asyncio.to_thread(_list_job_files, job_dir)
        return FileListResponse(namespace=namespace, job_id=job_id, files=files)

    @router.get("/results/{namespace}/{job_id}/{filename:path}")
    async def download_file(
        namespace: str, job_id: str, filename: str, request: Request
    ) -> StreamingResponse:
        """Download a result file with content negotiation."""
        job_dir = _resolve_job_dir(base_dir, namespace, job_id)
        zst_path = _safe_resolve(job_dir, filename + ".zst")
        raw_path = _safe_resolve(job_dir, filename)

        if zst_path and zst_path.is_file():
            return _serve_zst_file(request, zst_path, filename)
        if raw_path and raw_path.is_file():
            return _serve_raw_file(request, raw_path)
        raise HTTPException(404, f"File not found: {filename}")

    return router
