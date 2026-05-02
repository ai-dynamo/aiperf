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
from typing import Any, Literal

import aiofiles
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from aiperf.operator.results_layout import EPOCH_RE, resolve_run_dir
from aiperf.operator.routers.results_schemas import (
    FileEntry,
    FileListResponse,
    JobEntry,
    ResultsHistoryListResponse,
    RunHistoryEntry,
    RunHistoryListResponse,
)

CHUNK_SIZE = 64 * 1024
PROFILE_EXPORT_FILENAME = "profile_export_aiperf.json"


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
        with zipfile.ZipFile(
            buf, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zf:
            for f in files:
                arcname = _display_name(f)
                if f.suffix == ".zst":
                    with f.open("rb") as fh, dctx.stream_reader(fh) as reader:
                        payload = reader.read()
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
        yield data[i : i + CHUNK_SIZE]


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


def _extract_model_endpoint(latest_dir: Path) -> tuple[str | None, str | None]:
    """Read ``job_spec.json`` from the run dir and extract (model, endpoint).

    Mirrors the extraction in ``operator/runs_index.py`` so the SQLite
    index and the on-disk fallback agree on what a job's "model" is.
    Returns ``(None, None)`` for any failure — older jobs predate
    ``job_spec.json`` and we don't want to fail the entire ``/results``
    listing if one of them is unparseable.
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
    models_cfg = benchmark.get("models", {})
    # models can be: {"items": [{"name": "x"}]}, {"modelNames": ["x"]}, or
    # just ["x"]. Match the shape-tolerance in operator/runs_index.py.
    model_items: list[Any]
    if isinstance(models_cfg, list):
        model_items = models_cfg
    elif isinstance(models_cfg, dict):
        model_items = models_cfg.get("items", models_cfg.get("modelNames", []))
    else:
        model_items = []
    model_name: str | None = None
    if isinstance(model_items, list) and model_items:
        first = model_items[0]
        if isinstance(first, dict):
            model_name = first.get("name")
        else:
            model_name = str(first) if first is not None else None
    endpoint_cfg = benchmark.get("endpoint", {})
    endpoint_url: str | None = None
    if isinstance(endpoint_cfg, dict):
        urls = endpoint_cfg.get("urls", endpoint_cfg.get("url", []))
        if isinstance(urls, list) and urls:
            endpoint_url = str(urls[0]) if urls[0] is not None else None
        elif isinstance(urls, str):
            endpoint_url = urls
    return model_name, endpoint_url


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


def _resolve_job_dir(
    base_dir: Path,
    namespace: str,
    job_id: str,
    epoch: str | None = None,
) -> Path:
    """Resolve a run dir under ``<base>/<ns>/<name>/``.

    Callers serving concrete result files must pass an explicit epoch so the
    UI/API cannot silently drift to a different run via ``latest.txt``.
    """
    resolved = resolve_run_dir(base_dir, namespace, job_id, epoch=epoch)
    if resolved is None:
        target = f"{namespace}/{job_id}" + (f"/runs/{epoch}" if epoch else "")
        raise HTTPException(404, f"No results for {target}")
    return resolved


def _require_epoch_for_results(namespace: str, job_id: str) -> None:
    """Reject ambiguous non-epoch result lookups.

    Final artifacts are run-scoped, not job-scoped. Requiring
    ``/runs/<epoch>`` prevents callers from mixing a live job status with the
    latest persisted run's files.
    """
    raise HTTPException(
        409,
        f"Run epoch required; use /api/v1/results/{namespace}/{job_id}/runs/<epoch>",
    )


def _validate_epoch(epoch: str) -> None:
    """Raise 422 if ``epoch`` does not match the EPOCH_RE allowlist."""
    if not EPOCH_RE.match(epoch):
        raise HTTPException(422, f"Invalid epoch: {epoch}")


def _bundle_response(job_dir: Path, bundle_name: str) -> StreamingResponse:
    """Stream a zip bundle of ``job_dir`` with Content-Disposition set."""
    return StreamingResponse(
        _stream_job_bundle(job_dir),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{bundle_name}"',
            "X-Filename": bundle_name,
        },
    )


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


def _serve_job_file(
    request: Request, job_dir: Path, filename: str
) -> StreamingResponse:
    """Serve ``filename`` from ``job_dir``, preferring .zst + content negotiation."""
    zst_path = _safe_resolve(job_dir, filename + ".zst")
    raw_path = _safe_resolve(job_dir, filename)
    if zst_path and zst_path.is_file():
        return _serve_zst_file(request, zst_path, filename)
    if raw_path and raw_path.is_file():
        return _serve_raw_file(request, raw_path)
    raise HTTPException(404, f"File not found: {filename}")


async def _build_run_history_response(
    base_dir: Path, namespace: str, job_id: str
) -> RunHistoryListResponse:
    """Resolve every run dir for a job, raising 404 when none exist."""
    from aiperf.operator.results_layout import list_runs_async

    runs = await list_runs_async(base_dir, namespace, job_id)
    if not runs:
        raise HTTPException(404, f"No runs for {namespace}/{job_id}")
    latest = next((r.epoch for r in runs if r.is_latest), None)
    return RunHistoryListResponse(
        namespace=namespace,
        job_id=job_id,
        latest_epoch=latest,
        runs=[
            RunHistoryEntry(
                epoch=r.epoch,
                mtime_epoch=r.mtime_epoch,
                file_count=r.file_count,
                total_size_bytes=r.total_size_bytes,
                is_latest=r.is_latest,
            )
            for r in runs
        ],
    )


async def _build_jobs_response(base_dir: Path) -> ResultsHistoryListResponse:
    """Scan ``base_dir`` for jobs with stored results, returning empty on miss."""
    if not base_dir.exists():
        return ResultsHistoryListResponse()
    jobs = await asyncio.to_thread(_scan_job_dirs, base_dir)
    return ResultsHistoryListResponse(jobs=jobs)


async def _build_file_list_response(
    base_dir: Path, namespace: str, job_id: str, epoch: str | None = None
) -> FileListResponse:
    """Resolve a job's run dir and enumerate its files."""
    job_dir = _resolve_job_dir(base_dir, namespace, job_id, epoch=epoch)
    files = await asyncio.to_thread(_list_job_files, job_dir)
    return FileListResponse(namespace=namespace, job_id=job_id, files=files)


def _epoch_bundle_response(
    base_dir: Path, namespace: str, job_id: str, epoch: str
) -> StreamingResponse:
    """Validate ``epoch`` and return a bundle for the matching run dir."""
    _validate_epoch(epoch)
    job_dir = _resolve_job_dir(base_dir, namespace, job_id, epoch=epoch)
    return _bundle_response(job_dir, f"{namespace}__{job_id}__{epoch}.zip")


def create_results_files_router(base_dir: Path) -> APIRouter:
    """Create the router for file listing/download endpoints.

    Args:
        base_dir: Base directory containing ``<namespace>/<job_id>/`` result files.
    """
    router = APIRouter(prefix="/api/v1", tags=["results-files"])

    @router.get("/results", response_model=ResultsHistoryListResponse)
    async def list_jobs() -> ResultsHistoryListResponse:
        """List all namespaces and jobs with stored results."""
        return await _build_jobs_response(base_dir)

    @router.get("/results/{namespace}/{job_id}.zip")
    async def download_bundle(namespace: str, job_id: str) -> StreamingResponse:
        """Reject non-epoch zip downloads; callers must pin a run epoch."""
        _require_epoch_for_results(namespace, job_id)

    @router.get("/results/{namespace}/{job_id}", response_model=FileListResponse)
    async def list_job_files(namespace: str, job_id: str) -> FileListResponse:
        """Reject non-epoch file listings; callers must pin a run epoch."""
        _require_epoch_for_results(namespace, job_id)

    @router.get(
        "/results/{namespace}/{job_id}/runs",
        response_model=RunHistoryListResponse,
    )
    async def list_runs_endpoint(namespace: str, job_id: str) -> RunHistoryListResponse:
        """List every run dir for a job, newest first, with summary metadata."""
        return await _build_run_history_response(base_dir, namespace, job_id)

    @router.get("/results/{namespace}/{job_id}/runs/{epoch}.zip")
    async def download_historical_bundle(
        namespace: str, job_id: str, epoch: str
    ) -> StreamingResponse:
        return _epoch_bundle_response(base_dir, namespace, job_id, epoch)

    @router.get(
        "/results/{namespace}/{job_id}/runs/{epoch}",
        response_model=FileListResponse,
    )
    async def list_historical_files(
        namespace: str, job_id: str, epoch: str
    ) -> FileListResponse:
        _validate_epoch(epoch)
        return await _build_file_list_response(base_dir, namespace, job_id, epoch)

    @router.get("/results/{namespace}/{job_id}/runs/{epoch}/profile_export")
    async def profile_export_quick(
        namespace: str,
        job_id: str,
        epoch: str,
        format: Literal["json"] = "json",
    ) -> Response:
        """Quick-export alias for the canonical ``profile_export_aiperf.json``.

        Mirrors the per-file route but skips the directory-listing roundtrip
        the artifacts table normally performs. Reads the canonical artifact
        from the resolved run dir, transparently decompressing the ``.zst``
        companion when the uncompressed file is absent. Returns
        ``application/json`` with ``Content-Disposition: attachment;
        filename="profile_export_aiperf.json"``.

        Raises 404 if the artifact is absent (run still warming up, the
        sidecar's ready marker has gated the directory upstream of this
        router, or this run type doesn't produce a profile export).

        ``format`` is currently constrained to ``"json"``; the parameter
        exists so future shortcuts (csv/parquet) can be added without a
        new route.
        """
        del format  # Reserved for future format shortcuts; only "json" today.
        _validate_epoch(epoch)
        job_dir = _resolve_job_dir(base_dir, namespace, job_id, epoch=epoch)
        try:
            payload = await asyncio.to_thread(_read_profile_export_bytes, job_dir)
        except FileNotFoundError:
            raise HTTPException(
                404,
                f"File not found: {PROFILE_EXPORT_FILENAME}",
            ) from None
        return Response(
            content=payload,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{PROFILE_EXPORT_FILENAME}"'
                ),
                "X-Filename": PROFILE_EXPORT_FILENAME,
            },
        )

    @router.get("/results/{namespace}/{job_id}/runs/{epoch}/{filename:path}")
    async def download_historical_file(
        namespace: str,
        job_id: str,
        epoch: str,
        filename: str,
        *,
        request: Request,
    ) -> StreamingResponse:
        _validate_epoch(epoch)
        job_dir = _resolve_job_dir(base_dir, namespace, job_id, epoch=epoch)
        return _serve_job_file(request, job_dir, filename)

    @router.get("/results/{namespace}/{job_id}/{filename:path}")
    async def download_file(
        namespace: str, job_id: str, filename: str, request: Request
    ) -> StreamingResponse:
        """Reject non-epoch file downloads; callers must pin a run epoch."""
        _require_epoch_for_results(namespace, job_id)

    return router
