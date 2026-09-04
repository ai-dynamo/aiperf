# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep aggregate artifact routes for the operator UI API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from aiperf.common.results_markers import EPOCH_RE
from aiperf.operator.results_layout import resolve_sweep_dir
from aiperf.operator.routers._path_params import validate_results_path_params
from aiperf.operator.routers.results_files_io import (
    _list_artifact_files,
    _read_profile_export_bytes,
    _safe_resolve,
    _serve_artifact_file,
    _stream_artifact_bundle,
)
from aiperf.operator.routers.results_schemas import FileListResponse

SWEEP_AGGREGATE_DIRS = ("sweep_aggregate",)
SWEEP_PROFILE_EXPORT_FILENAMES = {
    "json": "profile_export_aiperf_aggregate.json",
    "csv": "profile_export_aiperf_aggregate.csv",
}


def _validate_epoch_param(epoch: str) -> None:
    if not EPOCH_RE.match(epoch):
        raise HTTPException(422, f"Invalid epoch: {epoch}")


def _resolve_sweep_epoch_dir(
    base_dir: Path, namespace: str, name: str, epoch: str
) -> Path:
    validate_results_path_params(namespace, name)
    _validate_epoch_param(epoch)
    sweep_dir = resolve_sweep_dir(base_dir, namespace, name, epoch=epoch)
    if sweep_dir is None:
        raise HTTPException(
            404, f"Sweep epoch not found: {namespace}/{name} epoch={epoch}"
        )
    return sweep_dir


async def _build_sweep_artifact_list_response(
    base_dir: Path, namespace: str, name: str, epoch: str
) -> FileListResponse:
    sweep_dir = _resolve_sweep_epoch_dir(base_dir, namespace, name, epoch)
    files = await asyncio.to_thread(
        _list_artifact_files, sweep_dir, SWEEP_AGGREGATE_DIRS
    )
    return FileListResponse(namespace=namespace, job_id=name, files=files)


def _sweep_bundle_response(
    base_dir: Path, namespace: str, name: str, epoch: str
) -> StreamingResponse:
    sweep_dir = _resolve_sweep_epoch_dir(base_dir, namespace, name, epoch)
    bundle_name = f"{namespace}__{name}__{epoch}__aggregate-artifacts.zip"
    return StreamingResponse(
        _stream_artifact_bundle(sweep_dir, SWEEP_AGGREGATE_DIRS),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{bundle_name}"',
            "X-Filename": bundle_name,
        },
    )


async def _sweep_profile_export_response(
    base_dir: Path,
    namespace: str,
    name: str,
    epoch: str,
    *,
    format: Literal["json", "csv"],
) -> Response:
    sweep_dir = _resolve_sweep_epoch_dir(base_dir, namespace, name, epoch)
    filename = SWEEP_PROFILE_EXPORT_FILENAMES[format]
    try:
        if format == "json":
            aggregate_dir = sweep_dir / "sweep_aggregate"
            payload = await asyncio.to_thread(
                _read_profile_export_bytes, aggregate_dir, filename
            )
        else:
            relative_filename = f"sweep_aggregate/{filename}"
            scoped_target = _safe_resolve(sweep_dir, relative_filename)
            link_path = sweep_dir / relative_filename
            if (
                scoped_target is None
                or link_path.is_symlink()
                or not scoped_target.is_file()
            ):
                raise FileNotFoundError(filename)
            payload = await asyncio.to_thread(scoped_target.read_bytes)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {filename}") from None
    media_type = "application/json" if format == "json" else "text/csv"
    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Filename": filename,
        },
    )


def register_sweep_artifact_routes(router: APIRouter, base_dir: Path) -> None:
    @router.get(
        "/sweeps/{namespace}/{name}/epochs/{epoch}/artifacts.zip",
    )
    async def download_sweep_artifacts_bundle(
        namespace: str, name: str, epoch: str
    ) -> StreamingResponse:
        return _sweep_bundle_response(base_dir, namespace, name, epoch)

    @router.get(
        "/sweeps/{namespace}/{name}/epochs/{epoch}/artifacts",
        response_model=FileListResponse,
    )
    async def list_sweep_artifacts(
        namespace: str, name: str, epoch: str
    ) -> FileListResponse:
        return await _build_sweep_artifact_list_response(
            base_dir, namespace, name, epoch
        )

    @router.get("/sweeps/{namespace}/{name}/epochs/{epoch}/artifacts/profile_export")
    async def sweep_profile_export_quick(
        namespace: str,
        name: str,
        epoch: str,
        format: Literal["json", "csv"] = "json",
    ) -> Response:
        return await _sweep_profile_export_response(
            base_dir, namespace, name, epoch, format=format
        )

    @router.get("/sweeps/{namespace}/{name}/epochs/{epoch}/artifacts/{filename:path}")
    async def download_sweep_artifact_file(
        namespace: str,
        name: str,
        epoch: str,
        filename: str,
        *,
        request: Request,
    ) -> StreamingResponse:
        sweep_dir = _resolve_sweep_epoch_dir(base_dir, namespace, name, epoch)
        return _serve_artifact_file(
            request,
            sweep_dir,
            filename,
            allowed_relative_dirs=SWEEP_AGGREGATE_DIRS,
        )
