# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming API for immutable native Kubernetes results."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import FastAPI, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse

from .dashboard import router as dashboard_router
from .results import (
    MAX_MANIFEST_BYTES,
    ResultIdentity,
    ResultsExpired,
    ResultsIndex,
    UploadConflict,
    UploadInvalid,
    UploadTooLarge,
)
from .settings import OperatorSettings

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LENGTH = re.compile(r"^(?:0|[1-9][0-9]*)$")
_LIFECYCLE_TIMEOUT_SECONDS = 5.0
_LOGGER = logging.getLogger(__name__)


class ResultsLifecycle(Protocol):
    """Best-effort Kubernetes lifecycle update after durable publication."""

    async def mark_results_ready(
        self,
        namespace: str,
        job_id: str,
        run_id: str,
    ) -> None:
        """Best-effort publish readiness for the current matching AIPerfJob."""


def create_app(
    settings: OperatorSettings | None = None,
    index: ResultsIndex | None = None,
    lifecycle: ResultsLifecycle | None = None,
) -> FastAPI:
    """Create the dependency-injectable operator API."""
    settings = settings or OperatorSettings()
    index = index or ResultsIndex(Path(settings.artifact_root))
    app = FastAPI(title="AIPerf Kubernetes Operator")
    app.include_router(dashboard_router)
    lifecycle_tasks: set[asyncio.Task[None]] = set()

    async def update_lifecycle(
        namespace: str,
        job_id: str,
        run_id: str,
    ) -> None:
        if lifecycle is None:
            return
        try:
            async with asyncio.timeout(_LIFECYCLE_TIMEOUT_SECONDS):
                await lifecycle.mark_results_ready(namespace, job_id, run_id)
        except TimeoutError:
            _LOGGER.warning(
                "durable results committed but lifecycle update timed out",
                extra={
                    "namespace": namespace,
                    "job_id": job_id,
                    "run_id": run_id,
                },
            )
        except Exception:
            _LOGGER.exception(
                "durable results committed but lifecycle update failed",
                extra={
                    "namespace": namespace,
                    "job_id": job_id,
                    "run_id": run_id,
                },
            )

    def detach_lifecycle_update(namespace: str, job_id: str, run_id: str) -> None:
        task = asyncio.create_task(update_lifecycle(namespace, job_id, run_id))
        lifecycle_tasks.add(task)
        task.add_done_callback(lifecycle_tasks.discard)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/index/stats")
    async def stats() -> dict[str, int | float]:
        return await asyncio.to_thread(index.stats)

    @app.get("/api/results/{namespace}/{job_id}/{run_id}/manifest")
    async def manifest(
        namespace: str,
        job_id: str,
        run_id: str,
    ) -> dict[str, object]:
        identity = ResultIdentity(namespace, job_id, run_id)
        try:
            result = index.ready_manifest(identity)
        except ResultsExpired as error:
            raise HTTPException(status_code=410, detail=str(error)) from error
        if result is None:
            raise HTTPException(status_code=409, detail="results are not ready")
        return result

    @app.get("/api/results/{namespace}/{job_id}/{run_id}/artifacts/{path:path}")
    async def artifact(
        namespace: str,
        job_id: str,
        run_id: str,
        path: str,
    ) -> StreamingResponse:
        identity = ResultIdentity(namespace, job_id, run_id)
        try:
            opened = await asyncio.to_thread(index.open_artifact, identity, path)
        except ResultsExpired as error:
            raise HTTPException(status_code=410, detail=str(error)) from error
        except (FileNotFoundError, OSError, UploadConflict, UploadInvalid):
            raise HTTPException(
                status_code=404, detail="artifact is not declared and ready"
            ) from None
        return StreamingResponse(
            opened.chunks(),
            media_type=opened.content_type,
            headers={"Content-Length": str(opened.length)},
        )

    def upload_metadata(
        content_sha256: str | None,
        content_length: str | None,
        wire_length: str | None,
    ) -> tuple[str, int]:
        if content_sha256 is None or not _SHA256.fullmatch(content_sha256):
            raise HTTPException(status_code=422, detail="invalid content digest")
        if content_length is None or not _LENGTH.fullmatch(content_length):
            raise HTTPException(status_code=422, detail="invalid content length")
        if wire_length is None or wire_length != content_length:
            raise HTTPException(
                status_code=422, detail="HTTP content length is inconsistent"
            )
        return content_sha256, int(content_length)

    def upload_error(error: Exception) -> HTTPException:
        if isinstance(error, ResultsExpired):
            return HTTPException(status_code=410, detail=str(error))
        if isinstance(error, UploadTooLarge):
            return HTTPException(status_code=413, detail=str(error))
        if isinstance(error, (UploadConflict, OSError)):
            return HTTPException(status_code=409, detail=str(error))
        return HTTPException(status_code=422, detail=str(error))

    @app.put(
        "/api/uploads/{namespace}/{job_id}/{run_id}/artifacts/{path:path}",
        status_code=201,
    )
    async def upload_artifact(
        namespace: str,
        job_id: str,
        run_id: str,
        path: str,
        request: Request,
        x_aiperf_content_sha256: Annotated[
            str | None, Header(alias="X-AIPerf-Content-SHA256")
        ] = None,
        x_aiperf_content_length: Annotated[
            str | None, Header(alias="X-AIPerf-Content-Length")
        ] = None,
    ) -> Response:
        digest, length = upload_metadata(
            x_aiperf_content_sha256,
            x_aiperf_content_length,
            request.headers.get("content-length"),
        )
        try:
            created = await index.stage_artifact(
                ResultIdentity(namespace, job_id, run_id),
                path,
                request.stream(),
                digest,
                length,
            )
        except (
            OSError,
            ResultsExpired,
            UploadConflict,
            UploadInvalid,
            UploadTooLarge,
        ) as error:
            raise upload_error(error) from error
        return Response(status_code=201 if created else 200)

    @app.post("/api/uploads/{namespace}/{job_id}/{run_id}/manifest", status_code=201)
    async def upload_manifest(
        namespace: str,
        job_id: str,
        run_id: str,
        request: Request,
        x_aiperf_content_sha256: Annotated[
            str | None, Header(alias="X-AIPerf-Content-SHA256")
        ] = None,
        x_aiperf_content_length: Annotated[
            str | None, Header(alias="X-AIPerf-Content-Length")
        ] = None,
    ) -> Response:
        digest, length = upload_metadata(
            x_aiperf_content_sha256,
            x_aiperf_content_length,
            request.headers.get("content-length"),
        )
        if length > MAX_MANIFEST_BYTES:
            raise HTTPException(status_code=413, detail="manifest exceeds upload limit")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_MANIFEST_BYTES or len(body) > length:
                raise HTTPException(
                    status_code=413, detail="manifest exceeds its declared length"
                )
        identity = ResultIdentity(namespace, job_id, run_id)
        try:
            created = await asyncio.to_thread(
                index.commit_manifest, identity, bytes(body), digest, length
            )
        except (
            OSError,
            ResultsExpired,
            UploadConflict,
            UploadInvalid,
            UploadTooLarge,
        ) as error:
            raise upload_error(error) from error
        detach_lifecycle_update(namespace, job_id, run_id)
        return Response(status_code=201 if created else 200)

    return app
