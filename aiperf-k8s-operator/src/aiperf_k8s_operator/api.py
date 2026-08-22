# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated streaming API for immutable native Kubernetes results."""

from __future__ import annotations

import asyncio
import hmac
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
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
from .upload_auth import verify_results_read_token, verify_upload_signature

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RunAuthorities:
    """Immutable authorities bound to one exact Kubernetes object incarnation."""

    object_uid: str
    upload_public_key: str
    read_token_sha256: str


class RunAuthorityProvider(Protocol):
    """Resolve one transactional immutable run-authority record."""

    async def authorities(
        self, namespace: str, job_id: str, run_id: str
    ) -> RunAuthorities | None:
        """Return exact authorities, or None when the identity is unauthorized."""

    async def mark_results_ready(
        self,
        namespace: str,
        job_id: str,
        run_id: str,
        object_uid: str,
    ) -> None:
        """Publish readiness for the same object after durable manifest commit."""


class RejectRunAuthorities:
    """Fail closed when Kubernetes authority wiring is unavailable."""

    async def authorities(
        self, namespace: str, job_id: str, run_id: str
    ) -> RunAuthorities | None:
        return None

    async def mark_results_ready(
        self,
        namespace: str,
        job_id: str,
        run_id: str,
        object_uid: str,
    ) -> None:
        raise RuntimeError("result authority is not configured")


def create_app(
    settings: OperatorSettings | None = None,
    index: ResultsIndex | None = None,
    authorities: RunAuthorityProvider | None = None,
) -> FastAPI:
    """Create the dependency-injectable operator API."""
    settings = settings or OperatorSettings()
    index = index or ResultsIndex(Path(settings.artifact_root))
    authorities = authorities or RejectRunAuthorities()
    app = FastAPI(title="AIPerf Kubernetes Operator")
    app.include_router(dashboard_router)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def admin_authorized(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = settings.index_rebuild_token
        provided = authorization.removeprefix("Bearer ") if authorization else ""
        if not expected or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=401, detail="missing or invalid bearer token"
            )

    @app.get("/index/stats", dependencies=[Depends(admin_authorized)])
    async def stats() -> dict[str, int]:
        return index.stats()

    @app.post("/index/rebuild", dependencies=[Depends(admin_authorized)])
    async def rebuild() -> dict[str, str]:
        await asyncio.to_thread(index.rebuild)
        return {"status": "rebuilt"}

    async def read_authority(
        namespace: str,
        job_id: str,
        run_id: str,
        authorization: str | None,
        proxy_token: str | None,
    ) -> ResultIdentity:
        if authorization is not None and proxy_token is not None:
            raise HTTPException(status_code=401, detail="ambiguous results-read authority")
        if proxy_token is not None:
            bearer = proxy_token
        elif authorization is not None and authorization.startswith("Bearer "):
            bearer = authorization[len("Bearer ") :]
        else:
            bearer = ""
        record = await authorities.authorities(namespace, job_id, run_id)
        if record is None or not verify_results_read_token(
            record.read_token_sha256, bearer
        ):
            raise HTTPException(status_code=401, detail="invalid results-read authority")
        return ResultIdentity(namespace, job_id, run_id, record.object_uid)

    @app.get("/api/results/{namespace}/{job_id}/{run_id}/manifest")
    async def manifest(
        namespace: str,
        job_id: str,
        run_id: str,
        authorization: Annotated[str | None, Header()] = None,
        x_aiperf_results_token: Annotated[
            str | None, Header(alias="X-AIPerf-Results-Token")
        ] = None,
    ) -> dict[str, object]:
        identity = await read_authority(
            namespace, job_id, run_id, authorization, x_aiperf_results_token
        )
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
        authorization: Annotated[str | None, Header()] = None,
        x_aiperf_results_token: Annotated[
            str | None, Header(alias="X-AIPerf-Results-Token")
        ] = None,
    ) -> StreamingResponse:
        identity = await read_authority(
            namespace, job_id, run_id, authorization, x_aiperf_results_token
        )
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

    async def upload_authority(
        namespace: str,
        job_id: str,
        run_id: str,
        kind: str,
        path: str,
        content_sha256: str | None,
        content_length: str | None,
        signature: str | None,
    ) -> tuple[RunAuthorities, str, int]:
        if not content_sha256 or content_length is None or not signature:
            raise HTTPException(status_code=401, detail="missing upload authority")
        if not _SHA256.fullmatch(content_sha256):
            raise HTTPException(status_code=422, detail="invalid content digest")
        try:
            length = int(content_length)
        except ValueError as error:
            raise HTTPException(status_code=422, detail="invalid content length") from error
        if length < 0:
            raise HTTPException(status_code=422, detail="invalid content length")
        record = await authorities.authorities(namespace, job_id, run_id)
        if record is None or not verify_upload_signature(
            record.upload_public_key,
            signature,
            namespace,
            job_id,
            run_id,
            record.object_uid,
            kind,
            path,
            content_sha256,
            length,
        ):
            raise HTTPException(status_code=401, detail="invalid upload authority")
        return record, content_sha256, length

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
        x_aiperf_signature: Annotated[
            str | None, Header(alias="X-AIPerf-Signature")
        ] = None,
    ) -> Response:
        record, digest, length = await upload_authority(
            namespace,
            job_id,
            run_id,
            "artifact",
            path,
            x_aiperf_content_sha256,
            x_aiperf_content_length,
            x_aiperf_signature,
        )
        try:
            created = await index.stage_artifact(
                ResultIdentity(namespace, job_id, run_id, record.object_uid),
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
        x_aiperf_signature: Annotated[
            str | None, Header(alias="X-AIPerf-Signature")
        ] = None,
    ) -> Response:
        record, digest, length = await upload_authority(
            namespace,
            job_id,
            run_id,
            "manifest",
            "results-manifest.json",
            x_aiperf_content_sha256,
            x_aiperf_content_length,
            x_aiperf_signature,
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
        identity = ResultIdentity(namespace, job_id, run_id, record.object_uid)
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
        await authorities.mark_results_ready(
            namespace, job_id, run_id, record.object_uid
        )
        return Response(status_code=201 if created else 200)

    return app
