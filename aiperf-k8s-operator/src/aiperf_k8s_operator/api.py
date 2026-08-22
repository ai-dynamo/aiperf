# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated read API for reconciled AIPerfJob state and manifest results."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Response

from .dashboard import router as dashboard_router
from .results import ResultsIndex
from .settings import OperatorSettings


def create_app(
    settings: OperatorSettings | None = None, index: ResultsIndex | None = None
) -> FastAPI:
    """Create a dependency-injectable API application for operator and tests."""
    settings = settings or OperatorSettings()
    index = index or ResultsIndex(Path(settings.artifact_root))
    app = FastAPI(title="AIPerf Kubernetes Operator")
    app.include_router(dashboard_router)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runs/{run_id}/status")
    def status(run_id: str) -> dict[str, object]:
        return index._runs.get(run_id, None).status if run_id in index._runs else {}

    @app.get("/runs/{run_id}/manifest")
    def manifest(run_id: str) -> dict[str, object]:
        result = index.ready_manifest(run_id)
        if result is None:
            raise HTTPException(status_code=409, detail="results are not ready")
        return result

    @app.get("/runs/{run_id}/artifacts/{path:path}")
    def artifact(run_id: str, path: str) -> Response:
        result = index.artifact(run_id, path)
        if result is None:
            raise HTTPException(
                status_code=404, detail="artifact is not declared and ready"
            )
        data, content_type = result
        return Response(data, media_type=content_type)

    @app.get("/index/stats")
    def stats() -> dict[str, int]:
        return index.stats()

    def authorized(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = settings.index_rebuild_token
        provided = authorization.removeprefix("Bearer ") if authorization else ""
        if not expected or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=401, detail="missing or invalid bearer token"
            )

    @app.post("/index/rebuild", dependencies=[Depends(authorized)])
    def rebuild() -> dict[str, str]:
        index.rebuild()
        return {"status": "rebuilding"}

    return app
