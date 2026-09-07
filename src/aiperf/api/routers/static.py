# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static router for AIPerf API.

Provides endpoints for serving static HTML files for the dashboard and index
pages. ``/dashboard`` serves the single-file legacy dashboard; ``/dashboard-v2``
serves the modular Preact-based v2 app mirrored from the operator UI stack.
"""

from __future__ import annotations

from mimetypes import guess_type
from pathlib import Path

import aiofiles
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from aiperf.api.routers.base_router import BaseRouter

static_router = APIRouter(tags=["Static"])

_STATIC_DIR = (Path(__file__).parent.parent / "static").resolve()
_STATIC_V2_DIR = (Path(__file__).parent.parent / "static-v2").resolve()
_STATIC_VENDOR_DIR = _STATIC_DIR / "vendor"


class StaticRouter(BaseRouter):
    """Static HTML file serving for dashboard and index pages."""

    def get_router(self) -> APIRouter:
        return static_router


async def _read_static(filename: str) -> str:
    """Read a legacy static HTML file with path traversal protection."""
    file_path = (_STATIC_DIR / filename).resolve()
    if not file_path.is_relative_to(_STATIC_DIR):
        raise HTTPException(400, "Invalid filename")

    try:
        async with aiofiles.open(file_path, encoding="utf-8") as f:
            return await f.read()
    except FileNotFoundError:
        raise HTTPException(404, f"{filename} not found") from None


def _guess_content_type(file_path: Path) -> str:
    """Sniff content type from extension, with an override for .js/.mjs.

    Browsers must see JS module files served as ``application/javascript``
    regardless of what ``mimetypes`` guesses from the local OS config.
    """
    content_type, _ = guess_type(str(file_path))
    if file_path.suffix in (".js", ".mjs"):
        return "application/javascript; charset=utf-8"
    if file_path.suffix == ".css" and content_type is None:
        return "text/css; charset=utf-8"
    return content_type or "application/octet-stream"


async def _read_vendor_asset(base_dir: Path, rel_path: str) -> tuple[bytes, str]:
    """Read a vendored third-party asset, returning ``(bytes, content_type)``.

    Shared by the legacy ``static/vendor/`` and ``static-v2/vendor/`` trees.
    Enforces path traversal protection: resolved path must stay under
    ``base_dir``.
    """
    file_path = (base_dir / rel_path).resolve()
    if not file_path.is_relative_to(base_dir):
        raise HTTPException(400, "Invalid asset path")

    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = await f.read()
    except FileNotFoundError:
        raise HTTPException(404, f"{rel_path} not found") from None
    except IsADirectoryError:
        raise HTTPException(404, f"{rel_path} is a directory") from None

    return data, _guess_content_type(file_path)


@static_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the index page."""
    return HTMLResponse(await _read_static("index.html"))


@static_router.get("/static-vendor/{asset:path}", include_in_schema=False)
async def static_vendor_asset(asset: str) -> Response:
    """Serve a vendored third-party asset (e.g. PrismJS) for the legacy dashboard."""
    data, content_type = await _read_vendor_asset(_STATIC_VENDOR_DIR, asset)
    return Response(content=data, media_type=content_type)


@static_router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the legacy single-file dashboard page."""
    return HTMLResponse(await _read_static("dashboard.html"))


@static_router.get("/dashboard-v2", include_in_schema=False)
async def dashboard_v2_redirect() -> RedirectResponse:
    """Redirect to the trailing-slash form so relative asset URLs resolve.

    ``index.html`` references ``./style.css`` and ``./app.js``; the browser
    resolves those against the URL's directory, which is ``/`` for
    ``/dashboard-v2`` and ``/dashboard-v2/`` for ``/dashboard-v2/``. Without
    the redirect the assets 404.
    """
    return RedirectResponse(url="/dashboard-v2/", status_code=307)


@static_router.get(
    "/dashboard-v2/", response_class=HTMLResponse, include_in_schema=False
)
async def dashboard_v2_index() -> HTMLResponse:
    """Serve the v2 dashboard entrypoint (``static-v2/index.html``)."""
    data, _ = await _read_vendor_asset(_STATIC_V2_DIR, "index.html")
    return HTMLResponse(data.decode("utf-8"))


@static_router.get("/dashboard-v2/{asset:path}", include_in_schema=False)
async def dashboard_v2_asset(asset: str) -> Response:
    """Serve any asset under ``static-v2/`` (e.g. ``app.js``, ``lib/state.js``, ``vendor/*``)."""
    if asset in ("", "/"):
        data, _ = await _read_vendor_asset(_STATIC_V2_DIR, "index.html")
        return HTMLResponse(data.decode("utf-8"))
    data, content_type = await _read_vendor_asset(_STATIC_V2_DIR, asset)
    return Response(content=data, media_type=content_type)
