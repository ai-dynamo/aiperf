# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency injection for the FastAPI service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends
from starlette.requests import HTTPConnection

if TYPE_CHECKING:
    from aiperf.api.api_service import FastAPIService


def get_service(conn: HTTPConnection) -> FastAPIService:
    """Get FastAPIService from app state. Works for both HTTP and WebSocket."""
    service = getattr(conn.app.state, "service", None)
    if service is None:
        raise RuntimeError("Service not initialized in app.state")
    return service


ServiceDep = Annotated["FastAPIService", Depends(get_service)]
