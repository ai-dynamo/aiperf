# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workers router component -- owns worker tracker state and /api/workers endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter

from aiperf.api.models.responses import WorkersResponse
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.mixins.worker_tracker_mixin import WorkerTrackerMixin

WorkersDep = Annotated["WorkersRouter", component_dependency("workers")]

workers_router = APIRouter()


class WorkersRouter(WorkerTrackerMixin, BaseRouter):
    """Owns worker tracker state and exposes /api/workers."""

    def get_router(self) -> APIRouter:
        return workers_router


@workers_router.get("/api/workers", response_model=WorkersResponse, tags=["API"])
async def get_workers(component: WorkersDep) -> WorkersResponse:
    """Get worker status with full stats."""
    return WorkersResponse(workers=component._worker_tracker.workers)
