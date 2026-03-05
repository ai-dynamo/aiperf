# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress router component -- owns benchmark progress state and /api/progress endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter
from pydantic import Field

from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.enums import CreditPhase
from aiperf.common.mixins.progress_tracker_mixin import (
    CombinedPhaseStats,
    ProgressTrackerMixin,
)
from aiperf.common.models import AIPerfBaseModel

ProgressDep = Annotated["ProgressRouter", component_dependency("progress")]

progress_router = APIRouter()


class ProgressResponse(AIPerfBaseModel):
    """Benchmark progress response."""

    phases: dict[CreditPhase, CombinedPhaseStats] = Field(
        default_factory=dict, description="Per-phase progress stats"
    )


class ProgressRouter(ProgressTrackerMixin, BaseRouter):
    """Owns benchmark progress state and exposes /api/progress."""

    def get_router(self) -> APIRouter:
        return progress_router


@progress_router.get("/api/progress", response_model=ProgressResponse, tags=["API"])
async def get_progress(component: ProgressDep) -> ProgressResponse:
    """Get benchmark progress with full phase stats."""
    return ProgressResponse(phases=component._progress_tracker._phases)
