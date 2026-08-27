# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress router component -- owns benchmark progress state and /api/progress endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter

from aiperf.api.models.responses import ProgressResponse
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.enums import MessageType, SystemState
from aiperf.common.hooks import on_message
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    ResultsExportedMessage,
    SystemStateChangedMessage,
)
from aiperf.common.mixins.progress_tracker_mixin import ProgressTrackerMixin
from aiperf.common.mixins.realtime_metrics_mixin import RealtimeMetricsMixin

ProgressDep = Annotated["ProgressRouter", component_dependency("progress")]

progress_router = APIRouter()


class ProgressRouter(RealtimeMetricsMixin, ProgressTrackerMixin, BaseRouter):
    """Owns benchmark progress state and exposes /api/progress."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Flips True only after the SystemController publishes
        # ResultsExportedMessage -- i.e. after ExporterManager.export_data()
        # has completed.
        self._results_exported: bool = False
        self._controller_failure: str | None = None
        # Mirrors SystemController's outer-lifecycle SystemState. Updated via
        # SYSTEM_STATE_CHANGED bus messages, so the API service (its own
        # process) can surface controller-side state on /api/progress without
        # an in-process controller handle.
        self._system_state: SystemState = SystemState.INITIALIZING

    def get_router(self) -> APIRouter:
        return progress_router

    @on_message(MessageType.RESULTS_EXPORTED)
    async def _on_results_exported(self, _message: ResultsExportedMessage) -> None:
        """Record that the controller has finished writing artifacts to disk."""
        self._results_exported = True

    @on_message(MessageType.SERVICE_ERROR)
    async def _on_service_error(self, message: BaseServiceErrorMessage) -> None:
        """Push a controller-plane failure before its pod exits."""
        self._controller_failure = f"{message.service_id}: {message.error.message}"

    @on_message(MessageType.SYSTEM_STATE_CHANGED)
    async def _on_system_state_changed(
        self, message: SystemStateChangedMessage
    ) -> None:
        """Record the controller's most-recent outer-lifecycle SystemState.

        Guards against late/duplicate delivery (e.g. from ZMQ retry/reconnect)
        walking the reported state backwards, mirroring the monotonicity
        check ``SystemController._set_system_state`` applies before publishing.
        """
        if message.state.rank < self._system_state.rank:
            return
        self._system_state = message.state


@progress_router.get("/api/progress", response_model=ProgressResponse, tags=["API"])
async def get_progress(component: ProgressDep) -> ProgressResponse:
    """Get benchmark progress with full phase stats."""
    return ProgressResponse(
        phases=component._progress_tracker._phases,
        results_exported=component._results_exported,
        system_state=component._system_state,
    )
