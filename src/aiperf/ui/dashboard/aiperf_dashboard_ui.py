# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.hooks import (
    AIPerfHook,
    on_start,
    on_stop,
)
from aiperf.config import BenchmarkRun
from aiperf.ui.base_ui import BaseAIPerfUI
from aiperf.ui.dashboard.aiperf_textual_app import AIPerfTextualApp
from aiperf.ui.dashboard.rich_log_viewer import LogConsumer

if TYPE_CHECKING:
    from aiperf.common.logging import LogQueue
    from aiperf.controller.system_controller import SystemController


class AIPerfDashboardUI(BaseAIPerfUI):
    """
    AIPerf Dashboard UI.

    This is the main Dashboard UI class that implements the AIPerfUIProtocol. It is
    responsible for managing the Textual App, its lifecycle, and passing the progress
    updates to the Textual App. It also manages the lifecycle of the log consumer,
    which is responsible for consuming log records from the shared log queue and
    displaying them in the log viewer.

    The reason for this wrapper is that the internal lifecycle of the Textual App is
    handled by Textual, and it is not fully compatible with our AIPerf lifecycle.
    """

    def __init__(
        self,
        log_queue: LogQueue,
        run: BenchmarkRun,
        controller: SystemController,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            controller=controller,
            **kwargs,
        )
        self.controller = controller
        self.run = run
        self.app: AIPerfTextualApp = AIPerfTextualApp(run=run, controller=controller)
        # Setup the log consumer to consume log records from the shared log queue
        self.log_consumer: LogConsumer = LogConsumer(log_queue=log_queue, app=self.app)
        self.attach_child_lifecycle(self.log_consumer)  # type: ignore

        # Attach the hooks directly to the function on the app, to avoid the extra function call overhead
        self.attach_hook(AIPerfHook.ON_RECORDS_PROGRESS, self.app.on_records_progress)
        self.attach_hook(AIPerfHook.ON_PHASE_PROGRESS, self.app.on_phase_progress)
        self.attach_hook(AIPerfHook.ON_WORKER_UPDATE, self.app.on_worker_update)
        self.attach_hook(
            AIPerfHook.ON_WORKER_STATUS_SUMMARY, self.app.on_worker_status_summary
        )
        self.attach_hook(AIPerfHook.ON_REALTIME_METRICS, self.app.on_realtime_metrics)
        self.attach_hook(
            AIPerfHook.ON_REALTIME_TELEMETRY_METRICS,
            self.app.on_realtime_telemetry_metrics,
        )

    @on_start
    async def _run_app(self) -> None:
        """Run the enhanced Dashboard application."""
        self.debug("Starting AIPerf Dashboard UI...")
        # Start the Textual App in the background
        self.execute_async(self.app.run_async())

    @on_stop
    async def _on_stop(self) -> None:
        """Stop the Dashboard application gracefully."""
        self.debug("Shutting down Dashboard UI")
        self.app.exit(return_code=0)
