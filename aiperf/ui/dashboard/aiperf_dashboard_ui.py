# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import AIPerfUIType, WorkerStatus
from aiperf.common.factories import AIPerfUIFactory
from aiperf.common.hooks import (
    on_profiling_progress,
    on_records_progress,
    on_start,
    on_stop,
    on_warmup_progress,
    on_worker_status_summary,
    on_worker_update,
)
from aiperf.common.models import (
    RecordsStats,
    RequestsStats,
    WorkerStats,
)
from aiperf.common.protocols import AIPerfUIProtocol
from aiperf.ui.base_ui import BaseAIPerfUI
from aiperf.ui.dashboard.aiperf_textual_app import AIPerfTextualApp
from aiperf.ui.dashboard.rich_log_viewer import LogConsumer


@implements_protocol(AIPerfUIProtocol)
@AIPerfUIFactory.register(AIPerfUIType.DASHBOARD)
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
        log_queue: multiprocessing.Queue,
        user_config: UserConfig,
        **kwargs,
    ) -> None:
        super().__init__(user_config=user_config, **kwargs)
        self.user_config = user_config
        self.app: AIPerfTextualApp = AIPerfTextualApp()
        # Setup the log consumer to consume log records from the shared log queue
        self.log_consumer: LogConsumer = LogConsumer(log_queue=log_queue, app=self.app)
        self.attach_child_lifecycle(self.log_consumer)  # type: ignore

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

    @on_records_progress
    async def _on_records_progress(self, records_stats: RecordsStats) -> None:
        """Forward records progress updates to the Textual App."""
        await self.app.on_records_progress(records_stats)

    @on_profiling_progress
    async def _on_profiling_progress(self, profiling_stats: RequestsStats) -> None:
        """Forward requests phase progress updates to the Textual App."""
        await self.app.on_profiling_progress(profiling_stats)

    @on_warmup_progress
    async def _on_warmup_progress(self, warmup_stats: RequestsStats) -> None:
        """Forward warmup progress updates to the Textual App."""
        await self.app.on_warmup_progress(warmup_stats)

    @on_worker_update
    async def _on_worker_update(self, worker_id: str, worker_stats: WorkerStats):
        """Forward worker updates to the Textual App."""
        await self.app.on_worker_update(worker_id, worker_stats)

    @on_worker_status_summary
    async def _on_worker_status_summary(self, worker_status_summary: dict[str, WorkerStatus]) -> None:  # fmt: skip
        """Forward worker status summary updates to the Textual App."""
        await self.app.on_worker_status_summary(worker_status_summary)
