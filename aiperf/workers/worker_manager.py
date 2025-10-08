# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import time

from pydantic import Field

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    DEFAULT_MAX_WORKERS_CAP,
    DEFAULT_WORKER_CHECK_INTERVAL,
    DEFAULT_WORKER_ERROR_RECOVERY_TIME,
    DEFAULT_WORKER_HIGH_LOAD_CPU_USAGE,
    DEFAULT_WORKER_HIGH_LOAD_RECOVERY_TIME,
    DEFAULT_WORKER_STALE_TIME,
    DEFAULT_WORKER_STATUS_SUMMARY_INTERVAL,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import MessageType, ServiceType
from aiperf.common.enums.worker_enums import WorkerStatus
from aiperf.common.hooks import background_task, on_message, on_start, on_stop
from aiperf.common.messages import (
    ShutdownWorkersCommand,
    SpawnWorkersCommand,
    WorkerHealthMessage,
)
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.models.progress_models import WorkerStats


class WorkerStatusInfo(WorkerStats):
    """Information about a worker's status."""

    worker_id: str = Field(..., description="The ID of the worker")
    last_error_ns: int | None = Field(
        default=None,
        description="The last time the worker had an error",
    )
    last_high_load_ns: int | None = Field(
        default=None,
        description="The last time the worker was in high load",
    )


class WorkerManager(BaseComponentService):
    """
    The WorkerManager service is primary responsibility to manage the worker processes.
    It will spawn the workers, monitor their health, and stop them when the service is stopped.
    In the future it will also be responsible for the auto-scaling of the workers.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

        self.trace("WorkerManager.__init__")
        self.worker_infos: dict[str, WorkerStatusInfo] = {}

        self.cpu_count = multiprocessing.cpu_count()
        self.debug(lambda: f"Detected {self.cpu_count} CPU cores/threads")

        self.max_concurrency = self.user_config.loadgen.concurrency
        self.max_workers = self.service_config.workers.max
        if self.max_workers is None:
            # Default to 75% of the CPU cores - 1, with a cap of DEFAULT_MAX_WORKERS_CAP, and a minimum of 1
            self.max_workers = max(
                1, min(int(self.cpu_count * 0.75) - 1, DEFAULT_MAX_WORKERS_CAP)
            )
            self.debug(
                lambda: f"Auto-setting max workers to {self.max_workers} due to no max workers specified."
            )

        # Cap the worker count to the max concurrency, but only if the user is in concurrency mode.
        if self.max_concurrency and self.max_concurrency < self.max_workers:
            self.max_workers = self.max_concurrency
            self.debug(
                lambda: f"Capping max workers to {self.max_workers} due to concurrency."
            )

        # Ensure we have at least the min workers
        self.max_workers = max(
            self.max_workers,
            self.service_config.workers.min or 1,
        )
        self.initial_workers = self.max_workers

    @on_start
    async def _start(self) -> None:
        """Start worker manager-specific components."""
        self.debug("WorkerManager starting")

        await self.send_command_and_wait_for_response(
            SpawnWorkersCommand(
                service_id=self.service_id,
                num_workers=self.initial_workers,
                # Target the system controller directly to avoid broadcasting to all services.
                target_service_type=ServiceType.SYSTEM_CONTROLLER,
            )
        )
        self.debug("WorkerManager started")

    @on_stop
    async def _stop(self) -> None:
        self.debug("WorkerManager stopping")

        await self.publish(
            ShutdownWorkersCommand(
                service_id=self.service_id,
                all_workers=True,
                # Target the system controller directly to avoid broadcasting to all services.
                target_service_type=ServiceType.SYSTEM_CONTROLLER,
            )
        )

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        worker_id = message.service_id
        info = self.worker_infos.get(worker_id)
        if not info:
            info = WorkerStatusInfo(
                worker_id=worker_id,
                last_update_ns=time.time_ns(),
                status=WorkerStatus.HEALTHY,
                health=message.health,
                task_stats=message.task_stats,
            )
            self.worker_infos[worker_id] = info
        self._update_worker_status(info, message)

    def _update_worker_status(
        self, info: WorkerStatusInfo, message: WorkerHealthMessage
    ) -> None:
        """Check the status of a worker."""
        info.last_update_ns = time.time_ns()
        # Error Status
        if message.task_stats.failed > info.task_stats.failed:
            info.last_error_ns = time.time_ns()
            info.status = WorkerStatus.ERROR
        elif (time.time_ns() - (info.last_error_ns or 0)) / NANOS_PER_SECOND < DEFAULT_WORKER_ERROR_RECOVERY_TIME:  # fmt: skip
            info.status = WorkerStatus.ERROR

        # High Load Status
        elif message.health.cpu_usage > DEFAULT_WORKER_HIGH_LOAD_CPU_USAGE:
            info.last_high_load_ns = time.time_ns()
            info.status = WorkerStatus.HIGH_LOAD
        elif (time.time_ns() - (info.last_high_load_ns or 0)) / NANOS_PER_SECOND < DEFAULT_WORKER_HIGH_LOAD_RECOVERY_TIME:  # fmt: skip
            info.status = WorkerStatus.HIGH_LOAD

        # Idle Status
        elif message.task_stats.total == 0 or message.task_stats.in_progress == 0:
            info.status = WorkerStatus.IDLE

        # Healthy Status
        else:
            info.status = WorkerStatus.HEALTHY

        info.health = message.health
        info.task_stats = message.task_stats

    @background_task(immediate=False, interval=DEFAULT_WORKER_CHECK_INTERVAL)
    async def _worker_status_loop(self) -> None:
        """Check the status of all workers."""
        self.debug("Checking worker status")

        for _, info in self.worker_infos.items():
            if (time.time_ns() - (info.last_update_ns or 0)) / NANOS_PER_SECOND > DEFAULT_WORKER_STALE_TIME:  # fmt: skip
                info.status = WorkerStatus.STALE

    @background_task(immediate=False, interval=DEFAULT_WORKER_STATUS_SUMMARY_INTERVAL)
    async def _worker_summary_loop(self) -> None:
        """Generate a summary of the worker status."""
        summary = WorkerStatusSummaryMessage(
            service_id=self.service_id,
            worker_statuses={
                worker_id: info.status for worker_id, info in self.worker_infos.items()
            },
        )
        self.debug(lambda: f"Publishing worker status summary: {summary}")
        await self.publish(summary)


def main() -> None:
    bootstrap_and_run_service(WorkerManager)


if __name__ == "__main__":
    main()
