# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from aiperf.common.base_component_service import BaseComponentService

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType, MessageType
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_command, on_message
from aiperf.common.messages import WorkerHealthMessage, WorkerStartupStateMessage
from aiperf.plugin.enums import ServiceType
from aiperf.ui.utils import format_bytes
from aiperf.workers.worker_group_state import (
    WorkerStatusInfo,
    build_worker_status_summary,
    mark_stale_workers,
    update_worker_status,
)


class WorkerManager(BaseComponentService):
    """Monitors worker health and publishes status summaries to the message bus."""

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )
        self.worker_infos: dict[str, WorkerStatusInfo] = {}

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        worker_id = message.service_id
        info = self._get_or_create_worker_info(worker_id)
        self._update_worker_status(info, message)

    @on_message(MessageType.WORKER_STARTUP_STATE)
    async def _on_worker_startup_state(
        self, message: WorkerStartupStateMessage
    ) -> None:
        info = self._get_or_create_worker_info(message.service_id)
        info.startup_state = message.startup_state
        info.startup_state_updated_ns = message.request_ns
        await self._publish_worker_summary()

    def _get_or_create_worker_info(self, worker_id: str) -> WorkerStatusInfo:
        info = self.worker_infos.get(worker_id)
        if info is None:
            info = WorkerStatusInfo(worker_id=worker_id)
            self.worker_infos[worker_id] = info
        return info

    def _update_worker_status(
        self, info: WorkerStatusInfo, message: WorkerHealthMessage
    ) -> None:
        """Check the status of a worker."""
        update_worker_status(info, message, warning=self.warning)

    @background_task(immediate=False, interval=Environment.WORKER.CHECK_INTERVAL)
    async def _worker_status_loop(self) -> None:
        """Check the status of all workers."""
        self.debug("Checking worker status")
        mark_stale_workers(self.worker_infos)

    @background_task(
        immediate=False, interval=Environment.WORKER.STATUS_SUMMARY_INTERVAL
    )
    async def _worker_summary_loop(self) -> None:
        """Generate a summary of the worker status."""
        await self._publish_worker_summary()

    async def _publish_worker_summary(self) -> None:
        """Publish the current worker status and startup-state summary."""
        await self.publish(
            build_worker_status_summary(
                service_id=self.service_id,
                worker_infos=self.worker_infos,
            )
        )

    @on_command(CommandType.REPORT_WORKER_STATUS_SUMMARY)
    async def _on_report_worker_status_summary(self, message: Command) -> None:
        """Publish an immediate worker status summary on controller request."""
        await self._publish_worker_summary()

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _on_profile_complete(self, message: Command) -> None:
        """Handle profile complete by printing worker stats."""
        await self._print_worker_stats("Profile Complete")

    @on_command(CommandType.PROFILE_CANCEL)
    async def _on_profile_cancel(self, message: Command) -> None:
        """Handle profile cancel by printing worker stats."""
        await self._print_worker_stats("Profile Cancelled")

    async def _print_worker_stats(self, title: str) -> None:
        """Print worker process stats using rich."""
        if not self.worker_infos:
            return

        console = Console()

        table = Table(title=f"Worker Process Stats | {title}")
        table.add_column("Worker", justify="left", style="cyan")
        table.add_column("RSS (MB)", justify="right", style="green")
        table.add_column("CPU (%)", justify="right", style="yellow")
        table.add_column("Threads", justify="right")
        table.add_column("Vol CtxSw", justify="right")
        table.add_column("Invol CtxSw", justify="right")
        table.add_column("Total Read", justify="right", style="blue")
        table.add_column("Total Write", justify="right", style="magenta")
        table.add_column("CPU Time (s)", justify="right")
        table.add_column("Tasks", justify="right")

        for worker_id, info in sorted(self.worker_infos.items()):
            agg = info.health_aggregates

            mem = agg.memory_usage
            mem_str = (
                f"{mem.min / 1e6:.1f} / {mem.avg / 1e6:.1f} / {mem.max / 1e6:.1f}"
                if mem.count > 0
                else "N/A"
            )

            cpu = agg.cpu_usage
            cpu_str = (
                f"{cpu.min:.1f} / {cpu.avg:.1f} / {cpu.max:.1f}"
                if cpu.count > 0
                else "N/A"
            )

            threads = agg.num_threads
            threads_str = (
                f"{int(threads.min)} / {threads.avg:.1f} / {int(threads.max)}"
                if threads.count > 0
                else "N/A"
            )

            vol_ctx = agg.voluntary_ctx_switches
            vol_ctx_str = (
                f"{int(vol_ctx.max - vol_ctx.min):,}"
                if vol_ctx.count > 0
                and vol_ctx.min is not None
                and vol_ctx.max is not None
                else "N/A"
            )

            invol_ctx = agg.involuntary_ctx_switches
            invol_ctx_str = (
                f"{int(invol_ctx.max - invol_ctx.min):,}"
                if invol_ctx.count > 0
                and invol_ctx.min is not None
                and invol_ctx.max is not None
                else "N/A"
            )

            io_read = agg.io_read_bytes
            io_read_str = (
                format_bytes(int(io_read.max - io_read.min))
                if io_read.count > 0
                and io_read.min is not None
                and io_read.max is not None
                else "N/A"
            )

            io_write = agg.io_write_bytes
            io_write_str = (
                format_bytes(int(io_write.max - io_write.min))
                if io_write.count > 0
                and io_write.min is not None
                and io_write.max is not None
                else "N/A"
            )

            cpu_user = agg.cpu_time_user
            cpu_sys = agg.cpu_time_system
            if (
                cpu_user.count > 0
                and cpu_sys.count > 0
                and cpu_user.min is not None
                and cpu_sys.min is not None
                and cpu_user.max is not None
                and cpu_sys.max is not None
            ):
                cpu_time_str = f"u:{cpu_user.max - cpu_user.min:.1f} s:{cpu_sys.max - cpu_sys.min:.1f}"
            else:
                cpu_time_str = "N/A"

            tasks = info.task_stats
            tasks_str = f"{tasks.completed}/{tasks.total}"
            if tasks.failed > 0:
                tasks_str += f" ({tasks.failed} failed)"

            table.add_row(
                worker_id.split("-")[-1],
                mem_str,
                cpu_str,
                threads_str,
                vol_ctx_str,
                invol_ctx_str,
                io_read_str,
                io_write_str,
                cpu_time_str,
                tasks_str,
            )

        # Totals row
        total_tasks = sum(i.task_stats.total for i in self.worker_infos.values())
        total_completed = sum(
            i.task_stats.completed for i in self.worker_infos.values()
        )
        total_failed = sum(i.task_stats.failed for i in self.worker_infos.values())

        all_mem_min = min(
            (i.health_aggregates.memory_usage.min or float("inf"))
            for i in self.worker_infos.values()
        )
        all_mem_max = max(
            (i.health_aggregates.memory_usage.max or 0)
            for i in self.worker_infos.values()
        )
        all_cpu_max = max(
            (i.health_aggregates.cpu_usage.max or 0) for i in self.worker_infos.values()
        )
        all_vol_ctx_delta = sum(
            (i.health_aggregates.voluntary_ctx_switches.max or 0)
            - (i.health_aggregates.voluntary_ctx_switches.min or 0)
            for i in self.worker_infos.values()
        )
        all_invol_ctx_delta = sum(
            (i.health_aggregates.involuntary_ctx_switches.max or 0)
            - (i.health_aggregates.involuntary_ctx_switches.min or 0)
            for i in self.worker_infos.values()
        )
        all_io_read_delta = sum(
            (i.health_aggregates.io_read_bytes.max or 0)
            - (i.health_aggregates.io_read_bytes.min or 0)
            for i in self.worker_infos.values()
        )
        all_io_write_delta = sum(
            (i.health_aggregates.io_write_bytes.max or 0)
            - (i.health_aggregates.io_write_bytes.min or 0)
            for i in self.worker_infos.values()
        )
        all_cpu_user_delta = sum(
            (i.health_aggregates.cpu_time_user.max or 0)
            - (i.health_aggregates.cpu_time_user.min or 0)
            for i in self.worker_infos.values()
        )
        all_cpu_sys_delta = sum(
            (i.health_aggregates.cpu_time_system.max or 0)
            - (i.health_aggregates.cpu_time_system.min or 0)
            for i in self.worker_infos.values()
        )

        total_tasks_str = f"{total_completed}/{total_tasks}"
        if total_failed > 0:
            total_tasks_str += f" ({total_failed} failed)"

        table.add_section()
        table.add_row(
            f"[bold]TOTAL ({len(self.worker_infos)} workers)[/bold]",
            f"[bold]{all_mem_min / 1e6:.1f} - {all_mem_max / 1e6:.1f}[/bold]",
            f"[bold]max: {all_cpu_max:.1f}[/bold]",
            "",
            f"[bold]{int(all_vol_ctx_delta):,}[/bold]",
            f"[bold]{int(all_invol_ctx_delta):,}[/bold]",
            f"[bold]{format_bytes(int(all_io_read_delta))}[/bold]",
            f"[bold]{format_bytes(int(all_io_write_delta))}[/bold]",
            f"[bold]u:{all_cpu_user_delta:.1f} s:{all_cpu_sys_delta:.1f}[/bold]",
            f"[bold]{total_tasks_str}[/bold]",
        )

        console.print("\n")
        console.print(table)
        console.print("[dim]Values shown as: min / avg / max[/dim]")
        console.file.flush()


def main() -> None:
    """Main entry point for the worker manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(ServiceType.WORKER_MANAGER)


if __name__ == "__main__":
    main()
