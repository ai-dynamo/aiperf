# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-reported process memory tracker.

Each service process reads its own memory and publishes a MemoryReportMessage.
The SystemController collects these reports into a ``MemoryTracker`` instance
and calls ``print_summary()`` at shutdown.

PSS (Proportional Set Size) divides shared pages proportionally among
processes, giving accurate memory footprints when mmap is used (e.g., for
datasets).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psutil

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class MemoryPhase(CaseInsensitiveStrEnum):
    """Lifecycle phase at which memory was captured."""

    STARTUP = "startup"
    POST_CONFIG = "post_config"
    SHUTDOWN = "shutdown"


@dataclass(slots=True)
class MemoryReading:
    """A single memory measurement."""

    pss: int | None = None
    """Proportional Set Size in bytes (shared pages divided among sharers)."""

    rss: int | None = None
    """Resident Set Size in bytes (total physical memory used)."""

    uss: int | None = None
    """Unique Set Size in bytes (memory unique to this process)."""

    shared: int | None = None
    """Shared memory in bytes (pages shared with other processes)."""


@dataclass(slots=True)
class MemorySnapshot:
    """Memory readings for a tracked process across lifecycle phases."""

    pid: int
    """Operating system process ID."""

    label: str
    """Human-readable identifier for the process (e.g. 'worker-0')."""

    group: str
    """Logical group the process belongs to (e.g. 'worker')."""

    _readings: dict[MemoryPhase, MemoryReading] = field(default_factory=dict)
    """Memory readings keyed by lifecycle phase."""

    def set_reading(self, phase: MemoryPhase, reading: MemoryReading) -> None:
        """Set a MemoryReading for the given phase."""
        self._readings[phase] = reading

    def get_reading(self, phase: MemoryPhase) -> MemoryReading | None:
        """Get the MemoryReading for the given phase, or None."""
        return self._readings.get(phase)

    def capture(self, phase: MemoryPhase) -> MemoryReading | None:
        """Read own-process memory and store it for the given phase.

        Returns:
            The captured MemoryReading, or None if unavailable.
        """
        reading = read_memory_self()
        if reading is not None:
            self._readings[phase] = reading
        return reading

    @property
    def startup(self) -> MemoryReading | None:
        return self._readings.get(MemoryPhase.STARTUP)

    @property
    def post_config(self) -> MemoryReading | None:
        return self._readings.get(MemoryPhase.POST_CONFIG)

    @property
    def shutdown(self) -> MemoryReading | None:
        return self._readings.get(MemoryPhase.SHUTDOWN)


@dataclass
class MemoryTracker:
    """Collects self-reported memory snapshots from service processes.

    Example usage::

        tracker = MemoryTracker()
        tracker.record("worker_0", "worker", pid=1234,
                        phase=MemoryPhase.STARTUP, reading=MemoryReading(pss=100))
        tracker.record("worker_0", "worker", pid=1234,
                        phase=MemoryPhase.SHUTDOWN, reading=MemoryReading(pss=200))
        tracker.print_summary()
    """

    _snapshots: dict[str, MemorySnapshot] = field(default_factory=dict)
    """Collected memory snapshots keyed by process label."""

    def record(
        self,
        label: str,
        group: str,
        pid: int,
        phase: MemoryPhase,
        reading: MemoryReading,
    ) -> None:
        """Record a memory reading for a service process.

        Creates a new MemorySnapshot if one doesn't exist for the given label.
        """
        if label not in self._snapshots:
            self._snapshots[label] = MemorySnapshot(pid=pid, label=label, group=group)
        self._snapshots[label].set_reading(phase, reading)

    def capture(
        self,
        label: str,
        group: str,
        pid: int,
        phase: MemoryPhase,
    ) -> MemoryReading | None:
        """Read own-process memory and record it for the given label/phase.

        Creates a new MemorySnapshot if one doesn't exist for the given label.

        Returns:
            The captured MemoryReading, or None if unavailable.
        """
        if label not in self._snapshots:
            self._snapshots[label] = MemorySnapshot(pid=pid, label=label, group=group)
        return self._snapshots[label].capture(phase)

    @property
    def snapshots(self) -> dict[str, MemorySnapshot]:
        """Return all recorded memory snapshots."""
        return self._snapshots

    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()

    def print_summary(self, title: str = "Process Memory") -> None:
        """Print a Rich table summarizing all recorded memory snapshots."""
        from rich.box import SIMPLE_HEAVY
        from rich.console import Console
        from rich.table import Table

        from aiperf.ui.utils import format_memory

        if not self._snapshots:
            return

        console = Console()
        table = Table(title=title, box=SIMPLE_HEAVY, padding=(0, 1))
        table.add_column("Process", justify="left", style="cyan")
        table.add_column("PSS Start", justify="right", style="green")
        table.add_column("PSS Post-Config", justify="right", style="green")
        table.add_column("PSS End", justify="right", style="green")
        table.add_column("Delta", justify="right", style="yellow")

        total_start = 0
        total_end = 0
        count = 0

        for snap in sorted(self._snapshots.values(), key=lambda s: s.label):
            start_pss = snap.startup.pss if snap.startup else None
            post_config_pss = snap.post_config.pss if snap.post_config else None
            end_pss = snap.shutdown.pss if snap.shutdown else None

            start_str = format_memory(start_pss) if start_pss is not None else "N/A"
            post_config_str = (
                format_memory(post_config_pss) if post_config_pss is not None else "N/A"
            )
            end_str = format_memory(end_pss) if end_pss is not None else "N/A"
            if start_pss is not None and end_pss is not None:
                delta_str = format_memory(end_pss - start_pss, signed=True)
            else:
                delta_str = "N/A"

            table.add_row(snap.label, start_str, post_config_str, end_str, delta_str)
            if start_pss is not None:
                total_start += start_pss
            if end_pss is not None:
                total_end += end_pss
            count += 1

        if count > 0:
            table.add_section()
            table.add_row(
                f"[bold]TOTAL ({count} processes)[/bold]",
                f"[bold]{format_memory(total_start)}[/bold]",
                "",
                f"[bold]{format_memory(total_end)}[/bold]",
                f"[bold]{format_memory(total_end - total_start, signed=True)}[/bold]",
            )
            console.print("\n")
            console.print(table)
            console.print(
                f"  Total: {format_memory(total_end)}  |  "
                f"Average: {format_memory(total_end // count)} per process"
            )
            console.file.flush()


# -- Self-reporting helpers --


def read_memory_self() -> MemoryReading | None:
    """Read full memory info for the current process.

    Returns:
        MemoryReading with pss/rss/uss/shared, or None if unavailable.
    """
    try:
        info = psutil.Process().memory_full_info()
        return MemoryReading(
            pss=info.pss,
            rss=info.rss,
            uss=info.uss,
            shared=info.shared,
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        return None


def read_pss_self() -> int | None:
    """Read PSS for the current process.

    Returns:
        PSS in bytes, or None if unavailable.
    """
    try:
        return psutil.Process().memory_full_info().pss
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        return None
