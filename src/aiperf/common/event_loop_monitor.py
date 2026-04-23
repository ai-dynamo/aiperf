# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event loop health monitoring utility class.

Provides two complementary monitoring strategies:

1. **Async monitor** (lightweight): An asyncio task that sleeps and measures elapsed time.
   Detects blocks after the fact but cannot attribute them.

2. **Watchdog thread** (diagnostic): A daemon thread that pings the event loop and captures
   the event loop thread's stack frame when it fails to respond within the threshold. This
   captures what is blocking *while* the block is happening.

The watchdog thread is enabled via AIPERF_SERVICE_EVENT_LOOP_HEALTH_STACKTRACE=true.
"""

import asyncio
import sys
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import orjson

from aiperf.common.constants import (
    MILLIS_PER_SECOND,
    NANOS_PER_MILLIS,
    NANOS_PER_SECOND,
)
from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin


@dataclass(slots=True)
class _StackTraceEntry:
    """A deduplicated stack trace with occurrence metadata."""

    stack: str
    """The formatted stack trace string."""

    count: int = 0
    """Number of times this exact stack trace was captured."""

    first_seen: float = 0.0
    """Monotonic timestamp when this stack trace was first observed."""

    last_seen: float = 0.0
    """Monotonic timestamp when this stack trace was last observed."""

    activities: set[str] = field(default_factory=set)
    """Set of activity labels active when this stack trace was captured."""


class StackTraceCollector:
    """Collects and deduplicates watchdog stack traces for end-of-run summary.

    Stack traces are keyed by their content so repeated blocks at the same
    call site are counted rather than printed individually. The summary is
    printed once at shutdown, sorted by frequency.

    Thread-safe: the watchdog thread calls ``add`` while the event loop thread
    calls ``print_summary`` during shutdown (after the watchdog has stopped).
    A lock guards the shared dict for safety.
    """

    def __init__(self) -> None:
        self._traces: dict[str, _StackTraceEntry] = {}
        self._lock = threading.Lock()

    def add(self, stack: str, activity: str) -> None:
        """Record a stack trace capture."""
        now = time.monotonic()
        with self._lock:
            entry = self._traces.get(stack)
            if entry is None:
                entry = _StackTraceEntry(stack=stack, first_seen=now)
                self._traces[stack] = entry
            entry.count += 1
            entry.last_seen = now
            if activity:
                entry.activities.add(activity)

    @property
    def total_captures(self) -> int:
        """Total number of stack trace captures across all entries."""
        with self._lock:
            return sum(e.count for e in self._traces.values())

    def print_summary(self, service_id: str) -> None:
        """Print deduplicated stack trace summary to stderr."""
        with self._lock:
            if not self._traces:
                return
            entries = sorted(self._traces.values(), key=lambda e: e.count, reverse=True)
            total = sum(e.count for e in entries)

        lines = [
            f"\n--- Event loop block trace summary: {service_id} ---",
            f"  Unique call sites: {len(entries)}",
            f"  Total captures:    {total}",
            "",
        ]
        for i, entry in enumerate(entries, 1):
            pct = entry.count / total * 100
            lines.append(f"  [{i}] {entry.count}x ({pct:.1f}%)")
            if entry.activities:
                activities = ", ".join(sorted(entry.activities))
                lines.append(f"      Activities: {activities}")
            # Indent the stack trace for readability
            for stack_line in entry.stack.rstrip("\n").split("\n"):
                lines.append(f"      {stack_line}")
            lines.append("")

        lines.append("--- End block trace summary ---\n")
        print("\n".join(lines), file=sys.stderr, flush=True)

    def save_to_file(self, path: Path, service_id: str) -> None:
        """Write collected traces as JSON to the given file path."""
        with self._lock:
            if not self._traces:
                return
            entries = sorted(self._traces.values(), key=lambda e: e.count, reverse=True)
            total = sum(e.count for e in entries)

        data = {
            "service_id": service_id,
            "unique_call_sites": len(entries),
            "total_captures": total,
            "traces": [
                {
                    "rank": i,
                    "count": entry.count,
                    "percentage": round(entry.count / total * 100, 1),
                    "activities": sorted(entry.activities),
                    "stack_trace": entry.stack.rstrip("\n"),
                }
                for i, entry in enumerate(entries, 1)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


class EventLoopMonitor(AIPerfLoggerMixin):
    """Utility class that monitors event loop health and logs warnings when blocked.

    This utility class adds a background task that periodically checks if the event loop
    is responsive by sleeping for a known interval and measuring actual elapsed time.
    If the delta exceeds the configured threshold, it indicates blocking operations.

    When stacktrace capture is enabled, a watchdog thread runs alongside the async monitor.
    The thread pings the event loop via call_soon_threadsafe and captures the event loop
    thread's stack when the ping is not acknowledged within the threshold. This provides
    attribution for the blocking call.

    Use the `activity` context manager to label the current operation so the watchdog
    can include it in diagnostics::

        with monitor.activity("handling WORKER_HEALTH from worker-0"):
            await process_message(msg)

    Configurable via Environment.SERVICE:
    - AIPERF_SERVICE_EVENT_LOOP_HEALTH_ENABLED: Enable/disable monitoring (default: True)
    - AIPERF_SERVICE_EVENT_LOOP_HEALTH_INTERVAL: Sleep interval in seconds (default: 0.25)
    - AIPERF_SERVICE_EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS: Warning threshold in ms (default: 10)
    - AIPERF_SERVICE_EVENT_LOOP_HEALTH_STACKTRACE: Enable watchdog thread stack capture (default: False)
    """

    def __init__(
        self, service_id: str, artifact_dir: Path | None = None, **kwargs
    ) -> None:
        super().__init__(service_id=service_id, **kwargs)
        self._service_id = service_id
        self._artifact_dir = artifact_dir
        self._event_loop_health_task = None
        self._stop_requested = False
        self._callback: Callable[[float], Awaitable] | None = None

        # Watchdog thread state
        self._watchdog_thread: threading.Thread | None = None
        self._event_loop_thread_id: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Current activity label — written by the event loop thread, read by the
        # watchdog thread. A plain str assignment is atomic in CPython (GIL), so
        # no lock is needed.
        self._current_activity: str = ""

        # Collect all delta samples (ms) for end-of-run summary
        self._delta_samples: list[float] = []

        # Deduplicated stack trace collector for watchdog captures
        self._stack_collector = StackTraceCollector()

    @contextmanager
    def activity(self, label: str) -> Iterator[None]:
        """Label the current event-loop operation for watchdog diagnostics.

        The watchdog thread reads this label when it detects a block, so it can
        report *what* the event loop was doing (e.g. which message type).
        """
        self._current_activity = label
        try:
            yield
        finally:
            self._current_activity = ""

    def set_callback(self, callback: Callable[[float], Awaitable]) -> None:
        """Set the callback to be called when the event loop is blocked."""
        self._callback = callback

    def start(self) -> None:
        """Start the event loop health task."""
        self._stop_requested = False
        if self._event_loop_health_task is None:
            self._event_loop_health_task = asyncio.create_task(
                self._monitor_event_loop()
            )

        # Start watchdog thread for stack capture if enabled
        if (
            Environment.SERVICE.EVENT_LOOP_HEALTH_STACKTRACE
            and self._watchdog_thread is None
        ):
            self._loop = asyncio.get_running_loop()
            self._event_loop_thread_id = threading.get_ident()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_thread_run,
                name=f"el-watchdog-{self._service_id}",
                daemon=True,
            )
            self._watchdog_thread.start()

    def stop(self) -> None:
        """Stop the event loop health task and print summary."""
        if self._stop_requested:
            return
        self._stop_requested = True
        if self._event_loop_health_task is not None:
            self._event_loop_health_task.cancel()
        self._event_loop_health_task = None
        # Watchdog thread checks _stop_requested and will exit on its own

        if self._delta_samples:
            self._print_summary()
        self._stack_collector.print_summary(self._service_id)
        if self._artifact_dir is not None:
            safe_id = self._service_id.replace("/", "_").replace(":", "_")
            path = self._artifact_dir / "event_loop_traces" / f"{safe_id}.json"
            self._stack_collector.save_to_file(path, self._service_id)

    async def _monitor_event_loop(self) -> None:
        """Monitor event loop health and log warnings when latency exceeds threshold.

        This task detects blocked event loops by sleeping for a known interval
        and measuring actual elapsed time. If the delta exceeds the configured
        threshold, it indicates the event loop was blocked by other tasks.
        """
        if not Environment.SERVICE.EVENT_LOOP_HEALTH_ENABLED:
            return

        interval_sec = Environment.SERVICE.EVENT_LOOP_HEALTH_INTERVAL
        threshold_ns = (
            Environment.SERVICE.EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS * NANOS_PER_MILLIS
        )
        expected_ns = round(interval_sec * NANOS_PER_SECOND)

        while not self._stop_requested:
            start_perf_ns = time.perf_counter_ns()
            await asyncio.sleep(interval_sec)
            elapsed_ns = time.perf_counter_ns() - start_perf_ns
            delta_ns = elapsed_ns - expected_ns
            delta_ms = delta_ns / NANOS_PER_MILLIS
            self._delta_samples.append(delta_ms)
            if self.is_trace_enabled:
                self.trace(
                    f"Event loop health check: expected {interval_sec * MILLIS_PER_SECOND:.1f}ms, actual {elapsed_ns / NANOS_PER_MILLIS:.2f}ms, delta {delta_ms:.2f}ms"
                )
            if delta_ns > threshold_ns:
                self.warning(
                    f"Event loop for {self._service_id} is taking too long to run. Overhead: {delta_ms:,.2f}ms"
                )
                if self._callback is not None:
                    await self._callback(delta_ms)

    def _print_summary(self) -> None:
        """Print event loop latency summary to stderr."""
        samples = self._delta_samples
        if not samples:
            return

        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        threshold_ms = Environment.SERVICE.EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS
        over_threshold = sum(1 for s in samples if s > threshold_ms)

        def percentile(p: float) -> float:
            idx = int(p / 100 * (n - 1))
            return sorted_samples[idx]

        print(
            f"\n--- Event loop latency summary: {self._service_id} ---\n"
            f"  Samples:    {n}\n"
            f"  Min:        {sorted_samples[0]:.2f}ms\n"
            f"  Median:     {percentile(50):.2f}ms\n"
            f"  P90:        {percentile(90):.2f}ms\n"
            f"  P99:        {percentile(99):.2f}ms\n"
            f"  Max:        {sorted_samples[-1]:.2f}ms\n"
            f"  >{threshold_ms:.0f}ms:     {over_threshold}/{n} ({over_threshold / n * 100:.1f}%)\n"
            f"--- End latency summary ---\n",
            file=sys.stderr,
            flush=True,
        )

    def _watchdog_thread_run(self) -> None:
        """Watchdog thread that captures event loop thread stack on blocks.

        Runs in a daemon thread. Periodically schedules a no-op callback on the event loop
        via call_soon_threadsafe. If the callback isn't executed within the threshold,
        the event loop is blocked - capture the stack of the event loop thread.
        """
        loop = self._loop
        thread_id = self._event_loop_thread_id
        if loop is None or thread_id is None:
            return

        interval = Environment.SERVICE.EVENT_LOOP_HEALTH_INTERVAL
        threshold_sec = (
            Environment.SERVICE.EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS / MILLIS_PER_SECOND
        )

        ack = threading.Event()

        while not self._stop_requested:
            ack.clear()
            try:
                loop.call_soon_threadsafe(ack.set)
            except RuntimeError:
                # Event loop closed
                break

            # Wait for the event loop to execute our callback
            if ack.wait(timeout=threshold_sec):
                # Event loop responded in time
                time.sleep(interval)
                continue

            # Event loop did not respond - it's blocked right now.
            # Capture the stack of the event loop thread.
            frame = sys._current_frames().get(thread_id)
            if frame is not None:
                stack = "".join(traceback.format_stack(frame))
                activity = self._current_activity
                self._stack_collector.add(stack, activity)

            # Wait for the event loop to recover before checking again.
            # This prevents spamming logs if the block lasts a long time.
            ack.wait(timeout=interval)
