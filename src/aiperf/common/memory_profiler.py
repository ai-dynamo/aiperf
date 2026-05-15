# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory profiler for tracking memory usage and allocations.

Provides tracemalloc-based memory profiling for debugging memory growth issues.
Enable with AIPERF_DEV_MEMORY_PROFILE_ENABLED=true.

Example:
    from aiperf.common.memory_profiler import MemoryProfiler

    profiler = MemoryProfiler(service_id="worker-0")
    profiler.start()

    # ... do work ...

    profiler.snapshot("after_request")
    profiler.log_stats()

    # Or use as context manager for per-operation tracking:
    with profiler.track("process_response"):
        # ... process response ...
        pass
"""

import linecache
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.environment import Environment

_logger = AIPerfLogger(__name__)


@dataclass
class TracemallocSnapshot:
    """A snapshot of memory state at a point in time."""

    label: str
    """Identifier for this snapshot (e.g. 'baseline', 'after_work')."""

    current_bytes: int
    """Current traced memory usage in bytes."""

    peak_bytes: int
    """Peak traced memory usage in bytes since profiling started."""

    traced_bytes: int
    """Total bytes across the top allocator statistics."""

    top_stats: list[tracemalloc.Statistic] = field(default_factory=list)
    """Top memory allocation statistics sorted by size."""

    @property
    def current_mib(self) -> float:
        return self.current_bytes / BYTES_PER_MIB

    @property
    def peak_mib(self) -> float:
        return self.peak_bytes / BYTES_PER_MIB

    @property
    def traced_mib(self) -> float:
        return self.traced_bytes / BYTES_PER_MIB


class MemoryProfiler:
    """Memory profiler using tracemalloc for detailed allocation tracking.

    Tracks memory allocations with source file/line information. Useful for
    identifying memory leaks and understanding memory growth patterns.

    Usage:
        profiler = MemoryProfiler(service_id="worker-0")
        profiler.start()

        # Take snapshots at interesting points
        profiler.snapshot("baseline")
        # ... do work ...
        profiler.snapshot("after_work")

        # Compare snapshots
        profiler.compare("baseline", "after_work")

        # Get top allocators
        profiler.log_stats()
    """

    def __init__(
        self,
        service_id: str,
        top_n: int | None = None,
    ):
        """Initialize memory profiler.

        Args:
            service_id: Service identifier for logging.
            top_n: Number of top allocators to track. Uses env default if None.
        """
        self.service_id = service_id
        self.top_n = top_n or Environment.DEV.MEMORY_PROFILE_TOP_N
        self._snapshots: dict[str, TracemallocSnapshot] = {}
        self._started = False
        self._request_count = 0
        self._baseline_snapshot: tracemalloc.Snapshot | None = None

    @property
    def enabled(self) -> bool:
        """Check if memory profiling is enabled via environment."""
        return Environment.DEV.MEMORY_PROFILE_ENABLED

    def start(self) -> None:
        """Start tracemalloc if not already started."""
        if not self.enabled:
            return

        if not tracemalloc.is_tracing():
            # Store 25 frames for detailed backtraces
            tracemalloc.start(25)
            _logger.info(f"[{self.service_id}] Memory profiling started (tracemalloc)")

        self._started = True
        # Take baseline snapshot
        self._baseline_snapshot = tracemalloc.take_snapshot()
        self.snapshot("baseline")

    def stop(self) -> None:
        """Stop tracemalloc and log final stats."""
        if not self._started:
            return

        self.log_stats()
        self.log_growth_since_baseline()

        tracemalloc.stop()
        self._started = False
        _logger.info(f"[{self.service_id}] Memory profiling stopped")

    def snapshot(self, label: str) -> TracemallocSnapshot | None:
        """Take a memory snapshot with the given label.

        Args:
            label: Identifier for this snapshot.

        Returns:
            TracemallocSnapshot or None if profiling not enabled.
        """
        if not self._started:
            return None

        current, peak = tracemalloc.get_traced_memory()
        snap = tracemalloc.take_snapshot()

        # Filter out tracemalloc internals
        snap = snap.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
                tracemalloc.Filter(False, tracemalloc.__file__),
            )
        )

        top_stats = snap.statistics("lineno")[: self.top_n]

        snapshot = TracemallocSnapshot(
            label=label,
            current_bytes=current,
            peak_bytes=peak,
            traced_bytes=sum(stat.size for stat in top_stats),
            top_stats=top_stats,
        )

        self._snapshots[label] = snapshot
        return snapshot

    def log_stats(self, label: str | None = None) -> None:
        """Log memory statistics.

        Args:
            label: Specific snapshot to log, or latest if None.
        """
        if not self._started:
            return

        if label and label in self._snapshots:
            snap = self._snapshots[label]
        else:
            # Take a fresh snapshot
            snap = self.snapshot("current")
            if snap is None:
                return

        _logger.info(
            f"[{self.service_id}] Memory: "
            f"current={snap.current_mib:.2f}MiB, "
            f"peak={snap.peak_mib:.2f}MiB, "
            f"requests={self._request_count}"
        )

        _logger.info(f"[{self.service_id}] Top {self.top_n} allocators:")
        for i, stat in enumerate(snap.top_stats, 1):
            frame = stat.traceback[0]
            # Get source line for context
            line = linecache.getline(frame.filename, frame.lineno).strip()
            size_mib = stat.size / BYTES_PER_MIB
            _logger.info(
                f"  {i}. {frame.filename}:{frame.lineno} "
                f"({size_mib:.3f}MiB, {stat.count} blocks)"
            )
            if line:
                _logger.info(f"     > {line[:80]}")

    def log_growth_since_baseline(self) -> None:
        """Log memory growth since baseline snapshot."""
        if not self._started or self._baseline_snapshot is None:
            return

        current_snap = tracemalloc.take_snapshot()
        current_snap = current_snap.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
                tracemalloc.Filter(False, tracemalloc.__file__),
            )
        )

        diff_stats = current_snap.compare_to(self._baseline_snapshot, "lineno")
        top_diff = [s for s in diff_stats if s.size_diff > 0][: self.top_n]

        if not top_diff:
            _logger.info(f"[{self.service_id}] No significant memory growth detected")
            return

        total_growth = sum(s.size_diff for s in top_diff)
        _logger.info(
            f"[{self.service_id}] Memory growth since baseline: "
            f"{total_growth / BYTES_PER_MIB:.2f}MiB across {len(top_diff)} locations"
        )

        _logger.info(f"[{self.service_id}] Top {self.top_n} growth sources:")
        for i, stat in enumerate(top_diff, 1):
            frame = stat.traceback[0]
            line = linecache.getline(frame.filename, frame.lineno).strip()
            growth_mib = stat.size_diff / BYTES_PER_MIB
            _logger.info(
                f"  {i}. {frame.filename}:{frame.lineno} "
                f"(+{growth_mib:.3f}MiB, +{stat.count_diff} blocks)"
            )
            if line:
                _logger.info(f"     > {line[:80]}")

    def compare(self, label1: str, label2: str) -> None:
        """Compare two snapshots and log differences.

        Args:
            label1: First snapshot label (baseline).
            label2: Second snapshot label (current).
        """
        if not self._started:
            return

        snap1 = self._snapshots.get(label1)
        snap2 = self._snapshots.get(label2)

        if not snap1 or not snap2:
            _logger.warning(
                f"[{self.service_id}] Cannot compare: "
                f"missing snapshot {label1 if not snap1 else label2}"
            )
            return

        diff_mib = (snap2.current_bytes - snap1.current_bytes) / BYTES_PER_MIB
        _logger.info(
            f"[{self.service_id}] Memory diff ({label1} -> {label2}): "
            f"{diff_mib:+.2f}MiB"
        )

    @contextmanager
    def track(self, operation: str) -> Iterator[None]:
        """Context manager to track memory for a specific operation.

        Args:
            operation: Name of the operation being tracked.

        Example:
            with profiler.track("process_response"):
                result = process_response(data)
        """
        if not self._started:
            yield
            return

        before = tracemalloc.get_traced_memory()[0]
        try:
            yield
        finally:
            after = tracemalloc.get_traced_memory()[0]
            diff_mib = (after - before) / BYTES_PER_MIB
            if abs(diff_mib) > 0.1:  # Only log significant changes
                _logger.debug(f"[{self.service_id}] {operation}: {diff_mib:+.3f}MiB")

    def on_request_complete(self) -> None:
        """Called when a request completes. Logs stats periodically."""
        if not self._started:
            return

        self._request_count += 1

        # Log stats periodically based on request count
        if self._request_count % 100 == 0:
            self.log_stats()

    def get_current_usage_mib(self) -> float:
        """Get current memory usage in MiB.

        Returns:
            Current traced memory in MiB, or 0 if not profiling.
        """
        if not self._started:
            return 0.0
        current, _ = tracemalloc.get_traced_memory()
        return current / BYTES_PER_MIB

    def get_object_counts(self) -> dict[str, int]:
        """Get counts of common object types that might indicate leaks.

        Returns:
            Dict mapping type name to count.
        """
        import gc

        counts: dict[str, int] = {}
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            counts[type_name] = counts.get(type_name, 0) + 1

        # Return top types
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_counts[: self.top_n * 2])

    def log_object_counts(self, label: str = "") -> None:
        """Log counts of top object types.

        Args:
            label: Optional label for the log entry.
        """
        if not self._started:
            return

        counts = self.get_object_counts()
        prefix = f"[{label}] " if label else ""
        _logger.info(
            f"[{self.service_id}] {prefix}Top object types: "
            + ", ".join(f"{k}={v}" for k, v in list(counts.items())[:10])
        )
