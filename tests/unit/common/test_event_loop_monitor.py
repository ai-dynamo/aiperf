# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EventLoopMonitor utility class."""

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from aiperf.common.environment import Environment
from aiperf.common.event_loop_monitor import EventLoopMonitor, StackTraceCollector

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_SERVICE_ID = "test_service"
BASE_TIME_NS = 1_000_000_000
NANOS_PER_MS = 1_000_000
NANOS_PER_SEC = 1_000_000_000


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


class MockPerfCounter:
    """Helper to simulate time.perf_counter_ns with configurable delay.

    Usage:
        mock = MockPerfCounter(interval_sec=0.1, extra_delay_ms=50)
        with mock.patch():
            await monitor._monitor_event_loop()
    """

    def __init__(
        self,
        interval_sec: float = 0.1,
        extra_delay_ms: float = 0,
        stop_after_iterations: int = 1,
        monitor: EventLoopMonitor | None = None,
    ) -> None:
        self.interval_ns = int(interval_sec * NANOS_PER_SEC)
        self.extra_delay_ns = int(extra_delay_ms * NANOS_PER_MS)
        self.stop_after = stop_after_iterations
        self.monitor = monitor
        self._call_count = 0
        self._iteration = 0

    def __call__(self) -> int:
        self._call_count += 1
        if self._call_count % 2 == 1:  # Start of iteration
            self._iteration += 1
            if self._iteration > self.stop_after and self.monitor:
                self.monitor._stop_requested = True
            return BASE_TIME_NS + (self._iteration - 1) * self.interval_ns
        # End of iteration - add configured delay
        return BASE_TIME_NS + self._iteration * self.interval_ns + self.extra_delay_ns

    @contextmanager
    def patch(self):
        """Context manager to patch time.perf_counter_ns."""
        with patch(
            "aiperf.common.event_loop_monitor.time.perf_counter_ns",
            side_effect=self,
        ):
            yield


class MessageCapture:
    """Helper to capture log messages and optionally stop the monitor."""

    def __init__(
        self, monitor: EventLoopMonitor | None = None, stop_on_capture: bool = True
    ) -> None:
        self.messages: list[str] = []
        self.monitor = monitor
        self.stop_on_capture = stop_on_capture

    def __call__(self, msg: str) -> None:
        self.messages.append(msg)
        if self.stop_on_capture and self.monitor:
            self.monitor._stop_requested = True


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def monitor() -> EventLoopMonitor:
    """Create a basic EventLoopMonitor instance."""
    return EventLoopMonitor(service_id=DEFAULT_SERVICE_ID)


@pytest.fixture
def enabled_env(monkeypatch) -> None:
    """Configure environment for enabled monitoring with standard settings."""
    monkeypatch.setattr(Environment.SERVICE, "EVENT_LOOP_HEALTH_ENABLED", True)
    monkeypatch.setattr(Environment.SERVICE, "EVENT_LOOP_HEALTH_INTERVAL", 0.1)
    monkeypatch.setattr(
        Environment.SERVICE, "EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS", 25.0
    )


@pytest.fixture
def disabled_env(monkeypatch) -> None:
    """Configure environment with monitoring disabled."""
    monkeypatch.setattr(Environment.SERVICE, "EVENT_LOOP_HEALTH_ENABLED", False)


# -----------------------------------------------------------------------------
# Tests: Initialization
# -----------------------------------------------------------------------------


class TestEventLoopMonitorInit:
    """Tests for EventLoopMonitor initialization."""

    def test_init_sets_default_state(self, monitor):
        """Test that __init__ sets correct default state."""
        assert monitor._event_loop_health_task is None
        assert monitor._stop_requested is False
        assert monitor._callback is None
        assert monitor._service_id == DEFAULT_SERVICE_ID

    @pytest.mark.parametrize("attr", ["warning", "trace", "is_trace_enabled"])
    def test_inherits_logger_mixin_attributes(self, monitor, attr):
        """Test that EventLoopMonitor inherits logging capabilities."""
        assert hasattr(monitor, attr)


# -----------------------------------------------------------------------------
# Tests: Callback Management
# -----------------------------------------------------------------------------


class TestEventLoopMonitorCallback:
    """Tests for callback management."""

    def test_set_callback_stores_callback(self, monitor):
        """Test that set_callback stores the provided callback."""

        async def my_callback(delta_ms: float) -> None:
            pass

        monitor.set_callback(my_callback)
        assert monitor._callback is my_callback

    def test_set_callback_replaces_previous(self, monitor):
        """Test that set_callback can replace an existing callback."""

        async def callback1(delta_ms: float) -> None:
            pass

        async def callback2(delta_ms: float) -> None:
            pass

        monitor.set_callback(callback1)
        monitor.set_callback(callback2)
        assert monitor._callback is callback2


# -----------------------------------------------------------------------------
# Tests: Lifecycle (start/stop)
# -----------------------------------------------------------------------------


class TestEventLoopMonitorLifecycle:
    """Tests for start/stop lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, monitor):
        """Test that start() creates a monitoring task."""
        with patch.object(monitor, "_monitor_event_loop", new_callable=AsyncMock):
            monitor.start()
            try:
                assert monitor._event_loop_health_task is not None
                assert not monitor._stop_requested
            finally:
                monitor.stop()

    @pytest.mark.asyncio
    async def test_start_resets_stop_flag(self, monitor):
        """Test that start() resets the stop_requested flag."""
        monitor._stop_requested = True
        with patch.object(monitor, "_monitor_event_loop", new_callable=AsyncMock):
            monitor.start()
            try:
                assert not monitor._stop_requested
            finally:
                monitor.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, monitor):
        """Test that calling start() twice doesn't create duplicate tasks."""
        with patch.object(monitor, "_monitor_event_loop", new_callable=AsyncMock):
            monitor.start()
            first_task = monitor._event_loop_health_task
            monitor.start()
            try:
                assert monitor._event_loop_health_task is first_task
            finally:
                monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, monitor):
        """Test that stop() cancels the monitoring task and cleans up."""
        with patch.object(monitor, "_monitor_event_loop", new_callable=AsyncMock):
            monitor.start()
            monitor.stop()
            assert monitor._stop_requested is True
            assert monitor._event_loop_health_task is None

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, monitor):
        """Test that calling stop() multiple times is safe."""
        with patch.object(monitor, "_monitor_event_loop", new_callable=AsyncMock):
            monitor.start()
            monitor.stop()
            monitor.stop()  # Should not raise
            assert monitor._stop_requested is True

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, monitor):
        """Test that calling stop() without start() is safe."""
        monitor.stop()
        assert monitor._stop_requested is True


# -----------------------------------------------------------------------------
# Tests: Monitoring Behavior
# -----------------------------------------------------------------------------


class TestEventLoopMonitorBehavior:
    """Tests for the _monitor_event_loop method behavior."""

    @pytest.mark.asyncio
    async def test_exits_when_disabled(self, monitor, disabled_env):
        """Test that monitoring exits immediately when disabled."""
        # Should complete immediately without looping
        await monitor._monitor_event_loop()
        # If we get here without hanging, the test passes

    @pytest.mark.asyncio
    @pytest.mark.looptime
    @pytest.mark.parametrize(
        ("extra_delay_ms", "threshold_ms", "expect_warning"),
        [
            (50.0, 10.0, True),   # 50ms delay > 10ms threshold → warn
            (5.0, 10.0, False),   # 5ms delay < 10ms threshold → no warn
            (10.0, 10.0, False),  # 10ms delay == 10ms threshold → no warn (must exceed)
            (100.0, 50.0, True),  # 100ms delay > 50ms threshold → warn
        ],
    )  # fmt: skip
    async def test_warning_threshold_behavior(
        self,
        monitor,
        enabled_env,
        monkeypatch,
        time_traveler,
        extra_delay_ms,
        threshold_ms,
        expect_warning,
    ):
        """Test that warnings are logged only when delay exceeds threshold."""
        monkeypatch.setattr(
            Environment.SERVICE, "EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS", threshold_ms
        )

        capture = MessageCapture(monitor)
        mock_perf = MockPerfCounter(
            interval_sec=0.1,
            extra_delay_ms=extra_delay_ms,
            stop_after_iterations=1,
            monitor=monitor,
        )

        with mock_perf.patch(), patch.object(monitor, "warning", side_effect=capture):
            await monitor._monitor_event_loop()

        if expect_warning:
            assert len(capture.messages) == 1
            assert DEFAULT_SERVICE_ID in capture.messages[0]
            assert "taking too long" in capture.messages[0]
        else:
            assert len(capture.messages) == 0

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_callback_invoked_with_delta(
        self, monitor, enabled_env, monkeypatch, time_traveler
    ):
        """Test that callback is invoked with correct delta when threshold exceeded."""
        monkeypatch.setattr(
            Environment.SERVICE, "EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS", 5.0
        )

        callback_values: list[float] = []

        async def test_callback(delta_ms: float) -> None:
            callback_values.append(delta_ms)
            monitor._stop_requested = True

        monitor.set_callback(test_callback)
        mock_perf = MockPerfCounter(
            interval_sec=0.1,
            extra_delay_ms=50.0,  # 50ms extra delay
            stop_after_iterations=1,
            monitor=monitor,
        )

        with mock_perf.patch(), patch.object(monitor, "warning", lambda msg: None):
            await monitor._monitor_event_loop()

        assert len(callback_values) == 1
        assert callback_values[0] == pytest.approx(50.0, abs=1.0)

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_trace_logging_when_enabled(
        self, monitor, enabled_env, monkeypatch, time_traveler
    ):
        """Test that trace messages are logged when trace is enabled."""
        monkeypatch.setattr(
            Environment.SERVICE, "EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS", 1000.0
        )

        capture = MessageCapture(monitor)
        mock_perf = MockPerfCounter(
            interval_sec=0.1,
            extra_delay_ms=0,
            stop_after_iterations=1,
            monitor=monitor,
        )

        with (
            patch.object(
                EventLoopMonitor,
                "is_trace_enabled",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(monitor, "trace", side_effect=capture),
            mock_perf.patch(),
        ):
            await monitor._monitor_event_loop()

        assert len(capture.messages) == 1
        assert "Event loop health check" in capture.messages[0]
        assert "expected" in capture.messages[0]
        assert "actual" in capture.messages[0]


# -----------------------------------------------------------------------------
# Tests: Integration
# -----------------------------------------------------------------------------


class TestEventLoopMonitorIntegration:
    """Integration tests for full lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch):
        """Test full start/monitor/stop lifecycle with real async behavior."""
        monkeypatch.setattr(Environment.SERVICE, "EVENT_LOOP_HEALTH_ENABLED", True)
        monkeypatch.setattr(Environment.SERVICE, "EVENT_LOOP_HEALTH_INTERVAL", 0.01)
        monkeypatch.setattr(
            Environment.SERVICE, "EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS", 1000.0
        )

        monitor = EventLoopMonitor(service_id="integration_test")

        async def noop_callback(delta_ms: float) -> None:
            pass

        monitor.set_callback(noop_callback)
        monitor.start()

        # Let it run briefly
        await asyncio.sleep(0.05)

        monitor.stop()

        assert monitor._stop_requested is True
        assert monitor._event_loop_health_task is None


# -----------------------------------------------------------------------------
# Tests: StackTraceCollector
# -----------------------------------------------------------------------------


class TestStackTraceCollector:
    """Tests for the StackTraceCollector class."""

    def test_empty_collector_has_zero_captures(self):
        collector = StackTraceCollector()
        assert collector.total_captures == 0

    def test_add_single_trace(self):
        collector = StackTraceCollector()
        collector.add("File foo.py, line 1\n  do_stuff()\n", "handling msg")
        assert collector.total_captures == 1

    def test_duplicate_traces_are_counted(self):
        collector = StackTraceCollector()
        stack = "File foo.py, line 1\n  do_stuff()\n"
        collector.add(stack, "activity-a")
        collector.add(stack, "activity-b")
        collector.add(stack, "activity-a")
        assert collector.total_captures == 3

    def test_different_traces_tracked_separately(self):
        collector = StackTraceCollector()
        collector.add("stack-A\n", "")
        collector.add("stack-B\n", "")
        collector.add("stack-A\n", "")
        assert collector.total_captures == 3

    def test_activities_collected_as_set(self):
        collector = StackTraceCollector()
        stack = "stack-A\n"
        collector.add(stack, "act1")
        collector.add(stack, "act2")
        collector.add(stack, "act1")
        entry = collector._traces[stack]
        assert entry.activities == {"act1", "act2"}

    def test_empty_activity_not_added(self):
        collector = StackTraceCollector()
        collector.add("stack\n", "")
        entry = collector._traces["stack\n"]
        assert entry.activities == set()

    def test_print_summary_empty_is_silent(self, capsys):
        collector = StackTraceCollector()
        collector.print_summary("svc")
        assert capsys.readouterr().err == ""

    def test_print_summary_contains_expected_sections(self, capsys):
        collector = StackTraceCollector()
        collector.add("File foo.py, line 10\n  blocking_call()\n", "msg-A")
        collector.add("File foo.py, line 10\n  blocking_call()\n", "msg-B")
        collector.add("File bar.py, line 20\n  other_call()\n", "")
        collector.print_summary("worker-0")
        output = capsys.readouterr().err
        assert "Event loop block trace summary: worker-0" in output
        assert "Unique call sites: 2" in output
        assert "Total captures:    3" in output
        assert "2x (66.7%)" in output
        assert "1x (33.3%)" in output
        assert "blocking_call()" in output
        assert "other_call()" in output
        assert "Activities: msg-A, msg-B" in output
        assert "End block trace summary" in output

    def test_print_summary_sorted_by_frequency(self, capsys):
        collector = StackTraceCollector()
        collector.add("rare-stack\n", "")
        for _ in range(5):
            collector.add("frequent-stack\n", "")
        collector.print_summary("svc")
        output = capsys.readouterr().err
        # frequent should appear before rare
        freq_pos = output.index("frequent-stack")
        rare_pos = output.index("rare-stack")
        assert freq_pos < rare_pos

    def test_save_to_file_empty_is_noop(self, tmp_path):
        collector = StackTraceCollector()
        path = tmp_path / "traces" / "svc.json"
        collector.save_to_file(path, "svc")
        assert not path.exists()

    def test_save_to_file_creates_parent_dirs(self, tmp_path):
        collector = StackTraceCollector()
        collector.add("stack-A\n", "act1")
        path = tmp_path / "deep" / "nested" / "traces.json"
        collector.save_to_file(path, "svc")
        assert path.exists()

    def test_save_to_file_contains_correct_data(self, tmp_path):
        import orjson

        collector = StackTraceCollector()
        collector.add("File foo.py, line 10\n  block()\n", "msg-A")
        collector.add("File foo.py, line 10\n  block()\n", "msg-B")
        collector.add("File bar.py, line 20\n  other()\n", "")
        path = tmp_path / "traces.json"
        collector.save_to_file(path, "worker-0")

        data = orjson.loads(path.read_bytes())
        assert data["service_id"] == "worker-0"
        assert data["unique_call_sites"] == 2
        assert data["total_captures"] == 3
        assert len(data["traces"]) == 2
        # Sorted by frequency: 2x first, 1x second
        assert data["traces"][0]["count"] == 2
        assert data["traces"][0]["rank"] == 1
        assert data["traces"][0]["percentage"] == 66.7
        assert data["traces"][0]["activities"] == ["msg-A", "msg-B"]
        assert "block()" in data["traces"][0]["stack_trace"]
        assert data["traces"][1]["count"] == 1
