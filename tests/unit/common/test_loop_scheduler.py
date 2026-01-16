# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LoopScheduler.

LoopScheduler manages coroutine scheduling with:
- Automatic cleanup of pending/running coroutines
- Time domain bridging (perf_counter_ns → loop.time())
- Exception handling for failed tasks

Uses time_traveler_no_patch_sleep fixture which:
- Mocks time.perf_counter_ns() for LoopScheduler's time domain bridging
- Lets looptime handle asyncio.sleep() for virtual time advancement
"""

import asyncio
import time

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.utils import yield_to_event_loop

# =============================================================================
# Helpers
# =============================================================================


async def yield_to_scheduler(n: int = 2) -> None:
    """Yield to event loop n times to let scheduled callbacks/tasks run.

    LoopScheduler uses call_soon which requires TWO yields:
    - 1st: call_soon callback fires → creates task
    - 2nd: task runs to completion (or first await)

    Use n=3 for tasks that fail (extra yield for done_callback).
    Use n=4 for nested scheduling (outer schedules inner).
    """
    for _ in range(n):
        await yield_to_event_loop()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def scheduler(time_traveler_no_patch_sleep):
    """Create a fresh LoopScheduler for each test.

    Depends on time_traveler_no_patch_sleep to:
    - Mock time.perf_counter_ns() for time domain bridging
    - Enable looptime for virtual time advancement via asyncio.sleep()
    """
    return LoopScheduler()


# =============================================================================
# Basic Scheduling Tests
# =============================================================================


class TestScheduleLater:
    """Tests for schedule_later (relative delay scheduling)."""

    async def test_executes_after_delay(self, scheduler: LoopScheduler):
        """Coroutine executes after specified delay."""
        loop = asyncio.get_running_loop()
        executed = []

        async def task():
            executed.append(loop.time())

        start = loop.time()
        scheduler.schedule_later(1.0, task())

        assert executed == []
        assert scheduler.pending_count == 1

        await asyncio.sleep(1.5)

        assert len(executed) == 1
        assert executed[0] - start >= 1.0

    @pytest.mark.parametrize(
        "delays",
        [
            pytest.param([0], id="zero"),
            pytest.param([-1.0], id="negative"),
            pytest.param([0, -1.0, -100], id="multiple"),
        ],
    )
    async def test_zero_or_negative_delay_uses_execute_async(
        self, scheduler: LoopScheduler, delays: list[float]
    ):
        """Zero or negative delay delegates to execute_async."""
        executed = []

        async def task():
            executed.append(True)

        for delay in delays:
            scheduler.schedule_later(delay, task())

        # execute_async creates tasks immediately (no pending handles)
        assert scheduler.pending_count == 0
        assert scheduler.running_count == len(delays)

        await yield_to_scheduler()

        assert len(executed) == len(delays)
        assert scheduler.running_count == 0

    async def test_multiple_tasks_execute_in_order(self, scheduler: LoopScheduler):
        """Multiple tasks execute in delay order."""
        order = []

        async def task(name: str):
            order.append(name)

        scheduler.schedule_later(0.3, task("third"))
        scheduler.schedule_later(0.1, task("first"))
        scheduler.schedule_later(0.2, task("second"))

        assert scheduler.pending_count == 3

        await asyncio.sleep(0.5)

        assert order == ["first", "second", "third"]


class TestExecuteAsync:
    """Tests for execute_async (immediate task execution)."""

    async def test_executes_on_next_iteration(self, scheduler: LoopScheduler):
        """Coroutine executes on next event loop iteration."""
        executed = []

        async def task():
            executed.append(True)

        scheduler.execute_async(task())
        assert executed == []

        await yield_to_scheduler()

        assert executed == [True]

    async def test_tracked_in_pending(self, scheduler: LoopScheduler):
        """execute_async tasks are tracked in running_count (no pending handle)."""

        async def noop():
            pass

        scheduler.execute_async(noop())
        assert scheduler.pending_count == 0
        assert scheduler.running_count == 1
        await yield_to_scheduler()
        assert scheduler.running_count == 0


class TestScheduleAtPerfNs:
    """Tests for schedule_at_perf_ns (absolute time scheduling)."""

    async def test_executes_at_target_time(self, scheduler: LoopScheduler):
        """Coroutine scheduled at future perf_counter_ns timestamp executes."""
        executed = []

        async def task():
            executed.append(True)

        target_ns = time.perf_counter_ns() + int(1.0 * NANOS_PER_SECOND)
        scheduler.schedule_at_perf_ns(target_ns, task())

        assert scheduler.pending_count == 1
        assert executed == []

        await asyncio.sleep(1.5)

        assert len(executed) == 1

    async def test_past_time_uses_execute_async(self, scheduler: LoopScheduler):
        """Past timestamps delegate to execute_async."""
        executed = []

        async def task():
            executed.append(True)

        past_ns = time.perf_counter_ns() - int(1.0 * NANOS_PER_SECOND)
        scheduler.schedule_at_perf_ns(past_ns, task())

        # Past time → execute_async → no pending handle, task is running
        assert scheduler.pending_count == 0
        assert scheduler.running_count == 1

        await yield_to_scheduler()

        assert executed == [True]
        assert scheduler.running_count == 0

    async def test_multiple_execute_in_order(self, scheduler: LoopScheduler):
        """Multiple tasks scheduled at different perf_ns times execute in order."""
        order = []

        async def task(name: str):
            order.append(name)

        base = time.perf_counter_ns()
        scheduler.schedule_at_perf_ns(base + int(0.3 * NANOS_PER_SECOND), task("third"))
        scheduler.schedule_at_perf_ns(base + int(0.1 * NANOS_PER_SECOND), task("first"))
        scheduler.schedule_at_perf_ns(
            base + int(0.2 * NANOS_PER_SECOND), task("second")
        )

        await asyncio.sleep(0.5)

        assert order == ["first", "second", "third"]


# =============================================================================
# Cancellation Tests
# =============================================================================


class TestCancelAllPending:
    """Tests for cancel_all_pending."""

    async def test_cancels_timers(self, scheduler: LoopScheduler):
        """Pending timers are cancelled and coroutines closed."""
        executed = []

        async def task(name: str):
            executed.append(name)

        scheduler.schedule_later(1.0, task("one"))
        scheduler.schedule_later(2.0, task("two"))

        assert scheduler.pending_count == 2

        scheduler.cancel_all_pending()

        assert scheduler.pending_count == 0

        # Advance time past when they would have fired
        await asyncio.sleep(3.0)

        assert executed == []

    async def test_closes_coroutines(self, scheduler: LoopScheduler):
        """Cancelled coroutines have their cleanup handled."""
        finally_ran = []

        async def task_with_finally():
            try:
                await asyncio.sleep(10)
            finally:
                finally_ran.append(True)

        scheduler.schedule_later(1.0, task_with_finally())

        # Cancel before timer fires - coroutine is closed, not awaited
        # Note: close() runs finally blocks for generators but not coroutines
        # that haven't started. This is Python's coroutine semantics.
        scheduler.cancel_all_pending()

        assert scheduler.pending_count == 0


class TestCancelAllRunning:
    """Tests for cancel_all_running."""

    async def test_cancels_tasks(self, scheduler: LoopScheduler):
        """Running tasks are cancelled."""
        started = asyncio.Event()
        completed = []

        async def long_task():
            started.set()
            try:
                await asyncio.sleep(100)
                completed.append("done")
            except asyncio.CancelledError:
                completed.append("cancelled")
                raise

        scheduler.execute_async(long_task())
        await yield_to_scheduler()
        await started.wait()

        assert scheduler.running_count == 1

        tasks = scheduler.cancel_all_running()

        assert len(tasks) == 1
        assert scheduler.running_count == 0

        await asyncio.gather(*tasks, return_exceptions=True)

        assert completed == ["cancelled"]

    async def test_returns_tasks_for_cleanup(self, scheduler: LoopScheduler):
        """Returned tasks can be awaited for cleanup."""
        cleanup_done = []

        async def task_with_cleanup():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cleanup_done.append(True)
                raise

        scheduler.execute_async(task_with_cleanup())
        await yield_to_scheduler()

        tasks = scheduler.cancel_all_running()
        await asyncio.gather(*tasks, return_exceptions=True)

        assert cleanup_done == [True]


class TestCancelAll:
    """Tests for cancel_all (both pending and running)."""

    async def test_cancels_everything(self, scheduler: LoopScheduler):
        """Both pending timers and running tasks are cancelled."""
        results = []

        async def quick_task():
            results.append("quick")

        async def slow_task():
            await asyncio.sleep(100)
            results.append("slow")

        scheduler.execute_async(quick_task())
        scheduler.schedule_later(1.0, slow_task())
        scheduler.schedule_later(2.0, slow_task())

        await yield_to_scheduler()

        assert scheduler.pending_count == 2

        tasks = scheduler.cancel_all()

        assert scheduler.pending_count == 0
        assert scheduler.running_count == 0

        await asyncio.gather(*tasks, return_exceptions=True)

        assert results == ["quick"]


# =============================================================================
# State Property Tests
# =============================================================================


class TestStateProperties:
    """Tests for pending_count, running_count, is_idle."""

    async def test_pending_count_tracks_scheduled_timers(
        self, scheduler: LoopScheduler
    ):
        """pending_count reflects scheduled but not-yet-fired timers."""

        async def noop():
            pass

        assert scheduler.pending_count == 0

        scheduler.schedule_later(1.0, noop())
        assert scheduler.pending_count == 1

        scheduler.schedule_later(2.0, noop())
        assert scheduler.pending_count == 2

        await asyncio.sleep(2.5)

        assert scheduler.pending_count == 0

    async def test_running_count_tracks_executing_tasks(self, scheduler: LoopScheduler):
        """running_count reflects currently executing tasks."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_task():
            started.set()
            await release.wait()

        assert scheduler.running_count == 0

        scheduler.execute_async(blocking_task())
        await yield_to_scheduler()
        await started.wait()

        assert scheduler.running_count == 1

        release.set()
        await yield_to_scheduler()

        assert scheduler.running_count == 0

    async def test_is_idle_when_nothing_pending_or_running(
        self, scheduler: LoopScheduler
    ):
        """is_idle() returns True only when completely empty."""

        async def noop():
            pass

        assert scheduler.is_idle() is True

        scheduler.schedule_later(1.0, noop())
        assert scheduler.is_idle() is False

        scheduler.cancel_all_pending()
        assert scheduler.is_idle() is True


# =============================================================================
# Exception Handling Tests
# =============================================================================


class TestExceptionHandling:
    """Tests for task exception handling."""

    async def test_exception_handler_called_on_task_failure(
        self, scheduler: LoopScheduler
    ):
        """Exception handler is invoked when a task raises."""
        exceptions = []

        def handler(task: asyncio.Task):
            exceptions.append(task.exception())

        scheduler.set_exception_handler(handler)

        async def failing_task():
            raise ValueError("test error")

        scheduler.execute_async(failing_task())
        await yield_to_scheduler(n=3)  # Extra yield for done callback

        assert len(exceptions) == 1
        assert isinstance(exceptions[0], ValueError)
        assert str(exceptions[0]) == "test error"

    async def test_no_exception_handler_is_safe(self, scheduler: LoopScheduler):
        """Tasks can fail without exception handler (no crash)."""

        async def failing_task():
            raise ValueError("test error")

        scheduler.execute_async(failing_task())
        await yield_to_scheduler()

    async def test_cancelled_tasks_dont_trigger_exception_handler(
        self, scheduler: LoopScheduler
    ):
        """CancelledError is not passed to exception handler."""
        exceptions = []

        def handler(task: asyncio.Task):
            exceptions.append(task.exception())

        scheduler.set_exception_handler(handler)

        async def slow_task():
            await asyncio.sleep(100)

        scheduler.execute_async(slow_task())
        await yield_to_scheduler()

        tasks = scheduler.cancel_all_running()
        await asyncio.gather(*tasks, return_exceptions=True)

        assert exceptions == []


# =============================================================================
# Edge Cases and Integration
# =============================================================================


class TestEdgeCases:
    """Edge cases and integration tests."""

    async def test_repr_shows_counts(self, scheduler: LoopScheduler):
        """__repr__ includes pending and running counts."""

        async def noop():
            pass

        assert "pending=0" in repr(scheduler)
        assert "running=0" in repr(scheduler)

        scheduler.schedule_later(1.0, noop())
        assert "pending=1" in repr(scheduler)

        scheduler.cancel_all_pending()

    async def test_schedule_from_within_scheduled_task(self, scheduler: LoopScheduler):
        """Tasks can schedule more tasks."""
        order = []

        async def outer():
            order.append("outer_start")
            scheduler.execute_async(inner())
            order.append("outer_end")

        async def inner():
            order.append("inner")

        scheduler.execute_async(outer())
        await yield_to_scheduler(n=3)  # Allow outer to run and inner to complete

        assert order == ["outer_start", "outer_end", "inner"]

    async def test_high_volume_scheduling(self, scheduler: LoopScheduler):
        """Handles many scheduled tasks efficiently."""
        count = [0]

        async def increment():
            count[0] += 1

        # Schedule 100 tasks at various times (start at 0.01 to avoid zero-delay execute_async path)
        for i in range(100):
            scheduler.schedule_later((i + 1) * 0.01, increment())

        assert scheduler.pending_count == 100

        await asyncio.sleep(2.0)

        assert count[0] == 100
        assert scheduler.pending_count == 0
