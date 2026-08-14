# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time

import pytest

from aiperf.common.constants import IS_LINUX
from aiperf.timing.high_res_timer import ThreadPacer, TimerFdPacer

# The autouse no_sleep fixture rewrites asyncio.sleep to a bare event-loop yield,
# so spinning on it never releases the GIL and the pacer worker thread may never
# be scheduled. These helpers block in a worker thread (via asyncio.to_thread) so
# the real time.sleep hands the interpreter over without stalling the loop.


def _worker_is_waiting(pacer: ThreadPacer, timeout: float = 5.0) -> bool:
    """Block until the pacer worker has parked on its deadline."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        with pacer._condition:
            if pacer._waiting_generation is not None:
                return True
        time.sleep(0.001)
    return False


def _block_until(deadline_perf_s: float) -> None:
    """Block in real time until an absolute perf_counter deadline."""
    remaining = deadline_perf_s - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


@pytest.mark.asyncio
class TestThreadPacer:
    async def test_sleep_until_wakes_at_deadline_not_before(self):
        pacer = ThreadPacer()
        try:
            errors_us = []
            for _ in range(50):
                deadline = time.perf_counter() + 0.0002
                await pacer.sleep_until(deadline)
                errors_us.append((time.perf_counter() - deadline) * 1e6)
            assert min(errors_us) >= 0.0
        finally:
            pacer.close()

    async def test_sleep_until_past_deadline_returns_immediately(self):
        pacer = ThreadPacer()
        try:
            start = time.perf_counter()
            await pacer.sleep_until(start - 1.0)
            assert time.perf_counter() - start < 0.1
        finally:
            pacer.close()

    async def test_cancelled_sleep_does_not_wake_replacement_early(self) -> None:
        pacer = ThreadPacer()
        try:
            first_deadline = time.perf_counter() + 0.3
            first_sleep = asyncio.create_task(pacer.sleep_until(first_deadline))
            assert await asyncio.to_thread(_worker_is_waiting, pacer), (
                "pacer worker did not start waiting"
            )

            first_sleep.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_sleep

            replacement_deadline = time.perf_counter() + 0.5
            replacement_sleep = asyncio.create_task(
                pacer.sleep_until(replacement_deadline)
            )
            # Past the cancelled sleep's deadline: a leaked stale tick would have
            # completed the replacement early.
            await asyncio.to_thread(_block_until, first_deadline + 0.05)
            assert not replacement_sleep.done()

            await replacement_sleep
            assert time.perf_counter() >= replacement_deadline
        finally:
            pacer.close()
            await asyncio.to_thread(pacer._thread.join, 2)
            assert not pacer._thread.is_alive()

    async def test_close_is_idempotent_and_stops_thread(self) -> None:
        pacer = ThreadPacer()
        pacer.close()
        pacer.close()
        await asyncio.to_thread(pacer._thread.join, 2)
        assert not pacer._thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.skipif(not IS_LINUX, reason="timerfd is Linux-only")
class TestTimerFdPacer:
    async def test_sleep_until_wakes_at_deadline_not_before(self):
        pacer = TimerFdPacer()
        try:
            errors_us = []
            for _ in range(50):
                deadline = time.perf_counter() + 0.0002
                await pacer.sleep_until(deadline)
                errors_us.append((time.perf_counter() - deadline) * 1e6)
            assert min(errors_us) >= 0.0
        finally:
            pacer.close()

    async def test_sleep_until_past_deadline_returns_immediately(self):
        pacer = TimerFdPacer()
        try:
            start = time.perf_counter()
            await pacer.sleep_until(start - 1.0)
            assert time.perf_counter() - start < 0.05
        finally:
            pacer.close()

    async def test_close_is_idempotent(self):
        pacer = TimerFdPacer()
        pacer.close()
        pacer.close()
