# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import pytest

from aiperf.common.constants import IS_LINUX


@pytest.mark.asyncio
class TestThreadPacer:
    async def test_sleep_until_wakes_at_deadline_not_before(self):
        from aiperf.timing.high_res_timer import ThreadPacer

        pacer = ThreadPacer()
        try:
            errors_us = []
            for _ in range(50):
                deadline = time.perf_counter() + 0.0002
                await pacer.sleep_until(deadline)
                errors_us.append((time.perf_counter() - deadline) * 1e6)
            assert min(errors_us) >= 0.0
            errors_us.sort()
            # Well under the ~1ms event-loop timer wheel this replaces
            # (thread nanosleep + call_soon_threadsafe wake).
            assert errors_us[len(errors_us) // 2] < 900.0
        finally:
            pacer.close()

    async def test_sleep_until_past_deadline_returns_immediately(self):
        from aiperf.timing.high_res_timer import ThreadPacer

        pacer = ThreadPacer()
        try:
            start = time.perf_counter()
            await pacer.sleep_until(start - 1.0)
            assert time.perf_counter() - start < 0.1
        finally:
            pacer.close()

    async def test_close_is_idempotent_and_stops_thread(self):
        from aiperf.timing.high_res_timer import ThreadPacer

        pacer = ThreadPacer()
        pacer.close()
        pacer.close()
        pacer._thread.join(timeout=2)
        assert not pacer._thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.skipif(not IS_LINUX, reason="timerfd is Linux-only")
class TestTimerFdPacer:
    async def test_sleep_until_wakes_at_deadline_not_before(self):
        from aiperf.timing.high_res_timer import TimerFdPacer

        pacer = TimerFdPacer()
        try:
            errors_us = []
            for _ in range(50):
                deadline = time.perf_counter() + 0.0002
                await pacer.sleep_until(deadline)
                errors_us.append((time.perf_counter() - deadline) * 1e6)
            # Never wakes early (kernel holds absolute deadlines), and stays
            # well under the ~1ms event-loop timer granularity it replaces.
            assert min(errors_us) >= 0.0
            errors_us.sort()
            assert errors_us[len(errors_us) // 2] < 500.0
        finally:
            pacer.close()

    async def test_sleep_until_past_deadline_returns_immediately(self):
        from aiperf.timing.high_res_timer import TimerFdPacer

        pacer = TimerFdPacer()
        try:
            start = time.perf_counter()
            await pacer.sleep_until(start - 1.0)
            assert time.perf_counter() - start < 0.05
        finally:
            pacer.close()

    async def test_close_is_idempotent(self):
        from aiperf.timing.high_res_timer import TimerFdPacer

        pacer = TimerFdPacer()
        pacer.close()
        pacer.close()
