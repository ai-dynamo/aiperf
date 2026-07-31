# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import pytest

from aiperf.common.constants import IS_LINUX
from aiperf.timing.high_res_timer import ThreadPacer, TimerFdPacer


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

    async def test_close_is_idempotent_and_stops_thread(self):
        pacer = ThreadPacer()
        pacer.close()
        pacer.close()
        pacer._thread.join(timeout=2)
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
