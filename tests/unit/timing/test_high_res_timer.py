# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
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

    async def test_cancelled_sleep_does_not_wake_replacement_early(self) -> None:
        pacer = ThreadPacer()
        try:
            first_deadline = time.perf_counter() + 0.1
            first_sleep = asyncio.create_task(pacer.sleep_until(first_deadline))
            for _ in range(100):
                with pacer._condition:
                    if pacer._waiting_generation is not None:
                        break
                await asyncio.sleep(0)
            else:
                pytest.fail("pacer worker did not start waiting")

            first_sleep.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await first_sleep

            replacement_deadline = time.perf_counter() + 0.15
            replacement_sleep = asyncio.create_task(
                pacer.sleep_until(replacement_deadline)
            )
            while time.perf_counter() < first_deadline + 0.02:
                await asyncio.sleep(0)
            assert not replacement_sleep.done()

            await replacement_sleep
            assert time.perf_counter() >= replacement_deadline
        finally:
            pacer.close()
            pacer._thread.join(timeout=2)

    async def test_close_is_idempotent_and_stops_thread(self) -> None:
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
