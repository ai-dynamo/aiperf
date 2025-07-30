# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from aiperf.common.hooks import background_task
from aiperf.common.mixins import AIPerfLifecycleMixin
from tests.utils.time_traveler import TimeTraveler


class ExampleTaskClass(AIPerfLifecycleMixin):
    def __init__(self):
        super().__init__()
        self.lock = asyncio.Lock()
        self.running = False

    @background_task(immediate=True, interval=None)
    async def _run_task(self):
        async with self.lock:
            self.running = True

        while not self.stop_requested:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

        async with self.lock:
            self.running = False


@pytest.mark.asyncio
async def test_background_task(time_traveler: TimeTraveler):
    task_class = ExampleTaskClass()
    assert not task_class.running, "Task should not be running before starting"
    await task_class.initialize()
    await task_class.start()
    for _ in range(3):  # yield a few times to ensure the task got scheduled
        await time_traveler.yield_to_event_loop()
    async with task_class.lock:
        assert task_class.running, "Task should be running after starting"
    await task_class.stop()
    for _ in range(3):  # yield a few times to ensure the task got scheduled
        await time_traveler.yield_to_event_loop()
    async with task_class.lock:
        assert not task_class.running, "Task should not be running after stopping"
