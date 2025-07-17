# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from aiperf.common.hooks import AIPerfHook, AIPerfTaskMixin, aiperf_task


class ExampleTaskClass(AIPerfTaskMixin):
    def __init__(self):
        super().__init__()
        self.lock = asyncio.Lock()
        self.running = False

    @aiperf_task
    async def _run_task(self):
        self.running = True

        while True:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

        self.running = False


@pytest.mark.asyncio
async def test_aiperf_task():
    task_class = ExampleTaskClass()

    assert not task_class.running, "Task should not be running before starting"
    await task_class._start_tasks()
    await asyncio.sleep(0.01)  # avoid race condition
    assert task_class.running, "Task should be running after starting"

    await task_class._stop_tasks()
    await asyncio.sleep(0.01)  # avoid race condition
    assert not task_class.running, "Task should not be running after stopping"


class ExampleTaskClass2(ExampleTaskClass):
    async def initialize(self):
        await self.run_hooks(AIPerfHook.ON_INIT)

    async def stop(self):
        await self.run_hooks(AIPerfHook.ON_STOP)


@pytest.mark.asyncio
async def test_aiperf_task_will_run_on_init_and_stop():
    task_class = ExampleTaskClass2()
    await task_class.initialize()
    await asyncio.sleep(0.01)  # avoid race condition
    assert task_class.running, "Task should be running after starting"

    await task_class.stop()
    await asyncio.sleep(0.01)  # avoid race condition
    assert not task_class.running, "Task should not be running after stopping"
