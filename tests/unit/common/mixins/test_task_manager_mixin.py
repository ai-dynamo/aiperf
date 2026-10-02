# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from aiperf.common.mixins.task_manager_mixin import TaskManagerMixin


class _TaskManager(TaskManagerMixin):
    """Minimal standalone TaskManagerMixin user, standing in for a DEALER client."""

    def __init__(self):
        super().__init__()


class TestCancelAllTasksSelfCancellation:
    """Reproduces the SHUTDOWN-handler-cancels-its-own-task race.

    ``BaseComponentService.control_client`` is a ``ZMQStreamingDealerClient``
    (a ``TaskManagerMixin``). Inbound control-channel commands -- including
    SHUTDOWN -- are dispatched via ``control_client.execute_async(...)``, so
    the resulting task lands in the DEALER client's own ``self.tasks`` set,
    not the service's. The SHUTDOWN handler then calls ``await self.stop()``
    on the *service*, whose teardown calls ``comms.stop()``, which stops the
    DEALER client child, whose ``@on_stop`` hook calls
    ``self.cancel_all_tasks()`` on that same tracked set -- cancelling the
    very task that is currently running this call chain.
    """

    async def test_cancel_all_tasks_does_not_cancel_the_calling_task(self):
        """The fix: the calling task's own await self.stop() must complete.

        This is the green-path counterpart of the test above: once
        ``cancel_all_tasks`` excludes ``asyncio.current_task()``, the task
        that is itself running the SHUTDOWN handler must be able to finish
        its own teardown instead of being cancelled by its own call.
        """
        manager = _TaskManager()
        reached_after_cancel: dict[str, bool] = {}

        async def self_shutdown_handler() -> None:
            await manager.cancel_all_tasks()
            await asyncio.sleep(0)
            reached_after_cancel["done"] = True

        task = manager.execute_async(self_shutdown_handler())
        await asyncio.wait_for(task, timeout=1)

        assert reached_after_cancel.get("done") is True

    async def test_cancel_all_tasks_still_cancels_other_tasks(self):
        """Other tracked tasks must still be cancelled, only the caller is exempt."""
        manager = _TaskManager()
        other_cancelled = asyncio.Event()
        # A never-set Event (not asyncio.sleep, which the test suite's
        # auto-fixture makes instantaneous) parks the task indefinitely so it
        # is still pending when cancel_all_tasks runs.
        never_set = asyncio.Event()

        async def other_task() -> None:
            try:
                await never_set.wait()
            except asyncio.CancelledError:
                other_cancelled.set()
                raise

        other = manager.execute_async(other_task())
        # Let the other task start running so it is parked on never_set.wait().
        await asyncio.sleep(0)

        async def caller() -> None:
            await manager.cancel_all_tasks()

        caller_task = manager.execute_async(caller())
        await asyncio.wait_for(caller_task, timeout=1)

        with pytest.raises(asyncio.CancelledError):
            await other
        assert other_cancelled.is_set()
