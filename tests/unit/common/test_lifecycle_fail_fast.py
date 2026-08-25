# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup hooks fail fast; failure-path shutdown is bounded.

Without both, a service that fails during on_init/on_start becomes a silent
zombie container: later startup hooks keep running against an inconsistent
state (background tasks spawned after a probe already failed), and a blocked
on_stop hook then keeps the process alive indefinitely.
"""

from __future__ import annotations

import asyncio

import pytest

from aiperf.common.enums import LifecycleState
from aiperf.common.hooks import AIPerfHook, on_init, on_start, on_stop
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin


class _Recorder(AIPerfLifecycleMixin):
    """Lifecycle whose init hooks record their invocation order."""

    def __init__(self, **kwargs) -> None:
        self.ran: list[str] = []
        super().__init__(**kwargs)

    @on_init
    async def _first(self) -> None:
        self.ran.append("first")
        raise RuntimeError("probe failed")

    @on_init
    async def _second(self) -> None:
        self.ran.append("second")


class TestStartupFailFast:
    @pytest.mark.asyncio
    async def test_later_init_hooks_do_not_run_after_a_failure(self) -> None:
        component = _Recorder()
        with pytest.raises(asyncio.CancelledError):
            await component.initialize()
        assert component.ran == ["first"], (
            "a later on_init hook ran after an earlier one failed"
        )
        assert component.state is LifecycleState.FAILED

    @pytest.mark.asyncio
    async def test_stop_hooks_still_collect_every_error(self) -> None:
        """Cleanup stays best-effort so errors don't mask each other."""
        ran: list[str] = []

        class _Stopper(AIPerfLifecycleMixin):
            @on_start
            async def _noop(self) -> None:
                pass

            @on_stop
            async def _stop_a(self) -> None:
                ran.append("a")
                raise RuntimeError("a failed")

            @on_stop
            async def _stop_b(self) -> None:
                ran.append("b")

        component = _Stopper()
        await component.initialize()
        await component.start()
        # CancelledError (a BaseException) is how _fail re-raises.
        with pytest.raises(BaseException):  # noqa: B017,PT011
            await component.stop()
        assert set(ran) == {"a", "b"}


class TestRunHooksFailFastFlag:
    @pytest.mark.asyncio
    async def test_flag_defaults_to_collecting(self) -> None:
        component = _Recorder()
        with pytest.raises(Exception):  # noqa: B017 - AIPerfMultiError
            await component.run_hooks(AIPerfHook.ON_INIT)
        assert component.ran == ["first", "second"]

    @pytest.mark.asyncio
    async def test_flag_aborts_on_first_failure(self) -> None:
        component = _Recorder()
        with pytest.raises(Exception):  # noqa: B017 - HookError
            await component.run_hooks(AIPerfHook.ON_INIT, fail_fast=True)
        assert component.ran == ["first"]


class TestFailureShutdownIsBounded:
    def test_timeout_setting_exists(self) -> None:
        from aiperf.common.environment import Environment

        assert Environment.SERVICE.FAILURE_SHUTDOWN_TIMEOUT > 0
