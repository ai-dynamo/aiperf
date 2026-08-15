# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PhaseRunner._teardown_strategy`` routing and its failure guard.

The guard exists because the deferred path hands the teardown to
``execute_async``, which never retrieves the task result -- an exception there
would vanish silently instead of surfacing as a warning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit.timing.phase.test_runner import (
    cfg,
    make_runner,
    mock_callback,
    mock_cancel_policy,
    mock_conc_mgr,
)


@dataclass
class _TeardownStrategy:
    """Strategy implementing the optional ``teardown_phase`` hook."""

    raises: bool = False
    calls: list[int] = field(default_factory=list)

    async def setup_phase(self) -> None: ...

    async def execute_phase(self) -> None: ...

    async def handle_credit_return(self, credit) -> None: ...

    async def teardown_phase(self) -> None:
        self.calls.append(1)
        if self.raises:
            raise RuntimeError("teardown boom")


@dataclass
class _NoTeardownStrategy:
    """Linear-style strategy without the hook -- the runner must skip it."""

    async def setup_phase(self) -> None: ...

    async def execute_phase(self) -> None: ...

    async def handle_credit_return(self, credit) -> None: ...


def _runner():
    """PhaseRunner over inert doubles; only the teardown routing is under test."""
    conv_src = MagicMock()
    pub = MagicMock()
    pub.publish_phase_complete = AsyncMock()
    router = MagicMock()
    router.send_credit = AsyncMock()
    return make_runner(
        cfg(),
        conv_src,
        pub,
        router,
        mock_conc_mgr(),
        mock_cancel_policy(),
        mock_callback(),
    )


@pytest.mark.asyncio
async def test_teardown_awaited_inline_when_no_return_wait_is_pending() -> None:
    """With no in-flight return wait the hook is awaited before the call returns."""
    runner = _runner()
    strategy = _TeardownStrategy()

    await runner._teardown_strategy(strategy)

    assert strategy.calls == [1]


@pytest.mark.asyncio
async def test_strategy_without_the_hook_is_skipped() -> None:
    """A strategy that does not implement the protocol is a no-op, not an error."""
    runner = _runner()

    await runner._teardown_strategy(_NoTeardownStrategy())


@pytest.mark.asyncio
async def test_raising_teardown_is_logged_not_propagated() -> None:
    """A failing hook must not abort the phase's ``finally``.

    ``_teardown_strategy`` runs in ``run()``'s ``finally``, so letting the
    exception escape would mask the phase's real outcome.
    """
    runner = _runner()
    warnings: list[str] = []
    runner.warning = warnings.append
    strategy = _TeardownStrategy(raises=True)

    await runner._teardown_strategy(strategy)

    assert strategy.calls == [1]
    assert any("teardown_phase failed" in w for w in warnings)


@pytest.mark.asyncio
async def test_teardown_is_deferred_until_the_return_wait_completes() -> None:
    """A non-final seamless phase defers: its returns are still in flight.

    The strategy's return observer must stay installed until they land, so the
    hook must not fire while the return-wait task is pending.
    """
    runner = _runner()
    gate = asyncio.Event()

    async def _return_wait() -> None:
        await gate.wait()

    runner._return_wait_task = asyncio.create_task(_return_wait())
    strategy = _TeardownStrategy()

    await runner._teardown_strategy(strategy)
    assert strategy.calls == []

    gate.set()
    await runner._return_wait_task
    await asyncio.gather(*list(runner.tasks))

    assert strategy.calls == [1]


@pytest.mark.asyncio
async def test_deferred_teardown_failure_is_logged_not_swallowed() -> None:
    """The guard covers the deferred path too.

    ``execute_async`` never retrieves the task result, so an unguarded raise
    here would be discarded with no diagnostic at all.
    """
    runner = _runner()
    warnings: list[str] = []
    runner.warning = warnings.append
    gate = asyncio.Event()

    async def _return_wait() -> None:
        await gate.wait()

    runner._return_wait_task = asyncio.create_task(_return_wait())
    strategy = _TeardownStrategy(raises=True)

    await runner._teardown_strategy(strategy)
    gate.set()
    await runner._return_wait_task
    results = await asyncio.gather(*list(runner.tasks), return_exceptions=True)

    assert strategy.calls == [1]
    assert any("teardown_phase failed" in w for w in warnings)
    assert not [r for r in results if isinstance(r, BaseException)]


@pytest.mark.asyncio
async def test_already_finished_return_wait_tears_down_inline() -> None:
    """A completed return-wait task takes the inline path, not the deferred one."""
    runner = _runner()

    async def _done() -> None:
        return None

    runner._return_wait_task = asyncio.create_task(_done())
    await runner._return_wait_task
    strategy = _TeardownStrategy()

    await runner._teardown_strategy(strategy)

    assert strategy.calls == [1]
