# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide asyncio event loop policy configuration.

Deliberately dependency-light -- only the standard library and
``aiperf.common.constants`` -- so it can be imported eagerly at the very top
of every real process entrypoint (the ``aiperf`` console script in
``aiperf.cli``, and the standalone ``python -m`` module entrypoints such as
``aiperf.sweep_controller.main`` and ``aiperf.orchestrator.subprocess_runner``)
without pulling in aiperf's much heavier import graph (e.g.
``aiperf.common.environment``) just to flip one platform-conditional flag.
"""

import asyncio
import sys
import time

from aiperf.common.constants import IS_WINDOWS

# Before 3.13, time.monotonic() on Windows is GetTickCount64(), which advances
# in 15.625 ms ticks regardless of timeBeginPeriod (gh-88494).
_MONOTONIC_IS_COARSE_ON_WINDOWS = sys.version_info < (3, 13)

# Bound at import so a test patching time.perf_counter globally (with a short
# side_effect list, say) cannot drive or exhaust the event loop's clock.
_perf_counter = time.perf_counter


class PerfCounterSelectorEventLoop(asyncio.SelectorEventLoop):
    """A ``SelectorEventLoop`` whose clock is ``time.perf_counter()``.

    The stock loop reads ``time.monotonic()`` and fires any timer due within
    one clock resolution of now. With a 15.625 ms monotonic clock, timers fire
    up to a tick early and elapsed times read in whole ticks. On Windows
    ``perf_counter()`` is ``QueryPerformanceCounter()`` (1 us), the clock
    Python 3.13 moved ``monotonic()`` to.
    """

    def __init__(self) -> None:
        super().__init__()
        self._clock_resolution = time.get_clock_info("perf_counter").resolution

    def time(self) -> float:
        return _perf_counter()


class PerfCounterSelectorEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Builds ``PerfCounterSelectorEventLoop`` instances."""

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        return PerfCounterSelectorEventLoop()


def aiperf_event_loop_policy() -> asyncio.AbstractEventLoopPolicy | None:
    """The event loop policy aiperf runs on this platform, or ``None`` to keep
    the platform default."""
    if not IS_WINDOWS:
        return None
    if _MONOTONIC_IS_COARSE_ON_WINDOWS:
        return PerfCounterSelectorEventLoopPolicy()
    return asyncio.WindowsSelectorEventLoopPolicy()


def configure_event_loop_policy_for_platform() -> None:
    """On Windows, switch to a selector-based event loop before the event
    loop is created.

    pyzmq's async sockets call ``loop.add_reader()`` / ``loop.add_writer()``,
    which the default ``ProactorEventLoop`` on Windows does not implement.
    The selector policy must be set before ``asyncio.run()``/``uvloop.run()``
    constructs the loop. On Python < 3.13 the loop also reads
    ``perf_counter()`` instead of the 15.625 ms ``monotonic()`` clock; see
    ``PerfCounterSelectorEventLoop``.

    This must run before the FIRST ``asyncio.run()``/``uvloop.run()`` call in
    a given process, because ``asyncio.set_event_loop_policy()`` only affects
    loops created afterward -- it does not retroactively change one that
    already exists. Every real AIPerf process entrypoint must therefore call
    this as early as possible during startup: the ``aiperf`` console script
    (``aiperf.cli``, which every subcommand -- ``run``, ``service``, ``proxy``,
    ``chat``, ``profile`` and everything it drives, including preflight and
    multi-run orchestration -- is reached through), and each standalone
    ``python -m`` module entrypoint (``aiperf.sweep_controller.main``,
    ``aiperf.orchestrator.subprocess_runner``). Do not rely on any single
    subcommand or service bootstrap path to cover this for the whole
    process.

    uvloop is already auto-disabled on Windows via ``environment.py``, so on
    Windows this only matters for the asyncio path. On non-Windows platforms
    this is a no-op -- the default policy is already correct.
    """
    policy = aiperf_event_loop_policy()
    if policy is not None:
        asyncio.set_event_loop_policy(policy)
