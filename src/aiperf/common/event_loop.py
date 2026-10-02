# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide asyncio event loop policy configuration.

Deliberately dependency-light -- only ``asyncio`` and
``aiperf.common.constants`` -- so it can be imported eagerly at the very top
of every real process entrypoint (the ``aiperf`` console script in
``aiperf.cli``, and the standalone ``python -m`` module entrypoints such as
``aiperf.sweep_controller.main`` and ``aiperf.orchestrator.subprocess_runner``)
without pulling in aiperf's much heavier import graph (e.g.
``aiperf.common.environment``) just to flip one platform-conditional flag.
"""

import asyncio

from aiperf.common.constants import IS_WINDOWS


def configure_event_loop_policy_for_platform() -> None:
    """On Windows, switch to ``WindowsSelectorEventLoopPolicy`` before the
    event loop is created.

    pyzmq's async sockets call ``loop.add_reader()`` / ``loop.add_writer()``,
    which the default ``ProactorEventLoop`` on Windows does not implement.
    The selector policy must be set before ``asyncio.run()``/``uvloop.run()``
    constructs the loop.

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
    if IS_WINDOWS:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
