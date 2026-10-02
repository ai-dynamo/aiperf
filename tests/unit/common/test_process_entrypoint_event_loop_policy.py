# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every real AIPerf process entrypoint must apply the Windows event-loop
policy switch (``aiperf.common.event_loop.configure_event_loop_policy_for_platform``)
before it constructs its first event loop via ``asyncio.run()`` /
``uvloop.run()``.

``bootstrap_and_run_service`` (used by ``aiperf run`` / ``aiperf service`` /
every in-process service subprocess) already does this. This module covers
the entrypoints that historically did NOT: the ``aiperf`` console script
itself (``aiperf.cli``, which every other subcommand -- ``profile``'s
preflight/multi-run/single-run paths, ``proxy``, ``chat`` -- is reached
through), and the two standalone ``python -m`` module entrypoints
(``aiperf.sweep_controller.main``, ``aiperf.orchestrator.subprocess_runner``)
that never import ``aiperf.cli`` at all.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


class TestCliModuleAppliesEventLoopPolicyAtImportTime:
    """The ``aiperf`` console script (`aiperf = "aiperf.cli:app"`) and
    ``python -m aiperf`` both resolve to importing ``aiperf.cli`` before any
    subcommand's ``asyncio.run()``/``uvloop.run()`` call can execute. The
    policy switch must therefore happen at ``aiperf.cli`` import time."""

    def test_import_calls_configure_event_loop_policy(self) -> None:
        with patch(
            "aiperf.common.event_loop.configure_event_loop_policy_for_platform"
        ) as mock_configure:
            sys.modules.pop("aiperf.cli", None)
            try:
                importlib.import_module("aiperf.cli")
            finally:
                sys.modules.pop("aiperf.cli", None)

        mock_configure.assert_called_once_with()


class TestSubprocessRunnerEntrypointOrdering:
    """``python -m aiperf.orchestrator.subprocess_runner`` must configure the
    event-loop policy before ``main()`` (which eventually calls
    ``asyncio.run()`` deep inside ``bootstrap_and_run_service``) runs."""

    def test_script_entrypoint_configures_policy_before_main(self) -> None:
        from aiperf.orchestrator import subprocess_runner

        manager = MagicMock()
        with (
            patch(
                "aiperf.common.event_loop.configure_event_loop_policy_for_platform",
                manager.configure_event_loop_policy_for_platform,
            ),
            patch.object(
                subprocess_runner,
                "_release_inherited_pipes_on_windows",
                manager.release_pipes,
            ),
            patch.object(subprocess_runner, "main", manager.main),
        ):
            subprocess_runner._script_entrypoint()

        assert [c[0] for c in manager.mock_calls] == [
            "configure_event_loop_policy_for_platform",
            "release_pipes",
            "main",
        ]


class TestSweepControllerEntrypointOrdering:
    """``python -m aiperf.sweep_controller.main`` must configure the
    event-loop policy before ``asyncio.run(main())`` constructs the loop --
    setting it from inside the already-running async ``main()`` would be too
    late."""

    def test_script_entrypoint_configures_policy_before_asyncio_run(self) -> None:
        from aiperf.sweep_controller import main as sweep_controller_main

        manager = MagicMock()
        manager.asyncio_run.return_value = 0
        with (
            patch(
                "aiperf.common.event_loop.configure_event_loop_policy_for_platform",
                manager.configure_event_loop_policy_for_platform,
            ),
            patch.object(sweep_controller_main.asyncio, "run", manager.asyncio_run),
        ):
            result = sweep_controller_main._script_entrypoint()

        assert result == 0
        assert [c[0] for c in manager.mock_calls] == [
            "configure_event_loop_policy_for_platform",
            "asyncio_run",
        ]
