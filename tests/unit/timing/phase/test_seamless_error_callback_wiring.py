# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every seamless non-final runner must get the fatal-error callback.

The post-loop barrier in ``_execute_phases`` only inspects runners still in
``_active_runners``, and a seamless phase's complete-callback removes the runner
from that list as soon as it drains. So for a phase that drains before the loop
ends, ``set_phase_error_callback`` is the ONLY route by which a fatal
request-free control-node failure can fail the run. It must be installed
regardless of whether the phase also defers a server-profiler stop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.control_hooks import PreparedEndpointControlHooks
from aiperf.common.enums import CreditPhase
from aiperf.timing.phase_orchestrator import PhaseOrchestrator

from ..conftest import make_phase_config


def _orchestrator(*, with_profiler_hooks: bool) -> PhaseOrchestrator:
    orch = PhaseOrchestrator.__new__(PhaseOrchestrator)
    orch._ordered_phase_configs = [
        make_phase_config(phase=CreditPhase.PROFILING, seamless=True),
        make_phase_config(phase=CreditPhase.PROFILING),
    ]
    orch._active_runners = []
    orch._server_profiler_owners = set()
    orch._deferred_profiler_stops = set()
    orch._seamless_phase_error = None
    orch._control_headers = {}
    orch._control_hooks = (
        PreparedEndpointControlHooks(
            timeout_s=1.0,
            reset_urls=[],
            profiler_start_urls=["http://a:8000/start_profile"],
            profiler_stop_urls=["http://a:8000/stop_profile"],
            profiler_timeout_s=1.0,
            reset_max_retry_seconds=1.0,
        )
        if with_profiler_hooks
        else None
    )
    for attr in (
        "_conversation_source",
        "_phase_publisher",
        "_credit_router",
        "_concurrency_manager",
        "_callback_handler",
        "_url_sampler",
        "_run",
        "_session_tree_registry",
    ):
        setattr(orch, attr, MagicMock())
    return orch


@pytest.mark.parametrize(
    "with_profiler_hooks",
    [
        pytest.param(False, id="no-profiler-hooks"),
        pytest.param(True, id="profiler-defers-stop"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_seamless_non_final_runner_always_gets_error_callback(
    with_profiler_hooks: bool,
) -> None:
    orch = _orchestrator(with_profiler_hooks=with_profiler_hooks)
    created: list[MagicMock] = []

    def make_runner(**_kwargs: object) -> MagicMock:
        runner = MagicMock()
        runner.run = MagicMock(return_value=_noop())
        runner.return_wait_task = None
        runner.control_fatal_error = None
        created.append(runner)
        return runner

    async def _noop() -> None:
        return None

    with (
        patch("aiperf.timing.phase_orchestrator.PhaseRunner", side_effect=make_runner),
        patch.object(
            PhaseOrchestrator, "_start_server_profiler_for_runner", new=_async_noop
        ),
        patch.object(
            PhaseOrchestrator, "_stop_server_profiler_for_runner", new=_async_noop
        ),
    ):
        await orch._execute_phases()

    seamless_runner = created[0]
    seamless_runner.set_phase_error_callback.assert_called_once_with(
        orch._on_seamless_phase_error
    )


async def _async_noop(*_args: object, **_kwargs: object) -> None:
    return None
