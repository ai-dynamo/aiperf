# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A profiling phase followed by a seamless phase must still fail the run.

The reachable shape: a multi-profiling-phase config where a non-final profiling
phase transitions seamlessly to its successor AND ``server_profiler`` control
hooks are configured, so the profiler defers its stop to the phase-complete
callback. A fatal request-free control-node failure during that phase's detached
return-wait must fail the run.

Neither side of the origin/main merge covered this combination: main's seamless
tests run without profiler hooks, and the control-hook tests never raise a fatal
error on the deferred path. The gap let an auto-merge silently drop the error
callback on exactly this path while every existing test stayed green.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.control_hooks import PreparedEndpointControlHooks
from aiperf.common.enums import CreditPhase

from ..conftest import make_phase_config

FATAL = RuntimeError("control node died mid-phase")


def _hooks() -> PreparedEndpointControlHooks:
    return PreparedEndpointControlHooks(
        timeout_s=1.0,
        reset_urls=[],
        profiler_start_urls=["http://a:8000/start_profile"],
        profiler_stop_urls=["http://a:8000/stop_profile"],
        profiler_timeout_s=1.0,
    )


def _fake_runner_factory(runners: list[MagicMock], *, fail_first: bool):
    """Build PhaseRunners whose first instance drains with a fatal error.

    Mirrors PhaseRunner._on_return_wait_complete ordering: the error callback
    fires first, then the phase-complete callback (which removes the runner
    from _active_runners).
    """

    def make_runner(**_kwargs: object) -> MagicMock:
        runner = MagicMock()
        runner.return_wait_task = None
        runner.control_fatal_error = None
        is_first = not runners

        async def run(**_run_kwargs: object) -> None:
            if not (is_first and fail_first):
                return
            error_cb = runner.set_phase_error_callback.call_args
            if error_cb is not None:
                error_cb.args[0](FATAL)
            complete_cb = runner.set_phase_complete_callback.call_args
            if complete_cb is not None:
                complete_cb.args[0]()

        runner.run = run
        runners.append(runner)
        return runner

    return make_runner


@pytest.mark.asyncio
async def test_seamless_profiling_fatal_error_fails_run_with_profiler_hooks(
    create_orchestrator_harness,
) -> None:
    """The run must raise, not report success, and must release the profiler."""
    orch = create_orchestrator_harness([("c1", 1)]).orchestrator
    orch._ordered_phase_configs = [
        make_phase_config(phase=CreditPhase.PROFILING),
        make_phase_config(phase=CreditPhase.PROFILING, seamless=True),
    ]
    orch._control_hooks = _hooks()
    orch._control_headers = {}
    orch._credit_router.cancel_all_credits = AsyncMock()

    runners: list[MagicMock] = []
    with (
        patch(
            "aiperf.timing.phase_orchestrator.PhaseRunner",
            side_effect=_fake_runner_factory(runners, fail_first=True),
        ),
        patch(
            "aiperf.timing.phase_orchestrator.start_server_profiler",
            new_callable=AsyncMock,
        ) as start,
        patch(
            "aiperf.timing.phase_orchestrator.stop_server_profiler",
            new_callable=AsyncMock,
        ) as stop,
        pytest.raises(RuntimeError) as exc_info,
    ):
        await orch._execute_phases()

    assert exc_info.value is FATAL
    start.assert_awaited()
    stop.assert_awaited()
    assert not orch._server_profiler_owners


@pytest.mark.asyncio
async def test_seamless_profiling_clean_drain_reports_success_with_profiler_hooks(
    create_orchestrator_harness,
) -> None:
    """The same shape without a fatal error must complete and stop the profiler."""
    orch = create_orchestrator_harness([("c1", 1)]).orchestrator
    orch._ordered_phase_configs = [
        make_phase_config(phase=CreditPhase.PROFILING),
        make_phase_config(phase=CreditPhase.PROFILING, seamless=True),
    ]
    orch._control_hooks = _hooks()
    orch._control_headers = {}

    runners: list[MagicMock] = []
    with (
        patch(
            "aiperf.timing.phase_orchestrator.PhaseRunner",
            side_effect=_fake_runner_factory(runners, fail_first=False),
        ),
        patch(
            "aiperf.timing.phase_orchestrator.start_server_profiler",
            new_callable=AsyncMock,
        ),
        patch(
            "aiperf.timing.phase_orchestrator.stop_server_profiler",
            new_callable=AsyncMock,
        ),
    ):
        await orch._execute_phases()

    assert orch._seamless_phase_error is None
