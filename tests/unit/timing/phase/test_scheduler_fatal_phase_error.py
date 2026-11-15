# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detached seamless phases retain failures from scheduled root continuations."""

import asyncio
from typing import Any

import pytest

from aiperf.common.environment import Environment
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import ArrivalPattern, DatasetSamplingStrategy
from aiperf.timing.phase.runner import PhaseRunner
from tests.harness.time_traveler import TimeTraveler
from tests.unit.timing.conftest import make_phase_config


@pytest.fixture
async def time_traveler(
    time_traveler_no_patch_sleep: TimeTraveler,
) -> TimeTraveler:
    return time_traveler_no_patch_sleep


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_delayed_root_failure_reaches_seamless_orchestrator(
    create_orchestrator_harness: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe a real scheduler failure after the phase detaches its return wait."""
    monkeypatch.setattr(Environment.TIMING, "HIGH_RES_TIMER", False)
    harness = create_orchestrator_harness(
        [("a", 2), ("b", 2)],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    orchestrator, router = harness.orchestrator, harness.router
    first = make_phase_config(
        request_count=4,
        num_sessions=2,
        request_rate=10,
        arrival_pattern=ArrivalPattern.CONSTANT,
    )
    first = first.model_copy(
        update={"continuation_dispatch": "direct", "phase_index": 0}
    )
    second = make_phase_config(
        request_count=1,
        num_sessions=1,
        request_rate=10,
        arrival_pattern=ArrivalPattern.CONSTANT,
        seamless=True,
    )
    second = second.model_copy(update={"phase_index": 1})
    orchestrator._ordered_phase_configs = [first, second]
    for conversation in orchestrator._dataset_metadata.conversations:
        conversation.turns[1].delay_ms = 25

    blocked = asyncio.Event()
    release_failure = asyncio.Event()
    failure_raised = asyncio.Event()
    error_forwarded = asyncio.Event()
    error = RuntimeError("delayed root continuation transport failed")
    runners: list[PhaseRunner] = []
    scheduler_errors: list[BaseException] = []
    forwarded_errors: list[BaseException] = []
    phase_errors: list[str] = []
    send_credit = router.send_credit
    publish_sending_complete = (
        orchestrator._phase_publisher.publish_phase_sending_complete
    )
    on_seamless_error = orchestrator._on_seamless_phase_error

    def capture_log(message: Any, *args: Any, **kwargs: Any) -> None:
        phase_errors.append(message() if callable(message) else str(message))

    def create_runner(**kwargs: Any) -> PhaseRunner:
        runner = PhaseRunner(**kwargs)
        scheduled_failure = runner._on_scheduled_task_failed

        def observe_scheduler_failure(task: asyncio.Task) -> None:
            assert runner.return_wait_task is not None
            assert not task.cancelled()
            scheduler_errors.append(task.exception())
            scheduled_failure(task)

        runner._scheduler._exception_handler = observe_scheduler_failure
        monkeypatch.setattr(runner, "error", capture_log)
        runners.append(runner)
        return runner

    def observe_seamless_error(fatal: BaseException) -> None:
        forwarded_errors.append(fatal)
        on_seamless_error(fatal)
        error_forwarded.set()

    async def fail_delayed_root(credit: Credit) -> None:
        if credit.phase_index == 1:
            await error_forwarded.wait()
        elif credit.turn_index == 1:
            if credit.conversation_id == "a":
                blocked.set()
                await release_failure.wait()
                failure_raised.set()
                raise error
            await blocked.wait()
        await send_credit(credit)

    async def release_at_sending_complete(stats: Any) -> None:
        await publish_sending_complete(stats)
        if stats.phase_index == 0:
            assert blocked.is_set()
            release_failure.set()
            # The runner resumes before the failed task's done-callback, so
            # the scheduler observes the failure after return-wait detachment.
            await failure_raised.wait()

    monkeypatch.setattr("aiperf.timing.phase_orchestrator.PhaseRunner", create_runner)
    monkeypatch.setattr(
        orchestrator, "_on_seamless_phase_error", observe_seamless_error
    )
    monkeypatch.setattr(orchestrator, "error", capture_log)
    monkeypatch.setattr(router, "send_credit", fail_delayed_root)
    monkeypatch.setattr(
        orchestrator._phase_publisher,
        "publish_phase_sending_complete",
        release_at_sending_complete,
    )
    router.auto_return = True
    execution = asyncio.create_task(orchestrator._execute_phases())
    try:
        with pytest.raises(RuntimeError) as exc_info:
            await asyncio.wait_for(execution, timeout=2.0)
        assert exc_info.value is error
        assert scheduler_errors == [error]
        assert forwarded_errors == [error]
        assert orchestrator._seamless_phase_error is error
        assert len(runners) == 2
        assert runners[0].fatal_phase_error is error
        assert runners[0].return_wait_task.done()
        assert not runners[0].return_wait_task.cancelled()
        assert runners[0] not in orchestrator._active_runners
        assert [
            (credit.conversation_id, credit.turn_index)
            for credit in router.sent_credits
            if credit.phase_index == 0
        ] == [("a", 0), ("b", 0), ("b", 1)]
        assert len(phase_errors) == 2
        assert all("fatal phase failure" in message.lower() for message in phase_errors)
        assert all("control-node" not in message for message in phase_errors)
    finally:
        execution.cancel()
        pending = [execution, *router._pending]
        for runner in runners:
            pending.extend(runner._scheduler.cancel_all())
            pending.extend(
                task
                for task in (
                    runner._execution_task,
                    runner._progress_task,
                    runner.return_wait_task,
                )
                if task is not None
            )
            runner.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
