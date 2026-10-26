# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    ReplayTurnReference,
    TurnMetadata,
)
from aiperf.credit.dispatch import ChildDispatchResult
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing import replay_dependencies
from aiperf.timing.replay_dependencies import (
    ReplayBarrierCoordinator,
    ReplayResumeBoundary,
)


def _metadata() -> DatasetMetadata:
    abc = [ReplayTurnReference(conversation_id=name, turn_index=0) for name in "abc"]
    return DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
        conversations=[
            ConversationMetadata(
                conversation_id=name, turns=[TurnMetadata(timestamp_ms=0)]
            )
            for name in "abc"
        ]
        + [
            ConversationMetadata(
                conversation_id="d",
                turns=[TurnMetadata(timestamp_ms=10, replay_predecessors=abc)],
            )
        ],
    )


def _turn(
    name: str,
    root: str = "root",
    turn_index: int = 0,
    num_turns: int = 1,
) -> TurnToSend:
    return TurnToSend(
        conversation_id=name,
        x_correlation_id=f"{root}:{name}",
        turn_index=turn_index,
        num_turns=num_turns,
        root_correlation_id=root,
    )


def _credit(
    name: str,
    root: str = "root",
    turn_index: int = 0,
    num_turns: int = 1,
) -> Credit:
    return Credit(
        id=0,
        phase=CreditPhase.PROFILING,
        conversation_id=name,
        x_correlation_id=f"{root}:{name}",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        root_correlation_id=root,
    )


@pytest.mark.asyncio
async def test_abc_issue_together_and_d_waits_for_every_member() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []

    async def submit(name: str) -> None:
        await coordinator.submit(
            _turn(name),
            lambda name=name: _record_issue(issued, name),
        )

    await submit("a")
    await submit("b")
    await submit("c")
    await submit("d")
    assert issued == ["a", "b", "c"]

    coordinator.complete(_credit("a"))
    await asyncio.sleep(0)
    assert issued == ["a", "b", "c"]
    coordinator.complete(_credit("b"))
    await asyncio.sleep(0)
    assert issued == ["a", "b", "c"]
    coordinator.complete(_credit("c"))
    await asyncio.sleep(0)
    assert issued == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_any_terminal_outcome_releases_barrier() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    for name in "abcd":
        await coordinator.submit(
            _turn(name), lambda name=name: _record_issue(issued, name)
        )

    for name in "abc":
        coordinator.complete(_credit(name))
    await asyncio.sleep(0)

    assert issued[-1] == "d"


@pytest.mark.asyncio
async def test_runtime_roots_are_independent() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    await coordinator.submit(_turn("d", "one"), lambda: _record_issue(issued, "one:d"))
    await coordinator.submit(_turn("d", "two"), lambda: _record_issue(issued, "two:d"))

    for name in "abc":
        coordinator.complete(_credit(name, "one"))
    await asyncio.sleep(0)

    assert issued == ["one:d"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cap_seconds", [0.0, 0.01])
async def test_idle_watchdog_advances_only_the_completed_runtime_root(
    cap_seconds: float,
) -> None:
    """A fully idle tree advances its own timers without touching another tree."""
    scheduler = MagicMock()
    scheduler.running_count_for_group.return_value = 0
    advanced = asyncio.Event()

    def advance(*_args) -> float:
        advanced.set()
        return 4.5

    scheduler.cap_pending_delay_for_group.side_effect = advance
    coordinator = ReplayBarrierCoordinator(
        _metadata(),
        scheduler=scheduler,
        root_idle_gap_cap_seconds=cap_seconds,
    )
    coordinator.activate()

    coordinator.observe_issued(_credit("a", "one"))
    coordinator.observe_issued(_credit("a", "two"))
    coordinator.complete(_credit("a", "one"))
    await asyncio.wait_for(advanced.wait(), timeout=0.2)

    scheduler.cap_pending_delay_for_group.assert_called_once_with("one", 0.0)


@pytest.mark.asyncio
async def test_idle_watchdog_covers_initial_profiling_idle() -> None:
    """A root with only future timers is idle before its first request."""
    scheduler = MagicMock()
    scheduler.running_count_for_group.return_value = 0
    advanced = asyncio.Event()

    def advance(*_args) -> float:
        advanced.set()
        return 4.5

    scheduler.cap_pending_delay_for_group.side_effect = advance
    coordinator = ReplayBarrierCoordinator(
        _metadata(),
        scheduler=scheduler,
        root_idle_gap_cap_seconds=0.01,
    )
    coordinator.activate()

    coordinator.observe_idle_root("one")
    await asyncio.wait_for(advanced.wait(), timeout=0.2)

    scheduler.cap_pending_delay_for_group.assert_called_once_with("one", 0.0)


@pytest.mark.asyncio
async def test_initial_idle_watchdog_advances_real_group_timer() -> None:
    scheduler = LoopScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(),
        scheduler=scheduler,
        root_idle_gap_cap_seconds=0.02,
    )
    coordinator.activate()
    fired = asyncio.Event()

    async def mark_fired() -> None:
        fired.set()

    scheduler.schedule_later(0.2, mark_fired(), group_id="one")
    started = asyncio.get_running_loop().time()
    coordinator.observe_idle_root("one")
    await asyncio.wait_for(fired.wait(), timeout=0.1)

    assert asyncio.get_running_loop().time() - started < 0.08


@pytest.mark.asyncio
async def test_new_request_cancels_runtime_root_idle_watchdog() -> None:
    """Overlapping work prevents an idle cap from advancing replay timers."""
    scheduler = MagicMock()
    scheduler.running_count_for_group.return_value = 0
    coordinator = ReplayBarrierCoordinator(
        _metadata(),
        scheduler=scheduler,
        root_idle_gap_cap_seconds=0.05,
    )
    coordinator.activate()

    coordinator.observe_issued(_credit("a"))
    coordinator.complete(_credit("a"))
    handle = coordinator._roots["root"].idle_watchdog
    assert handle is not None
    coordinator.observe_issued(_credit("b"))

    assert handle.cancelled()
    await asyncio.sleep(0.06)

    scheduler.cap_pending_delay_for_group.assert_not_called()


@pytest.mark.asyncio
async def test_late_completion_does_not_recreate_closed_root() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    coordinator.observe_issued(_credit("a"))
    coordinator.close_root("root")

    coordinator.complete(_credit("a"))

    assert "root" not in coordinator._roots


@pytest.mark.asyncio
async def test_idle_watchdog_retries_when_advanced_turn_is_barrier_blocked() -> None:
    """A blocked candidate cannot consume the cap and strand later timers."""
    scheduler = LoopScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(),
        scheduler=scheduler,
        root_idle_gap_cap_seconds=0.02,
    )
    coordinator.activate()
    issued = asyncio.Event()

    async def submit_blocked() -> None:
        await coordinator.submit(_turn("d"), lambda: _record_issue([], "d"))

    async def submit_ready() -> None:
        async def mark_issued() -> bool:
            issued.set()
            return True

        await coordinator.submit(
            _turn("a"),
            mark_issued,
        )

    # The first cap advance lands on d, which is retained behind a/b/c.  The
    # watchdog must immediately retry and advance ready turn a as part of the
    # same already-expired idle interval.
    scheduler.schedule_later(1.0, submit_blocked(), group_id="root")
    scheduler.schedule_later(2.0, submit_ready(), group_id="root")
    started = asyncio.get_running_loop().time()
    coordinator.observe_idle_root("root")

    await asyncio.wait_for(issued.wait(), timeout=0.15)

    assert asyncio.get_running_loop().time() - started < 0.10
    await coordinator.cancel_pending(notify_refused=False)
    scheduler.cancel_all()


@pytest.mark.asyncio
async def test_scalar_peak_would_slip_d_after_only_one_completion() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    for name in "abcd":
        await coordinator.submit(
            _turn(name), lambda name=name: _record_issue(issued, name)
        )

    coordinator.complete(_credit("a"))
    await asyncio.sleep(0)

    assert "d" not in issued


@pytest.mark.asyncio
async def test_pending_turns_exposes_deferred_dispatch_for_phase_handoff() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    pending_turn = _turn("d")

    await coordinator.submit(
        pending_turn,
        lambda: _record_issue(issued, "d"),
    )

    assert issued == []
    assert coordinator.pending_turns("root") == (pending_turn,)
    assert coordinator.pending_turns_by_root() == {"root": (pending_turn,)}

    for name in "abc":
        coordinator.complete(_credit(name))
    await asyncio.sleep(0)

    assert issued == ["d"]
    assert coordinator.pending_turns("root") == ()
    assert coordinator.pending_turns_by_root() == {}


@pytest.mark.asyncio
async def test_retained_child_dispatch_reports_deferred_not_rejected() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()

    result = await coordinator.submit(
        _turn("d"),
        lambda: _record_issue([], "d"),
        retained_result=ChildDispatchResult.DEFERRED,
    )

    assert result is ChildDispatchResult.DEFERRED
    assert coordinator.pending_turns("root") == (_turn("d"),)


@pytest.mark.asyncio
async def test_release_time_deferral_is_expected_not_refused_or_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(replay_dependencies, "_logger", logger)
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issue_calls: list[str] = []
    refused: list[str] = []

    async def issue() -> ChildDispatchResult:
        issue_calls.append("d")
        return ChildDispatchResult.DEFERRED

    async def on_refused() -> None:
        refused.append("d")

    result = await coordinator.submit(
        _turn("d"),
        issue,
        on_refused=on_refused,
        retained_result=ChildDispatchResult.DEFERRED,
    )
    assert result is ChildDispatchResult.DEFERRED
    assert issue_calls == []

    for name in "abc":
        coordinator.complete(_credit(name))
    await asyncio.sleep(0)

    assert issue_calls == ["d"]
    assert refused == []
    assert coordinator.pending_turns_by_root() == {}
    logger.error.assert_not_called()
    logger.exception.assert_not_called()
    logger.debug.assert_called_once()


@pytest.mark.asyncio
async def test_paused_releases_retain_ready_pending_for_phase_handoff() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    pending_turn = _turn("d")

    await coordinator.submit(
        pending_turn,
        lambda: _record_issue(issued, "d"),
    )
    coordinator.pause_releases()

    for name in "abc":
        coordinator.complete(_credit(name))
    await asyncio.sleep(0)

    assert issued == []
    assert coordinator.pending_turns("root") == (pending_turn,)
    assert coordinator.pending_turns_by_root() == {"root": (pending_turn,)}


@pytest.mark.asyncio
async def test_paused_releases_retain_new_ready_submissions() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    pending_turn = _turn("d")

    for name in "abc":
        coordinator.complete(_credit(name))
    coordinator.pause_releases()
    await coordinator.submit(
        pending_turn,
        lambda: _record_issue(issued, "d"),
    )
    await asyncio.sleep(0)

    assert issued == []
    assert coordinator.pending_turns("root") == (pending_turn,)
    assert coordinator.pending_turns_by_root() == {"root": (pending_turn,)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "submission_order",
    [
        (("aux", 0, 1), ("flat", 1, 2)),
        (("flat", 1, 2), ("aux", 0, 1)),
    ],
)
async def test_resumed_prefix_is_exact_and_submission_order_independent(
    submission_order: tuple[tuple[str, int, int], ...],
) -> None:
    metadata = DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id="flat",
                turns=[
                    TurnMetadata(timestamp_ms=0),
                    TurnMetadata(
                        timestamp_ms=20,
                        replay_predecessors=[
                            ReplayTurnReference(conversation_id="aux", turn_index=0)
                        ],
                    ),
                ],
            ),
            ConversationMetadata(
                conversation_id="aux",
                turns=[
                    TurnMetadata(
                        timestamp_ms=10,
                        replay_predecessors=[
                            ReplayTurnReference(conversation_id="flat", turn_index=0)
                        ],
                    )
                ],
                is_root=False,
            ),
        ],
    )
    coordinator = ReplayBarrierCoordinator(metadata)
    coordinator.seed_completed_prefixes("root", (ReplayResumeBoundary("flat", 1),))
    coordinator.activate()
    issued: list[tuple[str, int]] = []

    async def record_issue(item: tuple[str, int]) -> bool:
        issued.append(item)
        return True

    for conversation_id, turn_index, num_turns in submission_order:
        await coordinator.submit(
            _turn(
                conversation_id,
                turn_index=turn_index,
                num_turns=num_turns,
            ),
            lambda conversation_id=conversation_id, turn_index=turn_index: record_issue(
                (conversation_id, turn_index)
            ),
        )

    assert issued == [("aux", 0)]

    coordinator.complete(_credit("aux"))
    await asyncio.sleep(0)

    assert issued == [("aux", 0), ("flat", 1)]
    assert coordinator.completed_prefixes("root") == (
        ReplayResumeBoundary("aux", 1),
        ReplayResumeBoundary("flat", 1),
    )


async def _record_issue(issued: list[str], name: str) -> bool:
    issued.append(name)
    return True


def _watchdog_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Idle watchdog root=")
    ]


def _watchdog_coordinator(
    scheduler: LoopScheduler | MagicMock,
) -> ReplayBarrierCoordinator:
    if isinstance(scheduler, MagicMock):
        scheduler.running_count_for_group.return_value = 0
    coordinator = ReplayBarrierCoordinator(
        _metadata(), scheduler=scheduler, root_idle_gap_cap_seconds=0.05
    )
    coordinator.activate()
    return coordinator


@pytest.mark.asyncio
async def test_idle_watchdog_debug_logs_arm_and_cancel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A CI gap failure needs the watchdog's arm/cancel history per root."""
    caplog.set_level(logging.DEBUG, logger=replay_dependencies.__name__)
    coordinator = _watchdog_coordinator(MagicMock())

    coordinator.observe_issued(_credit("a"))
    coordinator.complete(_credit("a"))
    coordinator.observe_issued(_credit("b"))

    assert _watchdog_lines(caplog) == [
        "Idle watchdog root=root: armed for 0.050s, in_flight=0",
        "Idle watchdog root=root: cancelled by b#0 issue, in_flight=1",
    ]


@pytest.mark.asyncio
async def test_idle_watchdog_debug_logs_completion_with_requests_in_flight(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stuck in-flight count is visible instead of silently skipping the cap."""
    caplog.set_level(logging.DEBUG, logger=replay_dependencies.__name__)
    coordinator = _watchdog_coordinator(MagicMock())

    coordinator.observe_issued(_credit("a"))
    coordinator.observe_issued(_credit("b"))
    coordinator.complete(_credit("a"))

    assert _watchdog_lines(caplog) == [
        "Idle watchdog root=root: not armed after a#0 completed, in_flight=1",
    ]
    assert coordinator._roots["root"].idle_watchdog is None


@pytest.mark.asyncio
async def test_idle_watchdog_debug_logs_already_armed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=replay_dependencies.__name__)
    coordinator = _watchdog_coordinator(MagicMock())

    coordinator.observe_idle_root("root")
    coordinator.observe_idle_root("root")

    assert _watchdog_lines(caplog) == [
        "Idle watchdog root=root: armed for 0.050s, in_flight=0",
        "Idle watchdog root=root: already armed, not re-armed",
    ]
    await coordinator.cancel_pending(notify_refused=False)


@pytest.mark.asyncio
async def test_idle_watchdog_debug_logs_fire_with_requests_in_flight(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=replay_dependencies.__name__)
    scheduler = MagicMock()
    coordinator = _watchdog_coordinator(scheduler)
    coordinator.observe_idle_root("root")
    coordinator._roots["root"].in_flight = 1

    coordinator._enforce_root_idle_cap("root")

    assert _watchdog_lines(caplog)[-1] == (
        "Idle watchdog root=root: fired with in_flight=1, nothing advanced"
    )
    scheduler.cap_pending_delay_for_group.assert_not_called()
    await coordinator.cancel_pending(notify_refused=False)


@pytest.mark.asyncio
async def test_idle_watchdog_debug_logs_fire_that_advances_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An expiry with nothing to advance still marks the cap expired.

    ``idle_cap_expired`` is what lets a later grouped timer re-enforce the cap
    (``_on_root_timer_done``), and ``pending_timers=0`` in the log separates
    "no timers left" from the in-flight case when diagnosing a CI gap failure.
    """
    caplog.set_level(logging.DEBUG, logger=replay_dependencies.__name__)
    scheduler = LoopScheduler()
    coordinator = _watchdog_coordinator(scheduler)
    coordinator.observe_idle_root("root")

    coordinator._enforce_root_idle_cap("root")

    assert _watchdog_lines(caplog)[-1] == (
        "Idle watchdog root=root: fired but advanced nothing, pending_timers=0"
    )
    assert coordinator._roots["root"].idle_cap_expired is True
    await coordinator.cancel_pending(notify_refused=False)


async def _control_timer() -> None:
    """A grouped timer that issues no request, like an unsatisfied join deadline."""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control_delay_sec",
    [
        param(-0.001, id="control-timer-already-due"),
        param(0.001, id="control-timer-due-1ms-later"),
    ],
)  # fmt: skip
async def test_idle_cap_advances_request_past_control_timer_that_issues_nothing(
    control_delay_sec: float,
) -> None:
    """A due control timer must not absorb an expired idle cap.

    When the watchdog fires, the earliest group timer is a control timer due
    now (or ~1 ms later), so the uniform shift is ~0 and the real next
    request stays seconds away. Once the control timer runs without issuing,
    the cap must still pull the request in.
    """
    scheduler = LoopScheduler()
    coordinator = _watchdog_coordinator(scheduler)
    loop = asyncio.get_running_loop()
    request_fired = asyncio.Event()

    async def request_timer() -> None:
        request_fired.set()

    scheduler.schedule_at(
        loop.time() + control_delay_sec, _control_timer(), group_id="root"
    )
    scheduler.schedule_later(5.0, request_timer(), group_id="root")
    coordinator._root_state("root")

    coordinator._enforce_root_idle_cap("root")
    await asyncio.wait_for(request_fired.wait(), timeout=0.5)

    assert scheduler.pending_count_for_group("root") == 0
    scheduler.cancel_all()


@pytest.mark.asyncio
async def test_idle_cap_does_not_advance_past_timer_that_issued_a_request() -> None:
    """A timer that issues a request ends the idle gap, so the cap stops.

    ``observe_issued`` clears ``idle_cap_expired``; when the issuing timer
    returns, the group observer must leave the next request at its original
    trace offset rather than pulling it in behind work already on the wire.
    """
    scheduler = LoopScheduler()
    coordinator = _watchdog_coordinator(scheduler)
    loop = asyncio.get_running_loop()
    issued = asyncio.Event()

    async def issuing_timer() -> None:
        coordinator.observe_issued(_credit("a"))
        issued.set()

    async def later_request() -> None:
        pass

    scheduler.schedule_at(loop.time() - 0.001, issuing_timer(), group_id="root")
    later_handle = scheduler.schedule_later(5.0, later_request(), group_id="root")
    later_when = scheduler._handles[later_handle][0].when()
    coordinator._root_state("root")

    coordinator._enforce_root_idle_cap("root")
    await asyncio.wait_for(issued.wait(), timeout=0.5)
    await asyncio.sleep(0)

    assert scheduler.pending_count_for_group("root") == 1
    [(handle, _)] = scheduler._handles.values()
    assert handle.when() == pytest.approx(later_when)
    scheduler.cancel_all()


@pytest.mark.asyncio
async def test_timer_issuing_nothing_before_cap_expires_keeps_replay_spacing() -> None:
    """Before the cap expires, a timer that issues nothing must not move others.

    The group observer runs after every grouped timer, but re-enforcement is
    gated on ``idle_cap_expired``. Without that gate, every join deadline in a
    normal trace would compress the spacing to the following request.
    """
    scheduler = LoopScheduler()
    coordinator = _watchdog_coordinator(scheduler)
    loop = asyncio.get_running_loop()
    control_done = asyncio.Event()

    async def control_timer() -> None:
        control_done.set()

    async def later_request() -> None:
        pass

    scheduler.schedule_at(loop.time() - 0.001, control_timer(), group_id="root")
    later_handle = scheduler.schedule_later(5.0, later_request(), group_id="root")
    later_when = scheduler._handles[later_handle][0].when()
    coordinator._root_state("root")

    await asyncio.wait_for(control_done.wait(), timeout=0.5)
    await asyncio.sleep(0)

    [(handle, _)] = scheduler._handles.values()
    assert handle.when() == pytest.approx(later_when)
    scheduler.cancel_all()


def _started_unissued_metadata() -> DatasetMetadata:
    """``_metadata()`` plus an independent request ``e`` on the same tree."""
    metadata = _metadata()
    metadata.conversations.append(
        ConversationMetadata(conversation_id="e", turns=[TurnMetadata(timestamp_ms=0)])
    )
    return metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("second_timer", ["barrier_blocked", "issues_nothing"])
async def test_idle_cap_waits_for_started_timer_that_has_not_issued(
    second_timer: str,
) -> None:
    """A started timer still on its way to the wire keeps the tree busy.

    Timer ``a`` has fired and is waiting for capacity (a prefill slot in
    production) before ``observe_issued``, so ``in_flight`` is still zero.
    A second timer then either lands on a barrier-blocked turn (the
    zero-delay rearm in ``submit``) or returns without issuing (the
    group-task observer). Neither may advance ``e``: it must keep its
    recorded 0.4 s spacing after ``a``.
    """
    loop = asyncio.get_running_loop()
    scheduler = LoopScheduler()
    coordinator = ReplayBarrierCoordinator(
        _started_unissued_metadata(),
        scheduler=scheduler,
        root_idle_gap_cap_seconds=0.02,
    )
    coordinator.activate()
    capacity = asyncio.Event()
    times: dict[str, float] = {}
    e_fired = asyncio.Event()

    async def timer_a() -> None:
        times["a"] = loop.time()

        async def issue() -> bool:
            await capacity.wait()
            coordinator.observe_issued(_credit("a"))
            return True

        await coordinator.submit(_turn("a"), issue)

    async def timer_second() -> None:
        if second_timer == "barrier_blocked":
            await coordinator.submit(_turn("d"), lambda: _record_issue([], "d"))

    async def timer_e() -> None:
        times["e"] = loop.time()
        e_fired.set()

    scheduler.schedule_later(1.0, timer_a(), group_id="root")
    scheduler.schedule_later(1.05, timer_second(), group_id="root")
    scheduler.schedule_later(1.4, timer_e(), group_id="root")
    coordinator.observe_idle_root("root")
    try:
        await asyncio.wait_for(e_fired.wait(), timeout=1.0)
    finally:
        capacity.set()
        await asyncio.sleep(0)
        await coordinator.cancel_pending(notify_refused=False)
        scheduler.cancel_all()

    assert times["e"] - times["a"] >= 0.35


@pytest.mark.asyncio
async def test_idle_cap_resumes_when_started_timer_returns_without_issuing() -> None:
    """Deferring for a running timer must not strand the expired cap.

    Once the running timer returns without issuing, the group-task observer
    re-enforces the cap, so the far-off next timer is still advanced.
    """
    scheduler = LoopScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(), scheduler=scheduler, root_idle_gap_cap_seconds=0.02
    )
    coordinator.activate()
    release = asyncio.Event()
    fired = asyncio.Event()

    async def slow_noop() -> None:
        await release.wait()

    async def far_timer() -> None:
        fired.set()

    scheduler.schedule_later(1.0, slow_noop(), group_id="root")
    scheduler.schedule_later(5.0, far_timer(), group_id="root")
    coordinator.observe_idle_root("root")
    loop = asyncio.get_running_loop()
    while scheduler.running_count_for_group("root") == 0:
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.05)  # watchdog retries are deferred while it runs
    assert not fired.is_set()

    started = loop.time()
    release.set()
    await asyncio.wait_for(fired.wait(), timeout=0.5)
    assert loop.time() - started < 0.2
    scheduler.cancel_all()
