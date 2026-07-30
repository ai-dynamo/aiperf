# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest

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
    coordinator = ReplayBarrierCoordinator(
        _metadata(),
        scheduler=scheduler,
        root_idle_gap_cap_seconds=0.05,
    )
    coordinator.activate()

    coordinator.observe_issued(_credit("a"))
    coordinator.complete(_credit("a"))
    coordinator.observe_issued(_credit("b"))
    await asyncio.sleep(0.06)

    scheduler.cap_pending_delay_for_group.assert_not_called()


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
