# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite replay's authored deadlines use captured transport events."""

from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from aiperf.common.enums import CreditPhase, ReplayDependencyEvent
from aiperf.common.models import DatasetMetadata
from aiperf.common.models.dataset_models import (
    ConversationMetadata,
    ReplayTurnReference,
    TurnMetadata,
)
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator


@dataclass
class ManualScheduler:
    scheduled: list[tuple[int, Coroutine[Any, Any, None]]] = field(default_factory=list)

    def schedule_at_perf_ns(
        self, deadline_ns: int, coroutine: Coroutine[Any, Any, None]
    ) -> int:
        self.scheduled.append((deadline_ns, coroutine))
        return len(self.scheduled)

    async def fire(self, deadline_ns: int) -> None:
        index = next(
            i for i, (due, _) in enumerate(self.scheduled) if due == deadline_ns
        )
        _, coroutine = self.scheduled.pop(index)
        await coroutine

    def close(self) -> None:
        for _, coroutine in self.scheduled:
            coroutine.close()
        self.scheduled.clear()


def _reference(
    conversation_id: str,
    *,
    event: ReplayDependencyEvent,
    delay_ms: int,
) -> ReplayTurnReference:
    return ReplayTurnReference(
        conversation_id=conversation_id,
        turn_index=0,
        event=event,
        delay_ns=delay_ms * 1_000_000,
    )


def _metadata() -> DatasetMetadata:
    dispatch = ReplayDependencyEvent.DISPATCH
    completion = ReplayDependencyEvent.COMPLETION
    nodes = (
        ("a0", "a0", 0, ()),
        (
            "a1",
            "a0",
            650,
            (
                _reference("a0", event=completion, delay_ms=30),
                _reference("c1", event=completion, delay_ms=40),
            ),
        ),
        ("a2", "a0", 700, (_reference("a1", event=completion, delay_ms=20),)),
        ("c0", "a0", 50, (_reference("a0", event=dispatch, delay_ms=50),)),
        ("c1", "a0", 400, (_reference("c0", event=completion, delay_ms=50),)),
        ("bg", "a0", 800, (_reference("a1", event=completion, delay_ms=200),)),
        ("b0", "b0", 80, ()),
        (
            "b1",
            "b0",
            750,
            (
                _reference("b0", event=completion, delay_ms=20),
                _reference("d0", event=completion, delay_ms=60),
            ),
        ),
        ("d0", "b0", 100, (_reference("b0", event=dispatch, delay_ms=40),)),
    )
    return DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id=name,
                parent_conversation_id=None if name == root else root,
                agent_depth=0 if name == root else 1,
                is_root=name == root,
                turns=[
                    TurnMetadata(
                        timestamp_ms=timestamp_ms,
                        replay_predecessors=list(dependencies),
                    )
                ],
            )
            for name, root, timestamp_ms, dependencies in nodes
        ],
    )


def _credit(name: str, root: str, credit_id: int) -> Credit:
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id=name,
        x_correlation_id=name,
        root_correlation_id=root,
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
        agent_depth=0 if name == root else 1,
        finite_replay=True,
    )


@pytest.mark.asyncio
async def test_nine_request_graph_uses_latest_captured_deadline() -> None:
    scheduler = ManualScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(), strict_finite=True, scheduler=scheduler
    )
    coordinator.activate_finite()
    roots = {
        name: "b0" if name in {"b0", "b1", "d0"} else "a0"
        for name in ("a0", "a1", "a2", "c0", "c1", "bg", "b0", "b1", "d0")
    }
    credits = {
        name: _credit(name, roots[name], index) for index, name in enumerate(roots)
    }
    issued: list[str] = []

    async def submit(name: str) -> None:
        credit = credits[name]
        turn = TurnToSend(
            conversation_id=name,
            x_correlation_id=name,
            root_correlation_id=roots[name],
            turn_index=0,
            num_turns=1,
            agent_depth=credit.agent_depth,
        )

        async def issue() -> bool:
            issued.append(name)
            return True

        await coordinator.submit(turn, issue)

    try:
        for name in roots:
            await submit(name)
        assert issued == ["a0", "b0"]
        a_start, b_start = 1_000_000_000, 2_000_000_000
        coordinator.record_dispatch(credits["a0"], a_start, 0)
        coordinator.record_dispatch(credits["b0"], b_start, 0)
        assert [deadline for deadline, _ in scheduler.scheduled] == [
            a_start + 50_000_000,
            b_start + 40_000_000,
        ]

        async def dispatch(name: str, start_ns: int) -> None:
            await scheduler.fire(start_ns)
            coordinator.record_dispatch(credits[name], start_ns, 0)

        def complete(name: str, eof_ns: int) -> None:
            coordinator.record_completion(
                credits[name], eof_ns, clock_spread_ns=0, failed=False
            )

        await dispatch("c0", a_start + 50_000_000)
        await dispatch("d0", b_start + 40_000_000)
        complete("a0", a_start + 200_000_000)
        complete("c0", a_start + 200_000_000)
        complete("b0", b_start + 250_000_000)
        await dispatch("c1", a_start + 400_000_000)
        complete("c1", a_start + 700_000_000)
        complete("d0", b_start + 640_000_000)
        await dispatch("a1", a_start + 740_000_000)
        await dispatch("b1", b_start + 700_000_000)
        complete("a1", a_start + 840_000_000)
        await dispatch("a2", a_start + 860_000_000)
        await dispatch("bg", a_start + 1_040_000_000)
        complete("a2", a_start + 900_000_000)
        complete("bg", a_start + 1_120_000_000)
        complete("b1", b_start + 750_000_000)
        assert sorted(issued) == sorted(roots)
        assert not coordinator.has_pending_finite_work()
    finally:
        scheduler.close()


@pytest.mark.asyncio
async def test_early_timer_rechecks_absolute_deadline() -> None:
    scheduler = ManualScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(), strict_finite=True, scheduler=scheduler
    )
    coordinator.activate_finite()
    issued: list[str] = []
    root = _credit("a0", "a0", 1)
    child = TurnToSend(
        conversation_id="c0",
        x_correlation_id="c0",
        root_correlation_id="a0",
        turn_index=0,
        num_turns=1,
        agent_depth=1,
    )

    async def issue() -> bool:
        issued.append("c0")
        return True

    try:
        await coordinator.submit(child, issue)
        coordinator.record_dispatch(root, 1_000_000_000, 0)
        due = 1_050_000_000
        with patch(
            "aiperf.timing.replay_dependencies.time.perf_counter_ns",
            return_value=due - 1,
        ):
            await scheduler.fire(due)
        assert issued == []
        assert scheduler.scheduled[0][0] == due
        with patch(
            "aiperf.timing.replay_dependencies.time.perf_counter_ns", return_value=due
        ):
            await scheduler.fire(due)
        assert issued == ["c0"]
    finally:
        scheduler.close()


@pytest.mark.parametrize(
    "eof_ns,failed,expected",
    [
        (None, False, "EOF missing"),
        (None, True, "failed"),
        (1_000_000_001, True, "failed"),
    ],
)  # fmt: skip
def test_terminal_failure_is_fatal_even_without_dependents(
    eof_ns: int | None, failed: bool, expected: str
) -> None:
    scheduler = ManualScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(), strict_finite=True, scheduler=scheduler
    )
    leaf = _credit("bg", "a0", 1)
    coordinator.record_dispatch(leaf, 1_000_000_000, 0)
    with pytest.raises(RuntimeError, match=expected):
        coordinator.record_completion(leaf, eof_ns, clock_spread_ns=0, failed=failed)


def test_duplicate_and_closed_root_events_cannot_change_first_observation() -> None:
    scheduler = ManualScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(), strict_finite=True, scheduler=scheduler
    )
    root = _credit("a0", "a0", 1)
    coordinator.record_dispatch(root, 1_000_000_000, 0)
    coordinator.record_dispatch(root, 1_050_000_000, 0)
    assert coordinator._roots["a0"].root_start_perf_ns == 1_000_000_000
    coordinator.record_completion(root, 1_100_000_000, clock_spread_ns=0, failed=False)
    coordinator.record_completion(root, 1_200_000_000, clock_spread_ns=0, failed=False)
    assert (
        next(iter(coordinator._roots["a0"].completion_events.values())) == 1_100_000_000
    )
    coordinator.close_root("a0")
    coordinator.record_dispatch(root, 2_000_000_000, 0)
    coordinator.record_completion(root, 2_100_000_000, clock_spread_ns=0, failed=False)
    assert "a0" not in coordinator._roots


@pytest.mark.asyncio
async def test_late_eof_notification_keeps_authored_absolute_deadline() -> None:
    scheduler = ManualScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(), strict_finite=True, scheduler=scheduler
    )
    coordinator.activate_finite()
    root = _credit("a0", "a0", 1)
    parent = _credit("a1", "a0", 2)
    child = TurnToSend(
        conversation_id="a2",
        x_correlation_id="a2",
        root_correlation_id="a0",
        turn_index=0,
        num_turns=1,
        agent_depth=1,
    )
    issued: list[str] = []

    async def issue() -> bool:
        issued.append("a2")
        return True

    try:
        await coordinator.submit(child, issue)
        coordinator.record_dispatch(root, 1_000_000_000, 0)
        coordinator.record_dispatch(parent, 1_650_000_000, 0)
        with patch(
            "aiperf.timing.replay_dependencies.time.perf_counter_ns",
            return_value=1_900_000_000,
        ):
            coordinator.record_completion(
                parent, 1_750_000_000, clock_spread_ns=0, failed=False
            )
            assert scheduler.scheduled[0][0] == 1_770_000_000
            await scheduler.fire(1_770_000_000)
        assert issued == ["a2"]
    finally:
        scheduler.close()


@pytest.mark.parametrize("timestamp,spread", [(0, 0), (-1, 0), (1, -1), ("bad", 0)])  # fmt: skip
def test_invalid_transport_start_is_rejected(timestamp: object, spread: int) -> None:
    scheduler = ManualScheduler()
    coordinator = ReplayBarrierCoordinator(
        _metadata(), strict_finite=True, scheduler=scheduler
    )
    with pytest.raises((RuntimeError, ValueError), match="transport start"):
        coordinator.record_dispatch(_credit("a0", "a0", 1), timestamp, spread)


@pytest.mark.parametrize(
    "eof_ns,spread",
    [
        (0, 0),
        (-1, 0),
        (1_000_000_001, -1),
        ("bad", 0),
        (1_000_000_001, None),
    ],
)  # fmt: skip
def test_invalid_terminal_boundary_is_rejected(
    eof_ns: object, spread: int | None
) -> None:
    coordinator = ReplayBarrierCoordinator(
        _metadata(), strict_finite=True, scheduler=ManualScheduler()
    )
    root = _credit("a0", "a0", 1)
    coordinator.record_dispatch(root, 1_000_000_000, 0)
    with pytest.raises(RuntimeError, match="response EOF"):
        coordinator.record_completion(
            root, eof_ns, clock_spread_ns=spread, failed=False
        )
