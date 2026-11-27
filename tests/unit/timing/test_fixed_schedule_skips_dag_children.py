# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fixed schedule must not dispatch DAG children itself (AIP-1978).

A subagent's child conversation is dispatched by the BranchOrchestrator when
the parent's SPAWN branch fires. Scheduling it here as well sent every child
request twice, and the duplicates consumed the phase's credit budget -- so the
parent's join turn was later refused by the stop check and every parent turn
after the spawn was silently dropped. The run still exited 0.

``PhaseOrchestrator`` already filters the sampled set on ``is_root``; this
strategy did not. The filter is on ``is_root`` rather than ``agent_depth``
because SPAWN children keep ``agent_depth == 0`` for fresh-context semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.timing.strategies.fixed_schedule import FixedScheduleStrategy


@dataclass
class _Turn:
    timestamp_ms: float


@dataclass
class _Conv:
    conversation_id: str
    turns: list[_Turn]
    is_root: bool = True
    agent_depth: int = 0
    branch_ids: list = field(default_factory=list)


class _Source:
    def __init__(self, conversations):
        self.dataset_metadata = SimpleNamespace(conversations=conversations)

    def session_for_conversation(self, conversation_id, *, x_correlation_id):
        return SimpleNamespace(
            build_first_turn=lambda: SimpleNamespace(conversation_id=conversation_id)
        )


def _strategy(conversations) -> FixedScheduleStrategy:
    strategy = object.__new__(FixedScheduleStrategy)
    AIPerfLoggerMixin.__init__(strategy, logger_name="test-fixed-schedule")
    strategy._conversation_source = _Source(conversations)
    strategy._absolute_schedule = []
    strategy._config = SimpleNamespace(
        phase=None,
        auto_offset_timestamps=False,
        fixed_schedule_start_offset=None,
        fixed_schedule_end_offset=None,
    )
    strategy._schedule_zero_ms = 0.0
    return strategy


def _scheduled_ids(conversations) -> list[str]:
    import asyncio

    strategy = _strategy(conversations)
    asyncio.run(strategy.setup_phase())
    return [e.turn.conversation_id for e in strategy._absolute_schedule]


def test_dag_children_are_not_scheduled() -> None:
    """The child is spawned by the BranchOrchestrator, not scheduled here.

    `_Conv.agent_depth` defaults to 0, so this is also the SPAWN case: those
    children keep `agent_depth == 0` for fresh-context semantics, which is why
    the filter is on `is_root`.
    """
    convs = [
        _Conv("root", [_Turn(0.0)]),
        _Conv("root::sa:agent_001", [_Turn(1500.0)], is_root=False),
    ]
    assert _scheduled_ids(convs) == ["root"]


def test_multiple_roots_are_all_scheduled() -> None:
    convs = [
        _Conv("root_a", [_Turn(0.0)]),
        _Conv("root_b", [_Turn(10.0)]),
        _Conv("root_a::sa:a", [_Turn(5.0)], is_root=False),
    ]
    assert sorted(_scheduled_ids(convs)) == ["root_a", "root_b"]


def test_a_conversation_without_is_root_still_schedules() -> None:
    """Datasets with no DAG concept at all must be unaffected."""
    plain = SimpleNamespace(conversation_id="plain", turns=[_Turn(0.0)])
    assert _scheduled_ids([plain]) == ["plain"]


def test_all_children_raises_naming_the_real_cause() -> None:
    """Failing loudly is right; blaming timestamps would misdirect the reader."""
    convs = [_Conv("root::sa:a", [_Turn(0.0)], is_root=False)]
    with pytest.raises(
        ValueError, match="every conversation in this dataset is a DAG child"
    ):
        _scheduled_ids(convs)
