# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop graph replay refuses to start on a partially timestamped corpus.

``_wait_for_recorded_start`` silently returns -- no pacing -- for a trace with
no usable recorded start, so a partially timestamped corpus front-loaded its
untimestamped traces as a burst at t=0 on top of an otherwise faithful replay,
with no signal. Linear mode (``FixedScheduleStrategy.setup_phase``) refuses to
start in the equivalent situation; these tests pin the same contract for graph.
"""

from __future__ import annotations

from typing import Any

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import ConfigurationError
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    TraceRecord,
)
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.agent_graph_replay import AgentGraphReplayStrategy
from aiperf.timing.strategies.graph_trace_planner import GraphTracePlanner


class _Issuer:
    async def issue_graph_credit(self, turn: Any) -> bool:
        return True

    def mark_graph_sending_complete(self) -> None: ...
    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...
    async def end_graph_trace(self, trace_id: str) -> None: ...


class _ReversePlanner:
    """Planner stub whose draw order is the REVERSE of corpus order."""

    def __init__(self, parsed: ParsedGraph) -> None:
        self._parsed = parsed

    def plan_for_lane(self, trace: TraceRecord, lane_index: int) -> None:
        return None

    def graph_at_t_star(self, trace: TraceRecord, plan: Any, **kwargs: Any) -> tuple:
        return self._parsed, trace

    def draw_index(self, index: int, total: int) -> int:
        return (total - 1) - (index % total)

    def _draw_is_shuffled(self) -> bool:
        return False

    def _temporal_order(self, traces: list[TraceRecord]) -> list[int]:
        """Reverse corpus order, standing in for the real recorded-start sort.

        The unshuffled bound orders by recorded start rather than by
        ``draw_index``, so the reversal these tests rely on lives here.
        """
        return list(reversed(range(len(traces))))

    select_corpus = GraphTracePlanner.select_corpus


def _parsed(starts: dict[str, int | None]) -> ParsedGraph:
    """Multi-graph corpus: one trace per id, ``None`` start = no timestamp."""
    graphs = {
        trace_id: GraphRecord(
            nodes={
                "n": LlmNode(prompt=["hi"], output="out", recorded_start_unix_ms=start)
            },
            edges=[],
            state={},
        )
        for trace_id, start in starts.items()
    }
    return ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}),
        graphs=graphs,
        traces=[TraceRecord(id=trace_id, graph_ref=trace_id) for trace_id in starts],
    )


def _strategy(
    parsed: ParsedGraph,
    *,
    open_loop: bool = True,
    phase: CreditPhase = CreditPhase.PROFILING,
    **phase_kwargs: Any,
) -> AgentGraphReplayStrategy:
    config = CreditPhaseConfig(
        phase=phase,
        timing_mode=TimingMode.AGENT_GRAPH,
        **phase_kwargs,
    )
    strategy = AgentGraphReplayStrategy(
        config=config,
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
    )
    strategy._open_loop_replay = open_loop
    return strategy


@pytest.mark.asyncio
async def test_fully_timestamped_corpus_sets_up() -> None:
    """Every trace has a recorded start -> setup succeeds."""
    strategy = _strategy(_parsed({"t-1": 1_000, "t-2": 2_000, "t-3": 3_000}))

    await strategy.setup_phase()


@pytest.mark.asyncio
async def test_one_untimestamped_trace_raises_naming_it() -> None:
    """One trace without a recorded start aborts, naming that trace id."""
    strategy = _strategy(_parsed({"t-1": 1_000, "t-2": None, "t-3": 3_000}))

    with pytest.raises(ConfigurationError) as excinfo:
        await strategy.setup_phase()

    assert "t-2" in str(excinfo.value)
    assert "t-1" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_wholly_untimestamped_corpus_sets_up() -> None:
    """A corpus with NO recorded start anywhere is a hand-authored graph.

    ``recorded_start_unix_ms`` is stamped by exactly one producer (the dynamo
    trie lowering); every hand-authored graph (a format not included in this
    release) runs under the same default ``open_loop_replay=True`` with no
    timestamps at all and is paced by its AUTHORED EDGE DELAYS. Those runs
    must keep working.
    """
    strategy = _strategy(_parsed({"t-1": None, "t-2": None}))
    assert strategy._schedule_zero_unix_ms is None

    await strategy.setup_phase()


@pytest.mark.asyncio
async def test_partially_timestamped_corpus_raises() -> None:
    """The primary case: SOME traces timestamped, some not -> refuse to start.

    The untimestamped traces cannot be paced and would burst at t=0 on top of
    the otherwise faithful replay of the timestamped ones.
    """
    strategy = _strategy(_parsed({"t-1": 1_000, "t-2": None, "t-3": None}))
    assert strategy._schedule_zero_unix_ms == 1_000

    with pytest.raises(ConfigurationError) as excinfo:
        await strategy.setup_phase()

    message = str(excinfo.value)
    assert "t-2" in message
    assert "t-3" in message


@pytest.mark.asyncio
async def test_closed_loop_does_not_require_timestamps() -> None:
    """Closed-loop replay never paces on recorded starts, so it must not raise."""
    strategy = _strategy(_parsed({"t-1": None, "t-2": None}), open_loop=False)

    await strategy.setup_phase()


@pytest.mark.asyncio
async def test_warmup_phase_does_not_require_timestamps() -> None:
    """WARMUP skips ``_wait_for_recorded_start`` entirely, so it is exempt."""
    strategy = _strategy(_parsed({"t-1": None, "t-2": None}), phase=CreditPhase.WARMUP)

    await strategy.setup_phase()


@pytest.mark.asyncio
async def test_message_caps_ids_and_reports_true_total() -> None:
    """The id list is capped and the message still carries the TRUE total."""
    starts: dict[str, int | None] = {f"t-{i}": None for i in range(12)}
    starts["anchor"] = 1_000
    strategy = _strategy(_parsed(starts))

    with pytest.raises(ConfigurationError) as excinfo:
        await strategy.setup_phase()

    message = str(excinfo.value)
    named = [f"t-{i}" for i in range(12) if f"t-{i}" in message]
    assert len(named) == 5
    assert "and 7 more (total 12)" in message


@pytest.mark.asyncio
async def test_bounded_selection_excluding_untimestamped_traces_sets_up() -> None:
    """``--num-conversations`` bounding to a timestamped subset must not raise.

    Validation runs against the SELECTED corpus, so traces excluded by the
    bound cannot fail a run that never replays them.
    """
    parsed = _parsed({"t-1": None, "t-2": None, "t-3": 3_000, "t-4": 4_000})
    strategy = _strategy(parsed, expected_num_sessions=2)
    # Reverse draw -> the two LATEST (timestamped) traces are selected.
    strategy._planner = _ReversePlanner(parsed)

    await strategy.setup_phase()


@pytest.mark.asyncio
async def test_setup_selection_is_reused_by_execute_phase(monkeypatch) -> None:
    """The corpus is selected ONCE: ``execute_phase`` reuses setup's selection."""
    from aiperf.timing.strategies import agent_graph_replay as agr

    parsed = _parsed({"t-1": 1_000, "t-2": 2_000, "t-3": 3_000, "t-4": 4_000})
    strategy = _strategy(parsed, expected_num_sessions=2)
    strategy._planner = _ReversePlanner(parsed)

    await strategy.setup_phase()
    assert strategy._selected_traces is not None
    selected_at_setup = [t.id for t in strategy._selected_traces]

    ran: list[str] = []

    class _RecordingExecutor:
        def __init__(self, parsed: ParsedGraph, **kwargs: Any) -> None: ...

        async def run(self, run_trace: Any) -> None:
            ran.append(getattr(run_trace, "id", "?"))

    monkeypatch.setattr(agr, "TraceExecutor", _RecordingExecutor)
    strategy._build_adapter = lambda trace_id, instance_id, **kw: object()  # type: ignore[method-assign]
    strategy._first_token_sources_for = lambda trace: frozenset()  # type: ignore[method-assign]
    strategy._node_identity_for = lambda trace: None  # type: ignore[method-assign]
    strategy._release_adapter_if_idle = lambda instance_id: None  # type: ignore[method-assign]
    strategy._wait_for_recorded_start = _noop  # type: ignore[method-assign]

    await strategy.execute_phase()

    assert sorted(ran) == sorted(selected_at_setup)


async def _noop(*args: Any, **kwargs: Any) -> bool:
    """Stand in for ``_wait_for_recorded_start`` with the wait skipped.

    Returns ``True`` (admitted): the falsy ``None`` this used to return now
    means "admission closed", which would abandon every instance before it
    dispatched.
    """
    return True
