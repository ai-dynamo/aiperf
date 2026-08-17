# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round 3 runtime hardening: recycle stop-condition naming + unknown-return warning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.agent_graph_replay import AgentGraphReplayStrategy


def _cfg(
    concurrency: int = 1,
    expected_num_sessions: int | None = None,
    total_expected_requests: int | None = None,
    expected_duration_sec: float | None = None,
) -> CreditPhaseConfig:
    """A real profiling-phase config carrying just the stop-condition knobs."""
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENT_GRAPH,
        concurrency=concurrency,
        expected_num_sessions=expected_num_sessions,
        total_expected_requests=total_expected_requests,
        expected_duration_sec=expected_duration_sec,
    )


@dataclass
class _Lifecycle:
    """Stand-in lifecycle whose remaining-time answer acts as a stop condition."""

    _left: float | None = None

    def time_left_in_seconds(self) -> float | None:
        return self._left


def _minimal_strategy(
    config: CreditPhaseConfig, lifecycle: Any = None
) -> AgentGraphReplayStrategy:
    """A strategy over an EMPTY corpus: enough for the gating helpers and return router."""
    # No executor run happens here, so an empty ParsedGraph is sufficient.
    parsed = ParsedGraph(graph=GraphRecord(nodes={}, edges=[], state={}), traces=[])

    class _Issuer:
        async def issue_graph_credit(self, turn: Any) -> bool:
            return True

        def mark_graph_sending_complete(self) -> None: ...
        def graph_all_returned(self) -> bool:
            return True

        def set_graph_all_returned_event(self) -> None: ...
        async def end_graph_trace(self, trace_id: str) -> None: ...

    return AgentGraphReplayStrategy(
        config=config,
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
        lifecycle=lifecycle,
    )


# --- recycle stop-condition naming ------------------------------------------


@pytest.mark.parametrize(
    ("config", "lifecycle", "expected"),
    [
        param(_cfg(), None, False, id="no-caps-no-lifecycle"),
        param(_cfg(expected_num_sessions=5), None, True, id="session-cap"),
        param(_cfg(total_expected_requests=10), None, True, id="request-cap"),
        param(_cfg(expected_duration_sec=30.0), None, True, id="duration-cap"),
        param(_cfg(), _Lifecycle(_left=12.0), True, id="lifecycle-time-remaining"),
        param(_cfg(), _Lifecycle(_left=None), False, id="lifecycle-untimed"),
    ],
)  # fmt: skip
def test_recycle_has_stop_condition_matches_name(
    config: CreditPhaseConfig, lifecycle: Any, expected: bool
) -> None:
    """``_recycle_has_stop_condition`` is True iff a stop condition exists, matching its name."""
    # An inverse-named predicate here would be a double-negation trap for callers.
    strategy = _minimal_strategy(config, lifecycle)
    assert strategy._recycle_has_stop_condition() is expected


@pytest.mark.parametrize(
    ("recycle_is_bounded", "expected_lanes"),
    [
        param(False, 3, id="unbounded-clamps-to-corpus-size"),
        param(True, 8, id="bounded-sustains-full-concurrency"),
    ],
)  # fmt: skip
def test_resolve_lane_count_clamps_only_when_unbounded(
    recycle_is_bounded: bool, expected_lanes: int
) -> None:
    """Without a stop condition the run is a single corpus pass, so lanes clamp to it."""
    strategy = _minimal_strategy(_cfg(concurrency=8))
    assert (
        strategy._resolve_lane_count(3, recycle_is_bounded=recycle_is_bounded)
        == expected_lanes
    )


# --- unknown-return warning ---------------------------------------------------


@dataclass
class _Credit:
    """Minimal Credit-like carrying just the return-routing identity."""

    trace_id: str | None
    node_ordinal: int | None = None
    x_correlation_id: str = "x"
    turn_index: int = 0


def test_unknown_trace_return_logs_warning_with_instance_and_ordinal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A return for an instance with no live adapter WARNs, naming instance id and ordinal."""
    # Logging it at debug would make the dropped return invisible in the field.
    strategy = _minimal_strategy(_cfg())
    with caplog.at_level(logging.WARNING):
        strategy._on_graph_return(
            _Credit(trace_id="t-1#0.0", node_ordinal=7), error=None, cancelled=False
        )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "an unknown-instance return must log at WARNING"
    msg = warnings[-1].getMessage()
    assert "t-1#0.0" in msg
    assert "7" in msg


def test_none_trace_id_return_is_silent_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A credit with no trace_id is not a graph return, so it no-ops without warning."""
    # Only an UNKNOWN graph instance id is anomalous enough to warn.
    strategy = _minimal_strategy(_cfg())
    with caplog.at_level(logging.WARNING):
        strategy._on_graph_return(_Credit(trace_id=None), error=None, cancelled=False)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# --- lane refusal terminates the recycle loop ---------------------------------


@pytest.mark.asyncio
async def test_refused_instance_stops_the_lane_instead_of_recycling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused issue closes the lane; recycling again would only re-refuse.

    Without the ``return`` the ``while True`` spins: every pass increments
    ``_instances_started`` and is refused again by the same closed stop gate.
    Nothing else in the suite drives the lane loop.
    """
    strategy = _minimal_strategy(_cfg(expected_num_sessions=5))
    calls: list[int] = []

    async def _run_instance(trace, lane, recycle_pass, **kwargs):  # noqa: ANN001, ANN003, ANN202
        calls.append(recycle_pass)
        return True  # refused

    monkeypatch.setattr(strategy, "_run_instance", _run_instance)
    monkeypatch.setattr(
        strategy, "_resolve_pass0_lanes", lambda traces, lanes: (["t"], 1)
    )
    monkeypatch.setattr(
        strategy, "_resolve_lane_count", lambda n, recycle_is_bounded: 1
    )
    # A recycle gate that always admits, so only the refusal can stop the lane.
    monkeypatch.setattr(strategy, "_can_recycle", lambda: True)

    await strategy._run_lanes(["t"])

    assert calls == [0]
    assert strategy._instances_started == 1
