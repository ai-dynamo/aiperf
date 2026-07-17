# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-hardening unit checks: recycle stop-condition naming + unknown-return warning.

* ``_recycle_has_stop_condition`` returns True iff a stop condition exists,
  and the ``recycle_is_bounded`` var matches its meaning. We pin the boolean
  contract so a naming change stays behavior-neutral.
* ``_on_graph_return`` reports a dropped unknown-trace-id return as a
  WARNING carrying the instance id + node ordinal (not a debug no-op), so an
  orphaned-adapter condition is field-diagnosable. We pin the level + payload.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class _Cfg:
    phase: Any = None
    concurrency: int = 1
    expected_num_sessions: int | None = None
    total_expected_requests: int | None = None
    expected_duration_sec: float | None = None


@dataclass
class _Lifecycle:
    _left: float | None = None

    def time_left_in_seconds(self) -> float | None:
        return self._left


def _minimal_strategy(config: _Cfg, lifecycle: Any = None):
    """A strategy built over an EMPTY corpus -- enough to exercise the gating
    helpers and the return router without any executor run."""
    from aiperf.dataset.graph.models import GraphRecord, ParsedGraph
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    parsed = ParsedGraph(graph=GraphRecord(nodes={}, edges=[], state={}), traces=[])

    class _Issuer:
        async def issue_graph_credit(self, turn: Any) -> bool:
            return True

        def mark_graph_sending_complete(self) -> None: ...
        def graph_all_returned(self) -> bool:
            return True

        def set_graph_all_returned_event(self) -> None: ...

    return GraphIRReplayStrategy(
        config=config,
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        lifecycle=lifecycle,
    )


# --- recycle stop-condition naming ------------------------------------------


@pytest.mark.parametrize(
    "config,lifecycle,expected",
    [
        (_Cfg(), None, False),
        (_Cfg(expected_num_sessions=5), None, True),
        (_Cfg(total_expected_requests=10), None, True),
        (_Cfg(expected_duration_sec=30.0), None, True),
        (_Cfg(), _Lifecycle(_left=12.0), True),
        (_Cfg(), _Lifecycle(_left=None), False),
    ],
)  # fmt: skip
def test_recycle_has_stop_condition_matches_name(config, lifecycle, expected):
    """``_recycle_has_stop_condition`` returns True iff a stop condition EXISTS,
    so the name matches the meaning (an inverse-named predicate here is a
    double-negation trap for callers)."""
    strategy = _minimal_strategy(config, lifecycle)
    assert strategy._recycle_has_stop_condition() is expected


def test_resolve_lane_count_clamps_to_corpus_when_unbounded():
    """``recycle_is_bounded=False`` (no stop condition) clamps lanes to the corpus
    size; True leaves the full concurrency fan-out."""
    strategy = _minimal_strategy(_Cfg(concurrency=8))
    # Unbounded: single corpus pass -> clamp to total traces.
    assert strategy._resolve_lane_count(3, recycle_is_bounded=False) == 3
    # Bounded: sustain full concurrency even past the corpus size.
    assert strategy._resolve_lane_count(3, recycle_is_bounded=True) == 8


# --- unknown-return warning ---------------------------------------------------


@dataclass
class _Credit:
    trace_id: str | None
    node_ordinal: int | None = None
    x_correlation_id: str = "x"
    turn_index: int = 0


def test_unknown_trace_return_logs_warning_with_instance_and_ordinal(caplog):
    """A return for an instance id with no live adapter must WARN (not debug),
    naming the instance id + node ordinal so the drop is field-diagnosable."""
    strategy = _minimal_strategy(_Cfg())
    with caplog.at_level(logging.WARNING):
        strategy._on_graph_return(
            _Credit(trace_id="t-1#0.0", node_ordinal=7), error=None, cancelled=False
        )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "an unknown-instance return must log at WARNING"
    msg = warnings[-1].getMessage()
    assert "t-1#0.0" in msg
    assert "7" in msg


def test_none_trace_id_return_is_silent_noop(caplog):
    """A credit with no trace_id is not a graph return at all -> silent no-op,
    NOT a warning (only an UNKNOWN graph instance id warns)."""
    strategy = _minimal_strategy(_Cfg())
    with caplog.at_level(logging.WARNING):
        strategy._on_graph_return(_Credit(trace_id=None), error=None, cancelled=False)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
