# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``min_start_delay_us`` must contribute its gate even when the same edge also
carries a first-token anchor.

The first-token branch of ``_compute_firing_gate_us`` ``continue``s past the
rest of the loop body once an observed first-token wall is found. If
``min_start_delay_us`` were evaluated after that branch it would be silently
dropped on any edge setting both -- a shape no in-repo adapter emits today and
nothing rejects, so nothing else pins the ordering.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.graph.context import _TraceContext
from aiperf.graph.executor import TraceExecutor

_MIN_START_US = 5_000.0
_FIRABLE_WALL_US = 1_000.0
# The min-start gate (1000 + 5000) must dominate both anchor fallbacks so a
# dropped contribution changes the result instead of being masked by a max().
_EXPECTED_GATE_US = _FIRABLE_WALL_US + _MIN_START_US


def _llm(output: str) -> LlmNode:
    return LlmNode(prompt=[f"@{output}"], output=output)


def _both_fields_graph() -> GraphRecord:
    """Edge a->b carries min_start_delay_us AND the first-token anchor pair."""
    return GraphRecord(
        nodes={"a": _llm("a"), "b": _llm("b")},
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(
                source="a",
                target="b",
                min_start_delay_us=_MIN_START_US,
                delay_after_predecessor_start_us=10.0,
                delay_after_predecessor_first_token_us=20.0,
            ),
        ],
        state={},
    )


def _gate_for(
    *, first_token_wall_us: float | None, dispatch_wall_us: float | None
) -> float:
    parsed = ParsedGraph(graph=_both_fields_graph(), traces=[TraceRecord(id="t")])
    executor = TraceExecutor(parsed)
    # ``store`` is never dereferenced by the gate computation.
    ctx = _TraceContext(trace=parsed.traces[0], store=None)  # type: ignore[arg-type]
    if first_token_wall_us is not None:
        ctx.node_first_token_wall_us["a"] = first_token_wall_us
    if dispatch_wall_us is not None:
        ctx.node_dispatch_wall_us["a"] = dispatch_wall_us
    return executor._compute_firing_gate_us("b", ctx, _FIRABLE_WALL_US)


@pytest.mark.parametrize(
    ("first_token_wall_us", "dispatch_wall_us"),
    [
        param(100.0, 90.0, id="observed_first_token_taken"),
        param(None, 90.0, id="no_first_token_dispatch_fallback"),
        param(None, None, id="neither_wall_recorded"),
    ],
)  # fmt: skip
def test_compute_firing_gate_min_start_delay_survives_first_token_continue(
    first_token_wall_us: float | None, dispatch_wall_us: float | None
) -> None:
    """The min-start gate applies on every anchor path through the loop body.

    The observed-first-token case is the regression: that branch ``continue``s,
    so an edge evaluated in the wrong order fires ~5ms early.
    """
    assert (
        _gate_for(
            first_token_wall_us=first_token_wall_us,
            dispatch_wall_us=dispatch_wall_us,
        )
        == _EXPECTED_GATE_US
    )


def test_compute_firing_gate_first_token_anchor_wins_when_it_dominates() -> None:
    """The reorder must not clamp the gate to min_start_delay_us.

    A first token late enough to exceed the min-start gate still governs, so
    both contributions are genuinely max()-accumulated rather than one
    shadowing the other.
    """
    late_first_token_us = _EXPECTED_GATE_US + 1_000.0
    assert _gate_for(
        first_token_wall_us=late_first_token_us, dispatch_wall_us=90.0
    ) == pytest.approx(late_first_token_us + 20.0)
