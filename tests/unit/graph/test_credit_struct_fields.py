# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph addressing fields on TurnToSend/Credit (Task 7)."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.credit.structs import Credit, TurnToSend

_GRAPH_ADDRESS = ("t0", 3)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        param(
            {"trace_id": "t0", "node_ordinal": 3},
            _GRAPH_ADDRESS,
            id="explicitly-addressed",
        ),
        param({}, (None, None), id="defaults-when-unaddressed"),
    ],
)  # fmt: skip
def test_turn_to_send_graph_addressing(
    extra: dict[str, Any], expected: tuple[str | None, int | None]
) -> None:
    """TurnToSend round-trips graph addressing, defaulting to an unaddressed turn."""
    turn = TurnToSend(
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        **extra,
    )
    assert (turn.trace_id, turn.node_ordinal) == expected


def test_credit_carries_graph_addressing() -> None:
    """Credit carries the same graph addressing triple through to the worker."""
    credit = Credit(
        id=0,
        phase="profiling",
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
        trace_id="t0",
        node_ordinal=3,
    )
    assert (credit.trace_id, credit.node_ordinal) == _GRAPH_ADDRESS
