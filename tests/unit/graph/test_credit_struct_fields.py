# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph addressing fields on TurnToSend/Credit (Task 7)."""

from aiperf.credit.structs import Credit, TurnToSend


def test_turn_to_send_carries_graph_addressing():
    t = TurnToSend(
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        trace_id="t0",
        node_ordinal=3,
        phase_variant="profiling",
    )
    assert (t.trace_id, t.node_ordinal, t.phase_variant) == ("t0", 3, "profiling")


def test_turn_to_send_graph_addressing_defaults():
    t = TurnToSend(conversation_id="c", x_correlation_id="x", turn_index=0, num_turns=1)
    assert (t.trace_id, t.node_ordinal, t.phase_variant) == (None, None, "profiling")


def test_credit_carries_graph_addressing():
    c = Credit(
        id=0,
        phase="profiling",
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
        trace_id="t0",
        node_ordinal=3,
        phase_variant="profiling",
    )
    assert (c.trace_id, c.node_ordinal, c.phase_variant) == ("t0", 3, "profiling")
