# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit, TurnToSend


def _make_credit(**overrides) -> Credit:
    base = dict(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conv",
        x_correlation_id="x",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
    )
    base.update(overrides)
    return Credit(**base)


def test_credit_defaults():
    c = _make_credit()
    assert c.agent_depth == 0
    assert c.parent_correlation_id is None
    assert c.counts_toward_phase_target is True


def test_credit_with_depth_and_parent():
    c = _make_credit(agent_depth=2, parent_correlation_id="p")
    assert c.agent_depth == 2
    assert c.parent_correlation_id == "p"


def test_turn_to_send_propagates_depth_from_previous_credit():
    prev = _make_credit(
        agent_depth=2,
        parent_correlation_id="p",
        counts_toward_phase_target=False,
    )
    tts = TurnToSend.from_previous_credit(prev)
    assert tts.agent_depth == 2
    assert tts.parent_correlation_id == "p"
    assert tts.counts_toward_phase_target is False


def test_turn_to_send_defaults():
    tts = TurnToSend(
        conversation_id="c", x_correlation_id="x", turn_index=1, num_turns=2
    )
    assert tts.agent_depth == 0
    assert tts.parent_correlation_id is None
    assert tts.counts_toward_phase_target is True
