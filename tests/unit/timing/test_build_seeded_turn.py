# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SampledSession.build_seeded_turn (mid-conversation start turns)."""

import pytest
from pytest import param

from aiperf.common.models import ConversationMetadata, TurnMetadata
from aiperf.timing.conversation_source import SampledSession


def _session(n_turns: int) -> SampledSession:
    return SampledSession(
        conversation_id="c",
        metadata=ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata() for _ in range(n_turns)],
        ),
        x_correlation_id="x",
    )


class TestBuildSeededTurn:
    @pytest.mark.parametrize(
        "k",
        [
            param(1, id="k=1"),
            param(3, id="k=3"),
            param(4, id="k=last"),
        ],
    )  # fmt: skip
    def test_starts_at_k(self, k: int):
        turn = _session(5).build_seeded_turn(k)
        assert turn.turn_index == k
        assert turn.start_turn_index == k
        assert turn.num_turns == 5
        assert turn.is_session_start is True
        assert turn.is_final_turn is (k == 4)

    def test_clamps_to_last_turn(self):
        # A fraction-derived index can never consume the whole conversation.
        turn = _session(3).build_seeded_turn(10)
        assert turn.turn_index == 2
        assert turn.start_turn_index == 2
        assert turn.is_final_turn is True

    @pytest.mark.parametrize(
        "k",
        [
            param(0, id="zero"),
            param(-1, id="negative"),
        ],
    )  # fmt: skip
    def test_non_positive_falls_back_to_first_turn(self, k: int):
        turn = _session(4).build_seeded_turn(k)
        assert turn.turn_index == 0
        assert turn.start_turn_index == 0
        assert turn.is_session_start is True

    def test_empty_conversation_raises(self):
        with pytest.raises(ValueError, match="empty conversation"):
            _session(0).build_seeded_turn(1)
