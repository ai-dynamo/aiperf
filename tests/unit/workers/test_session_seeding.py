# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for mid-conversation session seeding (UserSession.hydrate_seed_history)."""

import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.workers.session_manager import UserSessionManager


def _conv(n_turns: int, with_seed_responses: bool = True) -> Conversation:
    """Conversation whose turns carry pre-synthesized assistant placeholders."""
    return Conversation(
        conversation_id="seed-conv",
        turns=[
            Turn(
                role="user",
                raw_messages=[{"role": "user", "content": f"u{i}"}],
                max_tokens=10 * (i + 1),
                seed_response=(
                    Turn(
                        role="assistant",
                        raw_messages=[{"role": "assistant", "content": f"a{i}"}],
                    )
                    if with_seed_responses
                    else None
                ),
            )
            for i in range(n_turns)
        ],
    )


class TestHydrateSeedHistory:
    @pytest.mark.parametrize(
        "k",
        [
            param(1, id="k=1"),
            param(2, id="k=2"),
            param(4, id="k=4"),
        ],
    )  # fmt: skip
    def test_hydrates_user_assistant_pairs_for_prior_turns(self, k: int) -> None:
        mgr = UserSessionManager()
        conv = _conv(5)
        session = mgr.create_and_store("cid", conv, num_turns=5)

        session.hydrate_seed_history(k)

        # Two entries (user + synthetic assistant) per reconstructed prior turn.
        assert len(session.turn_list) == 2 * k
        assert [t.role for t in session.turn_list] == ["user", "assistant"] * k

        # The real start turn appends on top via the normal advance path.
        session.advance_turn(k)
        assert len(session.turn_list) == 2 * k + 1
        assert session.turn_list[-1] is conv.turns[k]

    def test_zero_is_noop(self) -> None:
        mgr = UserSessionManager()
        session = mgr.create_and_store("cid", _conv(3), num_turns=3)
        session.hydrate_seed_history(0)
        assert session.turn_list == []

    def test_missing_seed_response_contributes_user_only(self) -> None:
        mgr = UserSessionManager()
        conv = _conv(3)
        conv.turns[1].seed_response = None  # a turn with no placeholder
        session = mgr.create_and_store("cid", conv, num_turns=3)

        session.hydrate_seed_history(3)

        # u0, a0, u1, u2, a2  (turn 1 contributes only its user side)
        assert [t.role for t in session.turn_list] == [
            "user",
            "assistant",
            "user",
            "user",
            "assistant",
        ]

    def test_deltas_with_responses_accumulates_authored_turns_only(self) -> None:
        mgr = UserSessionManager()
        conv = _conv(3)
        conv.context_mode = ConversationContextMode.DELTAS_WITH_RESPONSES
        session = mgr.create_and_store("cid", conv, num_turns=3)

        session.hydrate_seed_history(2)

        assert session.turn_list == conv.turns[:2]

    def test_full_context_starts_directly_at_self_contained_row(self) -> None:
        mgr = UserSessionManager()
        conv = _conv(3)
        conv.context_mode = ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        session = mgr.create_and_store("cid", conv, num_turns=3)

        session.hydrate_seed_history(2)
        assert session.turn_list == []

        selected = session.advance_turn(2)
        assert selected is conv.turns[2]
        assert session.turn_list == [conv.turns[2]]
