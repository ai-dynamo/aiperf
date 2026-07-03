# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``UserSession.is_fork_parent`` being stamped at session creation.

The flag is computed once from ``conversation.branches`` at
``UserSessionManager.create_and_store`` time and persisted on the
``UserSession``. Callers must not recompute it lazily on every read,
because the PAYLOAD_BYTES context-mode round-trip drops
``conversation.branches`` for wire-size — a lazy read after that round
trip would silently regress to ``False`` and break sticky-routing
eviction on FORK parents.
"""

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.workers.session_manager import UserSessionManager


@pytest.fixture
def session_manager() -> UserSessionManager:
    return UserSessionManager()


def _conv_with_fork(session_id: str = "root") -> Conversation:
    fork_branch = ConversationBranchInfo(
        branch_id=f"{session_id}:0",
        mode=ConversationBranchMode.FORK,
        child_conversation_ids=["child-x"],
    )
    return Conversation(
        session_id=session_id,
        turns=[Turn(branch_ids=[fork_branch.branch_id])],
        branches=[fork_branch],
    )


def _conv_with_spawn(session_id: str = "spawnroot") -> Conversation:
    spawn_branch = ConversationBranchInfo(
        branch_id=f"{session_id}:0",
        mode=ConversationBranchMode.SPAWN,
        child_conversation_ids=["child-y"],
    )
    return Conversation(
        session_id=session_id,
        turns=[Turn(branch_ids=[spawn_branch.branch_id])],
        branches=[spawn_branch],
    )


def _conv_no_fork(session_id: str = "linear") -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[Turn()],
    )


class TestIsForkParentStamping:
    def test_stamped_true_for_forking_conversation(
        self, session_manager: UserSessionManager
    ) -> None:
        conv = _conv_with_fork()
        session = session_manager.create_and_store(
            x_correlation_id="corr-fork",
            conversation=conv,
            num_turns=1,
        )
        assert session.is_fork_parent is True

    def test_stamped_false_for_linear_conversation(
        self, session_manager: UserSessionManager
    ) -> None:
        conv = _conv_no_fork()
        session = session_manager.create_and_store(
            x_correlation_id="corr-linear",
            conversation=conv,
            num_turns=1,
        )
        assert session.is_fork_parent is False

    def test_stamped_false_for_spawn_only_conversation(
        self, session_manager: UserSessionManager
    ) -> None:
        """SPAWN children do not require pinning their parent — only FORK does."""
        conv = _conv_with_spawn()
        session = session_manager.create_and_store(
            x_correlation_id="corr-spawn",
            conversation=conv,
            num_turns=1,
        )
        assert session.is_fork_parent is False

    def test_survives_payload_bytes_round_trip(
        self, session_manager: UserSessionManager
    ) -> None:
        """The flag must survive losing ``conversation.branches`` post-creation.

        Simulates the PAYLOAD_BYTES context-mode round-trip that strips
        ``branches`` from the conversation: a recompute-on-read
        implementation would flip the flag to ``False`` here.
        """
        conv = _conv_with_fork()
        session = session_manager.create_and_store(
            x_correlation_id="corr-pb",
            conversation=conv,
            num_turns=1,
        )
        assert session.is_fork_parent is True

        # Simulate the wire round-trip dropping branches.
        session.conversation.branches = []

        assert session.is_fork_parent is True


class TestForkIncompatibleConfigsRejected:
    """FORK-mode parents are structurally incompatible with two configs:
    ``MESSAGE_ARRAY_WITH_RESPONSES`` (replaces ``turn_list`` on every
    ``advance_turn``, wiping the seed) and ``raw_payload`` turns
    (no role/content for the child to render). dag_jsonl never produces
    either combination, but ``create_and_store`` defends against
    hand-authored or future-loader configs that bypass that pinning.
    """

    def test_fork_with_message_array_with_responses_rejected(
        self, session_manager: UserSessionManager
    ) -> None:
        from aiperf.common.enums import ConversationContextMode

        conv = _conv_with_fork()
        conv.context_mode = ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        with pytest.raises(NotImplementedError, match="message_array_with_responses"):
            session_manager.create_and_store(
                x_correlation_id="corr-bad-mode",
                conversation=conv,
                num_turns=1,
            )

    def test_fork_with_raw_payload_turn_rejected(
        self, session_manager: UserSessionManager
    ) -> None:
        conv = _conv_with_fork()
        # Replace the single turn with a raw_payload turn while keeping
        # the same branch_ids so the conversation is still fork-parent.
        branch_ids = list(conv.turns[0].branch_ids)
        conv.turns = [Turn(branch_ids=branch_ids, raw_payload={"some": "payload"})]
        with pytest.raises(NotImplementedError, match="raw_payload"):
            session_manager.create_and_store(
                x_correlation_id="corr-bad-payload",
                conversation=conv,
                num_turns=1,
            )

    def test_non_fork_with_message_array_with_responses_allowed(
        self, session_manager: UserSessionManager
    ) -> None:
        from aiperf.common.enums import ConversationContextMode

        conv = _conv_no_fork()
        conv.context_mode = ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        session = session_manager.create_and_store(
            x_correlation_id="corr-ok",
            conversation=conv,
            num_turns=1,
        )
        assert session.is_fork_parent is False
