# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for UserSessionManager to ensure Credit.num_turns is respected.

These tests ensure that the worker properly uses Credit.num_turns instead of
len(conversation.turns), which is critical for ramp-up users who start mid-session.
"""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.dataset_models import DatasetMetadata
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.workers.session_manager import UserSession, UserSessionManager


@pytest.fixture
def session_manager():
    """Create a UserSessionManager instance."""
    return UserSessionManager()


@pytest.fixture
def sample_conversation():
    """Create a sample conversation with 5 turns."""
    return Conversation(
        conversation_id="test-conv",
        turns=[
            Turn(messages=[{"role": "user", "content": f"Question {i + 1}"}])
            for i in range(5)
        ],
    )


class TestUserSessionManager:
    """Tests for UserSessionManager Credit.num_turns handling."""

    def test_create_session_uses_credit_num_turns_not_conversation_length(
        self, session_manager, sample_conversation
    ):
        """Ensure UserSession.num_turns comes from Credit, not conversation.

        This is critical for ramp-up users who may only execute 1 turn even though
        the conversation template has 5 turns available.
        """
        # Conversation has 5 turns, but Credit says only do 1
        session = session_manager.create_and_store(
            x_correlation_id="test-corr-id",
            conversation=sample_conversation,
            num_turns=1,  # Artificial cap from Credit
        )

        # UserSession should use Credit.num_turns (1), not len(conversation.turns) (5)
        assert session.num_turns == 1
        assert len(session.conversation.turns) == 5  # Conversation still has all turns

    def test_advance_turn_validates_against_credit_num_turns(
        self, session_manager, sample_conversation
    ):
        """Ensure turn validation uses Credit.num_turns."""
        session = session_manager.create_and_store(
            x_correlation_id="test-corr-id",
            conversation=sample_conversation,
            num_turns=2,  # Only 2 turns allowed
        )

        # Should be able to advance to turn 0 and 1
        session.advance_turn(0)
        assert session.turn_index == 0

        session.advance_turn(1)
        assert session.turn_index == 1

        # Should reject turn 2 (out of range for num_turns=2)
        with pytest.raises(
            ValueError,
            match="Turn index 2 is out of range for conversation with 2 turns",
        ):
            session.advance_turn(2)

    def test_ramp_up_user_single_turn_scenario(
        self, session_manager, sample_conversation
    ):
        """Test ramp-up user who only executes 1 turn (e.g., User 1 starting at Turn 5).

        This simulates multi-round-qa's ramp-up behavior where some users are
        initialized mid-session and only complete their final turn.
        """
        # User 1 in ramp-up: starts at question_id=5, only does 1 turn
        session = session_manager.create_and_store(
            x_correlation_id="ramp-up-user-1",
            conversation=sample_conversation,
            num_turns=1,  # Only 1 turn to execute
        )

        # Advance to turn 0 (their only turn)
        turn = session.advance_turn(0)

        # Should access first turn of conversation (conversation has all 5 turns available)
        assert turn.messages[0]["content"] == "Question 1"

        # After turn 0, is_final_turn should be True (0 == 1-1)
        # This would be determined by Credit.is_final_turn, which we validate here
        assert session.turn_index == 0
        assert session.num_turns == 1
        # Credit.is_final_turn would be: turn_index (0) == num_turns (1) - 1 → True

    def test_full_session_uses_all_conversation_turns(
        self, session_manager, sample_conversation
    ):
        """Test normal user who executes all turns (e.g., steady-state users)."""
        session = session_manager.create_and_store(
            x_correlation_id="full-session-user",
            conversation=sample_conversation,
            num_turns=5,  # All turns
        )

        assert session.num_turns == 5

        # Should be able to advance through all 5 turns
        for turn_idx in range(5):
            turn = session.advance_turn(turn_idx)
            assert turn.messages[0]["content"] == f"Question {turn_idx + 1}"

    def test_partial_session_mid_conversation(
        self, session_manager, sample_conversation
    ):
        """Test user who starts mid-session and does partial turns (e.g., User 4 doing 3 turns)."""
        session = session_manager.create_and_store(
            x_correlation_id="partial-user",
            conversation=sample_conversation,
            num_turns=3,  # Only 3 turns (simulating User 4 at question_id=3)
        )

        assert session.num_turns == 3

        # Can advance turns 0, 1, 2
        for turn_idx in range(3):
            turn = session.advance_turn(turn_idx)
            assert turn is not None

        # Turn 3 should fail (out of range)
        with pytest.raises(ValueError, match="out of range"):
            session.advance_turn(3)

    def test_url_index_stored_for_multi_url_load_balancing(
        self, session_manager, sample_conversation
    ):
        """Test that url_index is stored in session for multi-URL load balancing.

        When using multiple --url endpoints with multi-turn conversations, the first
        turn gets a url_index from the round-robin sampler. All subsequent turns must
        use the same url_index to ensure the entire conversation hits the same backend.
        """
        # First turn: Credit provides url_index=2 from round-robin
        session = session_manager.create_and_store(
            x_correlation_id="multi-url-session",
            conversation=sample_conversation,
            num_turns=3,
            url_index=2,  # From Credit on first turn
        )

        # Session stores the url_index for subsequent turns
        assert session.url_index == 2

        # All turns should use this stored url_index (worker reads from session)
        for turn_idx in range(3):
            session.advance_turn(turn_idx)
            # Worker would use session.url_index (2) for every turn
            assert session.url_index == 2

    def test_url_index_none_for_single_url_mode(
        self, session_manager, sample_conversation
    ):
        """Test that url_index can be None when only one URL is configured."""
        session = session_manager.create_and_store(
            x_correlation_id="single-url-session",
            conversation=sample_conversation,
            num_turns=2,
            url_index=None,  # No multi-URL load balancing
        )

        assert session.url_index is None


# ============================================================
# Fixtures for context mode tests
# ============================================================


def _make_session(
    context_mode: ConversationContextMode | None = None,
    num_turns: int = 3,
    default_context_mode: ConversationContextMode | None = None,
) -> UserSession:
    """Create a UserSession with the given context_mode on its conversation."""
    conversation = Conversation(
        conversation_id="ctx-conv",
        context_mode=context_mode,
        turns=[
            Turn(messages=[{"role": "user", "content": f"Q{i}"}])
            for i in range(num_turns)
        ],
    )
    mgr = UserSessionManager()
    mgr.set_default_context_mode(default_context_mode)
    return mgr.create_and_store(
        x_correlation_id="ctx-test",
        conversation=conversation,
        num_turns=num_turns,
    )


# ============================================================
# Context Mode Resolution
# ============================================================


class TestUserSessionContextModeResolution:
    """Verify context_mode resolves: conversation > dataset default > DELTAS_WITHOUT_RESPONSES."""

    @pytest.mark.parametrize(
        "conversation_mode,expected",
        [
            (None, ConversationContextMode.DELTAS_WITHOUT_RESPONSES),
            (ConversationContextMode.DELTAS_WITHOUT_RESPONSES, ConversationContextMode.DELTAS_WITHOUT_RESPONSES),
            (ConversationContextMode.DELTAS_WITH_RESPONSES, ConversationContextMode.DELTAS_WITH_RESPONSES),
            (ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES, ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES),
        ],
    )  # fmt: skip
    def test_context_mode_resolves_correctly(
        self,
        conversation_mode: ConversationContextMode | None,
        expected: ConversationContextMode,
    ) -> None:
        session = _make_session(context_mode=conversation_mode)
        assert session.context_mode == expected

    def test_dataset_default_used_when_conversation_has_none(self) -> None:
        session = _make_session(
            context_mode=None,
            default_context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
        )
        assert (
            session.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )

    def test_conversation_overrides_dataset_default(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
            default_context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
        )
        assert session.context_mode == ConversationContextMode.DELTAS_WITH_RESPONSES

    def test_global_default_when_both_none(self) -> None:
        session = _make_session(context_mode=None, default_context_mode=None)
        assert session.context_mode == ConversationContextMode.DELTAS_WITHOUT_RESPONSES


# ============================================================
# should_store_response
# ============================================================


class TestUserSessionShouldStoreResponse:
    """Verify should_store_response gates on context mode."""

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (ConversationContextMode.DELTAS_WITHOUT_RESPONSES, True),
            (ConversationContextMode.DELTAS_WITH_RESPONSES, False),
            (ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES, False),
            param(None, True, id="default-deltas-without-responses"),
        ],
    )  # fmt: skip
    def test_should_store_response_per_mode(
        self, mode: ConversationContextMode | None, expected: bool
    ) -> None:
        session = _make_session(context_mode=mode)
        assert session.should_store_response() is expected


# ============================================================
# turn_list with context mode
# ============================================================


class TestUserSessionTurnList:
    """Verify turn_list contains correct turns based on context mode."""

    def test_deltas_without_responses_returns_full_history(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        )
        session.advance_turn(0)
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 3  # Q0, A0, Q1
        assert turns[0].messages[0]["content"] == "Q0"
        assert turns[1].messages[0]["content"] == "A0"
        assert turns[2].messages[0]["content"] == "Q1"

    def test_deltas_with_responses_returns_dataset_turns_only(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES
        )
        session.advance_turn(0)
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 2  # Q0, Q1 (no assistant responses stored)
        assert turns[0].messages[0]["content"] == "Q0"
        assert turns[1].messages[0]["content"] == "Q1"

    def test_message_array_returns_only_last(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )
        session.advance_turn(0)
        session.advance_turn(1)
        session.advance_turn(2)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q2"

    def test_message_array_single_turn(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            num_turns=1,
        )
        session.advance_turn(0)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q0"

    def test_default_mode_returns_full_history(self) -> None:
        session = _make_session(context_mode=None)
        session.advance_turn(0)
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 3


# ============================================================
# Integration: context mode + should_store_response together
# ============================================================


class TestUserSessionContextModeWorkflow:
    """Verify the full workflow of context mode with store_response gating."""

    def test_deltas_without_responses_stores_responses_and_sends_full_history(
        self,
    ) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES, num_turns=2
        )
        session.advance_turn(0)
        assert session.should_store_response() is True
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        assert len(session.turn_list) == 3

    def test_deltas_with_responses_skips_live_responses_sends_dataset_turns(
        self,
    ) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES, num_turns=2
        )
        session.advance_turn(0)
        assert session.should_store_response() is False
        # Worker would NOT call store_response based on should_store_response()
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 2
        assert all(t.messages[0]["role"] == "user" for t in turns)

    def test_message_array_skips_responses_sends_only_current_turn(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            num_turns=2,
        )
        session.advance_turn(0)
        assert session.should_store_response() is False
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q1"


# ============================================================
# message_array_without_responses rejected
# ============================================================


class TestMessageArrayWithoutResponsesRejected:
    """MESSAGE_ARRAY_WITHOUT_RESPONSES is reserved and must be rejected early."""

    def test_conversation_rejects_unsupported_mode(self) -> None:
        with pytest.raises(ValidationError, match="not yet supported"):
            Conversation(
                context_mode=ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES,
            )

    def test_dataset_metadata_rejects_unsupported_default_mode(self) -> None:
        with pytest.raises(ValidationError, match="not yet supported"):
            DatasetMetadata(
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
                default_context_mode=ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES,
            )


# ============================================================
# DAG child session seeding from parent (sticky-routing locality)
# ============================================================


class TestDAGChildSeeding:
    """FORK-mode children inherit the parent's turn_list at creation time
    (sticky routing guarantees parent and child live on the same worker)."""

    def test_seed_turn_list_from_parent_session_under_fork(self, sample_conversation):
        from aiperf.common.models.dataset_models import Text, Turn

        mgr = UserSessionManager()
        parent = mgr.create_and_store(
            x_correlation_id="parent-1",
            conversation=sample_conversation,
            num_turns=2,
        )
        # Simulate the parent having completed a turn (advance + captured response).
        parent.turn_list = [
            Turn(role="user", texts=[Text(contents=["parent user"])]),
            Turn(role="assistant", texts=[Text(contents=["parent response"])]),
        ]

        child = mgr.create_and_store(
            x_correlation_id="child-1",
            conversation=sample_conversation,
            num_turns=1,
            parent_correlation_id="parent-1",
        )
        assert [t.role for t in child.turn_list] == ["user", "assistant"]
        assert child.turn_list == parent.turn_list
        # Must be a clone, not a shared reference.
        assert child.turn_list is not parent.turn_list

    def test_missing_parent_raises_runtime_error(self, sample_conversation):
        mgr = UserSessionManager()
        with pytest.raises(RuntimeError, match="FORK routing invariant violated"):
            mgr.create_and_store(
                x_correlation_id="child-1",
                conversation=sample_conversation,
                num_turns=1,
                parent_correlation_id="missing-parent",
            )

    def test_no_parent_corr_leaves_turn_list_empty(self, sample_conversation):
        mgr = UserSessionManager()
        session = mgr.create_and_store(
            x_correlation_id="solo-1",
            conversation=sample_conversation,
            num_turns=1,
        )
        assert session.turn_list == []

    def test_spawn_mode_does_not_require_parent_and_starts_empty(
        self, sample_conversation
    ):
        """SPAWN-mode children start with a fresh context and do NOT trigger
        the FORK-mode parent-lookup invariant, even when parent_correlation_id
        is set (they may share sticky routing for unrelated reasons)."""
        from aiperf.common.enums import ConversationBranchMode

        mgr = UserSessionManager()
        child = mgr.create_and_store(
            x_correlation_id="spawn-child",
            conversation=sample_conversation,
            num_turns=1,
            parent_correlation_id="never-registered-parent",
            branch_mode=ConversationBranchMode.SPAWN,
        )
        assert child.turn_list == []


# ============================================================
# FORK-pin eviction: refcount-based cache cleanup
# ============================================================


def _make_parent_conv_with_fork(child_ids: list[str]) -> Conversation:
    """Build a Conversation that declares a FORK branch for pin-testing."""
    from aiperf.common.enums import ConversationBranchMode
    from aiperf.common.models.branch import ConversationBranchInfo

    return Conversation(
        conversation_id="parent-conv",
        turns=[
            Turn(messages=[{"role": "user", "content": "q"}]),
            Turn(messages=[{"role": "user", "content": "final"}]),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="parent-conv:0",
                child_conversation_ids=child_ids,
                mode=ConversationBranchMode.FORK,
            )
        ],
    )


def _make_child_conv(session_id: str) -> Conversation:
    return Conversation(
        conversation_id=session_id,
        turns=[Turn(messages=[{"role": "user", "content": "c"}])],
    )


class TestForkPinEviction:
    """FORK parents stay pinned in the cache while live FORK children
    exist so late-arriving children can still seed from the parent's
    ``turn_list``. The pin is released (and the parent evicted) once
    the last child evicts — preventing the unbounded cache growth that
    a naive never-evict pin would cause on long-running DAG benchmarks.
    """

    def test_parent_with_no_children_pins_until_teardown(self):
        """A FORK-parent's ``evict`` unconditionally goes to pending — the
        worker's credit-return path evicts the parent BEFORE children
        have been dispatched back to this worker, so popping at evict
        time would race the children's sticky-routing seed lookup. If
        children truly never spawn, the parent stays pinned until
        session-manager teardown."""
        mgr = UserSessionManager()
        parent_conv = _make_parent_conv_with_fork(["child-1"])
        mgr.create_and_store(
            x_correlation_id="parent",
            conversation=parent_conv,
            num_turns=2,
        )
        assert mgr.get("parent") is not None

        mgr.evict("parent")

        # Parent stays cached; pending_eviction holds it until the last
        # (potential) child evicts.
        assert mgr.get("parent") is not None
        assert "parent" in mgr._pending_eviction

    def test_parent_with_live_fork_child_is_pinned(self):
        """FORK child in flight → parent goes to pending_eviction, not popped."""
        from aiperf.common.enums import ConversationBranchMode

        mgr = UserSessionManager()
        parent_conv = _make_parent_conv_with_fork(["child-1"])
        mgr.create_and_store(
            x_correlation_id="parent", conversation=parent_conv, num_turns=2
        )
        mgr.create_and_store(
            x_correlation_id="child-1",
            conversation=_make_child_conv("child-1"),
            num_turns=1,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.FORK,
        )

        mgr.evict("parent")

        assert mgr.get("parent") is not None
        assert "parent" in mgr._pending_eviction
        assert mgr._fork_child_count["parent"] == 1

    def test_child_evict_cascades_to_pending_parent(self):
        """Last FORK child evicting drops the parent if it was pending."""
        from aiperf.common.enums import ConversationBranchMode

        mgr = UserSessionManager()
        parent_conv = _make_parent_conv_with_fork(["child-1"])
        mgr.create_and_store(
            x_correlation_id="parent", conversation=parent_conv, num_turns=2
        )
        mgr.create_and_store(
            x_correlation_id="child-1",
            conversation=_make_child_conv("child-1"),
            num_turns=1,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.FORK,
        )
        mgr.evict("parent")  # pending_eviction now

        mgr.evict("child-1")

        assert mgr.get("parent") is None
        assert mgr.get("child-1") is None
        assert "parent" not in mgr._pending_eviction
        assert "parent" not in mgr._fork_child_count

    def test_multiple_children_decrement_one_at_a_time(self):
        """Parent stays pinned until the LAST FORK child evicts."""
        from aiperf.common.enums import ConversationBranchMode

        mgr = UserSessionManager()
        parent_conv = _make_parent_conv_with_fork(["c1", "c2", "c3"])
        mgr.create_and_store(
            x_correlation_id="parent", conversation=parent_conv, num_turns=2
        )
        for cid in ("c1", "c2", "c3"):
            mgr.create_and_store(
                x_correlation_id=cid,
                conversation=_make_child_conv(cid),
                num_turns=1,
                parent_correlation_id="parent",
                branch_mode=ConversationBranchMode.FORK,
            )
        assert mgr._fork_child_count["parent"] == 3

        mgr.evict("parent")
        assert mgr.get("parent") is not None

        mgr.evict("c1")
        assert mgr.get("parent") is not None
        assert mgr._fork_child_count["parent"] == 2

        mgr.evict("c2")
        assert mgr.get("parent") is not None
        assert mgr._fork_child_count["parent"] == 1

        mgr.evict("c3")
        assert mgr.get("parent") is None
        assert "parent" not in mgr._fork_child_count

    def test_child_evict_before_parent_evict_does_not_pop_parent(self):
        """Children evicting before the parent's own final turn must not
        cascade-evict — the parent is still live."""
        from aiperf.common.enums import ConversationBranchMode

        mgr = UserSessionManager()
        parent_conv = _make_parent_conv_with_fork(["c1"])
        mgr.create_and_store(
            x_correlation_id="parent", conversation=parent_conv, num_turns=2
        )
        mgr.create_and_store(
            x_correlation_id="c1",
            conversation=_make_child_conv("c1"),
            num_turns=1,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.FORK,
        )

        # Parent hasn't reached its final turn yet, so no evict(parent) call.
        mgr.evict("c1")

        assert mgr.get("parent") is not None
        assert "parent" not in mgr._pending_eviction
        # Refcount drops even without a pending parent — safe.
        assert mgr._fork_child_count.get("parent", 0) == 0

        # Parent's final turn later: FORK parent always goes pending (no
        # way to distinguish "children already done" from "children still
        # en route" at evict time without coupling to the orchestrator).
        # The pending set is cleaned up on phase teardown.
        mgr.evict("parent")
        assert "parent" in mgr._pending_eviction

    def test_spawn_child_does_not_bump_parent_refcount(self):
        """SPAWN children never seed from the parent's turn_list, so they
        should not pin the parent in the cache."""
        from aiperf.common.enums import ConversationBranchMode

        mgr = UserSessionManager()
        parent_conv = _make_parent_conv_with_fork(["spawn-1"])
        # NOTE: parent declares FORK branches (so the pin machinery runs),
        # but the child uses SPAWN mode — it's the child's mode that
        # determines whether the refcount is bumped.
        mgr.create_and_store(
            x_correlation_id="parent", conversation=parent_conv, num_turns=2
        )
        mgr.create_and_store(
            x_correlation_id="spawn-1",
            conversation=_make_child_conv("spawn-1"),
            num_turns=1,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.SPAWN,
        )

        assert "parent" not in mgr._fork_child_count

        # Parent still goes pending (it declares FORK branches); SPAWN
        # children alone can't cascade-drop it because they don't take
        # a refcount. In practice the FORK children that the branches
        # declare will be what cascades.
        mgr.evict("parent")
        assert "parent" in mgr._pending_eviction

    def test_non_dag_session_still_evicts_cleanly(self, sample_conversation):
        """The refactor must not regress plain-session eviction."""
        mgr = UserSessionManager()
        mgr.create_and_store(
            x_correlation_id="solo",
            conversation=sample_conversation,
            num_turns=1,
        )
        mgr.evict("solo")
        assert mgr.get("solo") is None

    def test_serial_single_turn_children_do_not_strand_parent(self):
        """Regression: single-turn FORK children created+evicted one at a time
        (serial dispatch / concurrency 1) must NOT strand later siblings. The
        live refcount hits zero between children, but the parent stays pinned
        until the full DECLARED child set has been created. Without the
        ``_fork_created >= _fork_expected`` gate this popped the parent after
        the first child, and every later child raised the
        ``FORK routing invariant violated`` error (wide fan-out)."""
        from aiperf.common.enums import ConversationBranchMode

        mgr = UserSessionManager()
        parent_conv = _make_parent_conv_with_fork(["c1", "c2", "c3"])
        mgr.create_and_store(
            x_correlation_id="parent", conversation=parent_conv, num_turns=2
        )
        # Parent's final turn completes before any child credit is processed.
        mgr.evict("parent")
        assert "parent" in mgr._pending_eviction

        # Serial: each child is created AND evicted before the next exists.
        for cid in ("c1", "c2"):
            mgr.create_and_store(
                x_correlation_id=cid,
                conversation=_make_child_conv(cid),
                num_turns=1,
                parent_correlation_id="parent",
                branch_mode=ConversationBranchMode.FORK,
            )
            assert mgr.get(cid) is not None  # seeds fine: parent still cached
            mgr.evict(cid)
            # Transient zero: live count is 0 but declared children remain,
            # so the parent MUST stay cached.
            assert mgr.get("parent") is not None, (
                f"parent stranded after {cid} evicted (transient-zero pop)"
            )

        # Final declared child: the full set now exists; its eviction cascades.
        mgr.create_and_store(
            x_correlation_id="c3",
            conversation=_make_child_conv("c3"),
            num_turns=1,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.FORK,
        )
        assert mgr.get("c3") is not None
        mgr.evict("c3")
        assert mgr.get("parent") is None
        assert "parent" not in mgr._pending_eviction
        assert "parent" not in mgr._fork_expected
        assert "parent" not in mgr._fork_created

    def test_partial_child_set_keeps_parent_pinned(self):
        """When the cap truncates some declared children (their credits never
        reach this worker), ``created < expected`` so the parent stays pinned
        until teardown rather than popping early. The children that DID seed
        all found the parent; no sibling is stranded."""
        from aiperf.common.enums import ConversationBranchMode

        mgr = UserSessionManager()
        parent_conv = _make_parent_conv_with_fork(["c1", "c2", "c3"])
        mgr.create_and_store(
            x_correlation_id="parent", conversation=parent_conv, num_turns=2
        )
        mgr.evict("parent")
        # Only one of three declared children is ever created.
        mgr.create_and_store(
            x_correlation_id="c1",
            conversation=_make_child_conv("c1"),
            num_turns=1,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.FORK,
        )
        mgr.evict("c1")
        assert mgr.get("parent") is not None
        assert "parent" in mgr._pending_eviction
