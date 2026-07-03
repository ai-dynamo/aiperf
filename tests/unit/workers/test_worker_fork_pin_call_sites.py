# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker call-site tests for FORK pin/seed/release refcount wiring.

The storage half (``UserSessionManager`` pin/release/evict semantics) is
covered by ``test_session_fork_refcount.py``; the ``is_fork_parent``
stamping by ``test_session_manager_fork_parent.py``. This file tests the
worker wiring: that ``Worker._pin_parent_if_fork_child`` /
``_seed_from_parent_if_fork_child`` / ``_release_and_evict_for_terminal``
drive that API correctly for FORK children, SPAWN children, and root
credits, using real ``UserSessionManager`` + ``Credit`` structs (no
MagicMock for the session structs — MagicMock hides attribute-path drift).

A bare ``Worker.__new__`` shell is used because the full ``Worker.__init__``
requires a live ZMQ comms + BenchmarkRun fixture that is irrelevant to the
session-cache logic under test; the three methods only touch
``self.session_manager`` and ``self.warning``.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import Text
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.credit.structs import Credit
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from aiperf.workers.session_manager import UserSessionManager
from aiperf.workers.worker import Worker
from tests.harness.endpoint_helpers import (
    _wrap_model_endpoint,
    create_config,
    create_request_info,
)


def _fork_conversation(session_id: str = "root") -> Conversation:
    fork_branch = ConversationBranchInfo(
        branch_id=f"{session_id}:0",
        mode=ConversationBranchMode.FORK,
        child_conversation_ids=["child-a"],
    )
    return Conversation(
        session_id=session_id,
        turns=[Turn(role="user", branch_ids=[fork_branch.branch_id])],
        branches=[fork_branch],
    )


def _linear_conversation(session_id: str = "child-a") -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[Turn(role="user")],
    )


def _make_credit(
    *,
    x_correlation_id: str,
    turn_index: int = 0,
    num_turns: int = 1,
    parent_correlation_id: str | None = None,
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK,
    has_forks: bool = False,
) -> Credit:
    return Credit(
        id=0,
        phase="profiling",
        conversation_id="conv",
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=1,
        parent_correlation_id=parent_correlation_id,
        branch_mode=branch_mode,
        has_forks=has_forks,
    )


@pytest.fixture
def worker() -> Worker:
    """Bare Worker shell with only the fork call-site collaborators wired."""
    w = Worker.__new__(Worker)
    w.session_manager = UserSessionManager()
    w.warning = lambda *args, **kwargs: None
    return w


def _seed_parent_with_history(worker: Worker, parent_corr: str) -> None:
    """Create the parent session and give it a two-Turn accumulated history."""
    parent = worker.session_manager.create_and_store(
        x_correlation_id=parent_corr,
        conversation=_fork_conversation(),
        num_turns=1,
    )
    parent.advance_turn(0)  # authored root turn
    parent.store_response(Turn(role="assistant", raw_messages=[{"role": "assistant"}]))
    assert len(parent.turn_list) == 2


class TestPinAndSeedForkChild:
    def test_fork_child_pins_parent_and_seeds_turn_list(self, worker: Worker) -> None:
        parent_corr = "parent-corr"
        child_corr = "child-corr"
        _seed_parent_with_history(worker, parent_corr)
        worker.session_manager.create_and_store(
            x_correlation_id=child_corr,
            conversation=_linear_conversation(),
            num_turns=1,
        )

        credit = _make_credit(
            x_correlation_id=child_corr,
            parent_correlation_id=parent_corr,
            branch_mode=ConversationBranchMode.FORK,
        )
        worker._pin_parent_if_fork_child(credit, child_corr)
        worker._seed_from_parent_if_fork_child(credit, child_corr)

        parent = worker.session_manager.get(parent_corr)
        child = worker.session_manager.get(child_corr)
        assert parent.fork_refcount == 1
        # Child inherited the parent's full accumulated history (copy, not alias).
        assert child.turn_list == parent.turn_list
        assert child.turn_list is not parent.turn_list

    def test_spawn_child_does_not_pin_or_seed(self, worker: Worker) -> None:
        parent_corr = "parent-corr"
        child_corr = "spawn-child"
        _seed_parent_with_history(worker, parent_corr)
        worker.session_manager.create_and_store(
            x_correlation_id=child_corr,
            conversation=_linear_conversation(),
            num_turns=1,
        )

        credit = _make_credit(
            x_correlation_id=child_corr,
            parent_correlation_id=parent_corr,
            branch_mode=ConversationBranchMode.SPAWN,
        )
        worker._pin_parent_if_fork_child(credit, child_corr)
        worker._seed_from_parent_if_fork_child(credit, child_corr)

        assert worker.session_manager.get(parent_corr).fork_refcount == 0
        assert worker.session_manager.get(child_corr).turn_list == []

    def test_root_credit_does_not_pin_or_seed(self, worker: Worker) -> None:
        root_corr = "root-corr"
        worker.session_manager.create_and_store(
            x_correlation_id=root_corr,
            conversation=_fork_conversation(),
            num_turns=1,
        )
        credit = _make_credit(x_correlation_id=root_corr, parent_correlation_id=None)
        # Must not raise and must not touch refcount.
        worker._pin_parent_if_fork_child(credit, root_corr)
        worker._seed_from_parent_if_fork_child(credit, root_corr)
        assert worker.session_manager.get(root_corr).fork_refcount == 0

    def test_fork_child_after_parent_evicted_warns_not_raises(
        self, worker: Worker
    ) -> None:
        child_corr = "orphan-child"
        worker.session_manager.create_and_store(
            x_correlation_id=child_corr,
            conversation=_linear_conversation(),
            num_turns=1,
        )
        credit = _make_credit(
            x_correlation_id=child_corr,
            parent_correlation_id="already-gone",
            branch_mode=ConversationBranchMode.FORK,
        )
        # Parent never existed: pin swallows KeyError, seed is a no-op.
        worker._pin_parent_if_fork_child(credit, child_corr)
        worker._seed_from_parent_if_fork_child(credit, child_corr)
        assert worker.session_manager.get(child_corr).turn_list == []


class TestReleaseAndEvictForTerminal:
    def test_non_fork_terminal_evicts_immediately(self, worker: Worker) -> None:
        corr = "linear-corr"
        worker.session_manager.create_and_store(
            x_correlation_id=corr,
            conversation=_linear_conversation(),
            num_turns=1,
        )
        credit = _make_credit(x_correlation_id=corr, parent_correlation_id=None)
        worker._release_and_evict_for_terminal(credit, corr)
        assert worker.session_manager.get(corr) is None

    def test_fork_parent_terminal_defers_eviction_until_children_join(
        self, worker: Worker
    ) -> None:
        parent_corr = "parent-corr"
        child_corr = "child-corr"
        _seed_parent_with_history(worker, parent_corr)
        worker.session_manager.create_and_store(
            x_correlation_id=child_corr,
            conversation=_linear_conversation(),
            num_turns=1,
        )

        # A FORK child has already pinned the parent (child arrived before the
        # parent's terminal return, the refcount-race path).
        child_credit = _make_credit(
            x_correlation_id=child_corr,
            parent_correlation_id=parent_corr,
            branch_mode=ConversationBranchMode.FORK,
        )
        worker._pin_parent_if_fork_child(child_credit, child_corr)
        assert worker.session_manager.get(parent_corr).fork_refcount == 1

        # Parent's terminal turn returns (it declared forks) -> defer eviction.
        parent_terminal = _make_credit(
            x_correlation_id=parent_corr,
            parent_correlation_id=None,
            has_forks=True,
        )
        worker._release_and_evict_for_terminal(parent_terminal, parent_corr)
        parent = worker.session_manager.get(parent_corr)
        assert parent is not None, "pinned FORK parent must survive its terminal turn"
        assert parent.pending_fork_eviction is True

        # Child completes -> releases the pin, which auto-evicts the parent and
        # evicts the child's own (non-fork) session.
        worker._release_and_evict_for_terminal(child_credit, child_corr)
        assert worker.session_manager.get(parent_corr) is None
        assert worker.session_manager.get(child_corr) is None


class TestForkSeededHistoryReachesWire:
    """End-to-end seam test: the FORK-seeded ``turn_list`` must reach the wire.

    Seeding into a ``turn_list`` nobody reads is invisible to the pin/seed
    call-site tests above — this test drives the seeded session's
    ``turn_list`` through a real ``ChatEndpoint.format_payload`` and asserts
    the parent's authored raw_messages + captured assistant reply appear in
    the payload BEFORE the child's own messages (the dag_jsonl FORK
    inheritance contract asserted by ``test_dag_full_topology``).
    """

    def test_fork_child_payload_contains_seeded_parent_history(self) -> None:
        manager = UserSessionManager()

        # Parent: dag_jsonl-style authored raw_messages turn declaring a fork.
        parent_conv = Conversation(
            session_id="root",
            turns=[
                Turn(
                    raw_messages=[
                        {"role": "system", "content": "root system prompt"},
                        {"role": "user", "content": "root user prompt"},
                    ],
                    branch_ids=["root:0"],
                )
            ],
            branches=[
                ConversationBranchInfo(
                    branch_id="root:0",
                    mode=ConversationBranchMode.FORK,
                    child_conversation_ids=["branch-a"],
                )
            ],
        )
        parent = manager.create_and_store(
            x_correlation_id="parent-corr",
            conversation=parent_conv,
            num_turns=1,
        )
        parent.advance_turn(0)
        # Captured live assistant reply (shape produced by build_assistant_turn).
        parent.store_response(
            Turn(role="assistant", texts=[Text(contents=["captured root reply"])])
        )

        # Child: its own authored raw_messages turn.
        child_conv = Conversation(
            session_id="branch-a",
            turns=[
                Turn(
                    raw_messages=[
                        {"role": "user", "content": "branch-a user message A"},
                        {"role": "user", "content": "branch-a user message B"},
                    ]
                )
            ],
        )
        child = manager.create_and_store(
            x_correlation_id="child-corr",
            conversation=child_conv,
            num_turns=1,
        )
        manager.pin_for_fork_child("parent-corr")
        manager.seed_from_parent("child-corr", "parent-corr")
        child.advance_turn(0)

        # Build the wire payload exactly as the worker does: RequestInfo.turns
        # is the session's accumulated turn_list.
        config = create_config(
            EndpointType.CHAT,
            base_url="http://localhost:8000",
            path="/v1/chat/completions",
        )
        endpoint = ChatEndpoint(model_endpoint=_wrap_model_endpoint(config))
        request_info = create_request_info(config=config, turns=child.turn_list)
        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == [
            {"role": "system", "content": "root system prompt"},
            {"role": "user", "content": "root user prompt"},
            {"role": "assistant", "content": "captured root reply"},
            {"role": "user", "content": "branch-a user message A"},
            {"role": "user", "content": "branch-a user message B"},
        ]
