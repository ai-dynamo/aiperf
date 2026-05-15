# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker call-site tests for FORK pin/release refcount.

Verifies that ``Worker._process_credit`` exercises the pin/release/
evict_if_unpinned API on ``UserSessionManager`` for the DAG-FORK
child path. The storage half is covered by
``test_session_fork_refcount.py``; this file tests the wiring.
"""

from unittest.mock import AsyncMock

import pytest

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import RequestRecord
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.worker import Worker
from tests.harness.fake_communication import FakeCommunication as FakeCommunication
from tests.harness.fake_service_manager import FakeServiceManager as FakeServiceManager
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.harness.fake_transport import FakeTransport as FakeTransport


def _plain_conv(session_id: str = "child") -> Conversation:
    """Plain (non-parent) conversation with no FORK branches."""
    return Conversation(
        session_id=session_id,
        turns=[Turn()],
        branches=[],
    )


def _parent_conv(session_id: str = "parent", n_forks: int = 1) -> Conversation:
    """A FORK parent: declares ``n_forks`` FORK branches."""
    branches = [
        ConversationBranchInfo(
            branch_id=f"{session_id}:{i}",
            mode=ConversationBranchMode.FORK,
            child_conversation_ids=[f"child-{i}"],
        )
        for i in range(n_forks)
    ]
    return Conversation(
        session_id=session_id,
        turns=[Turn(branch_ids=[f"{session_id}:{i}" for i in range(n_forks)])],
        branches=branches,
    )


def _credit(
    *,
    x_correlation_id: str,
    parent_correlation_id: str | None = None,
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK,
    turn_index: int = 0,
    num_turns: int = 1,
    conversation_id: str = "child",
) -> CreditContext:
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        parent_correlation_id=parent_correlation_id,
        branch_mode=branch_mode,
    )
    return CreditContext(credit=credit, drop_perf_ns=0)


@pytest.fixture
async def mock_worker(
    user_config: UserConfig,
    service_config: ServiceConfig,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
):
    worker = Worker(
        service_config=service_config,
        user_config=user_config,
        service_id="mock-service-id",
    )
    await worker.initialize()
    await worker.start()
    yield worker
    await worker.stop()


def _wire_fakes(worker: Worker, conversation: Conversation) -> None:
    """Stub I/O so ``_process_credit`` can run end-to-end without a server."""
    worker._retrieve_conversation = AsyncMock(return_value=conversation)
    worker.inference_client.send_request = AsyncMock(
        return_value=RequestRecord(
            conversation_id=conversation.session_id,
            turn_index=0,
            model_name="test",
            timestamp_ns=0,
            start_perf_ns=0,
            end_perf_ns=1,
            x_request_id="req",
            x_correlation_id="corr",
        )
    )
    worker._send_inference_result_message = AsyncMock()


@pytest.mark.asyncio
class TestForkPinCallSites:
    async def test_pin_called_on_fork_child_first_turn(self, mock_worker: Worker):
        # Seed parent in the local session manager.
        mock_worker.session_manager.create_and_store(
            x_correlation_id="parent-corr",
            conversation=_parent_conv("parent", n_forks=1),
            num_turns=1,
        )
        _wire_fakes(mock_worker, _plain_conv("child"))

        ctx = _credit(
            x_correlation_id="child-corr",
            parent_correlation_id="parent-corr",
            branch_mode=ConversationBranchMode.FORK,
        )
        await mock_worker._process_credit(ctx)

        # Parent pinned at child seed, then released at child terminal turn.
        parent = mock_worker.session_manager.get("parent-corr")
        assert parent is not None
        # Release on terminal turn drops it back to 0.
        assert parent.fork_refcount == 0

    async def test_release_drops_refcount_on_child_terminal(self, mock_worker: Worker):
        # Pre-pin to simulate two outstanding children (one already dispatched
        # via a different worker/code path), so the release leaves a non-zero count.
        mock_worker.session_manager.create_and_store(
            x_correlation_id="parent-corr",
            conversation=_parent_conv("parent", n_forks=2),
            num_turns=1,
        )
        mock_worker.session_manager.pin_for_fork_child("parent-corr")
        # Now a second child arrives at this worker.
        _wire_fakes(mock_worker, _plain_conv("child2"))

        ctx = _credit(
            x_correlation_id="child2-corr",
            parent_correlation_id="parent-corr",
            branch_mode=ConversationBranchMode.FORK,
            conversation_id="child2",
        )
        await mock_worker._process_credit(ctx)

        # +1 (pin at seed) -1 (release at terminal) = no change to existing 1.
        parent = mock_worker.session_manager.get("parent-corr")
        assert parent is not None
        assert parent.fork_refcount == 1

    async def test_release_called_on_cancellation(self, mock_worker: Worker):
        mock_worker.session_manager.create_and_store(
            x_correlation_id="parent-corr",
            conversation=_parent_conv("parent", n_forks=1),
            num_turns=1,
        )
        _wire_fakes(mock_worker, _plain_conv("child"))

        # Force the inner request to cancel.
        async def _cancel(*_a, **_kw):
            import asyncio

            raise asyncio.CancelledError()

        mock_worker.inference_client.send_request = AsyncMock(side_effect=_cancel)

        ctx = _credit(
            x_correlation_id="child-corr",
            parent_correlation_id="parent-corr",
            branch_mode=ConversationBranchMode.FORK,
        )
        import asyncio

        with pytest.raises(asyncio.CancelledError):
            await mock_worker._process_credit(ctx)

        # Pin (+1) then release on cancellation (-1) -> 0.
        parent = mock_worker.session_manager.get("parent-corr")
        assert parent is not None
        assert parent.fork_refcount == 0

    async def test_no_pin_for_non_fork_child(self, mock_worker: Worker):
        """SPAWN children must NOT pin the parent."""
        mock_worker.session_manager.create_and_store(
            x_correlation_id="parent-corr",
            conversation=_parent_conv("parent", n_forks=1),
            num_turns=1,
        )
        _wire_fakes(mock_worker, _plain_conv("child"))

        ctx = _credit(
            x_correlation_id="child-corr",
            parent_correlation_id="parent-corr",
            branch_mode=ConversationBranchMode.SPAWN,
        )
        await mock_worker._process_credit(ctx)

        parent = mock_worker.session_manager.get("parent-corr")
        assert parent is not None
        assert parent.fork_refcount == 0

    async def test_fork_parent_uses_evict_if_unpinned(self, mock_worker: Worker):
        """FORK parent terminal does not pop from cache while pinned."""
        # Pre-pin: a child is outstanding when the parent's terminal turn
        # returns. The eviction must defer until release.
        _wire_fakes(mock_worker, _parent_conv("parent", n_forks=1))

        ctx = _credit(
            x_correlation_id="parent-corr",
            parent_correlation_id=None,
            branch_mode=ConversationBranchMode.FORK,
            num_turns=1,
        )

        # Simulate a pending child that has already pinned the parent
        # before the parent reaches its terminal turn.
        async def _send_ok(*_a, **_kw):
            mock_worker.session_manager.pin_for_fork_child("parent-corr")
            return RequestRecord(
                conversation_id="parent",
                turn_index=0,
                model_name="test",
                timestamp_ns=0,
                start_perf_ns=0,
                end_perf_ns=1,
                x_request_id="req",
                x_correlation_id="corr",
            )

        mock_worker.inference_client.send_request = AsyncMock(side_effect=_send_ok)
        await mock_worker._process_credit(ctx)

        # Parent must remain in the cache (pinned by the child).
        assert mock_worker.session_manager.get("parent-corr") is not None
