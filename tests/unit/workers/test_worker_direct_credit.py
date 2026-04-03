# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Worker direct credit delivery methods.

Focuses on:
- receive_credit() and cancel_credits() public API
- Callback setter methods (set_credit_return_callback, set_first_token_callback)
- _send_credit_return routing (callback vs ZMQ)
- _send_first_token routing (callback vs ZMQ)
- End-to-end callback wiring during credit processing
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    Conversation,
    RequestRecord,
    SSEMessage,
    Text,
    Turn,
)
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit
from aiperf.workers.worker import Worker
from tests.harness.fake_tokenizer import FakeTokenizer

# ============================================================
# Fixtures
# ============================================================


def _make_credit(
    *,
    credit_id: int = 1,
    phase: CreditPhase = CreditPhase.PROFILING,
    conversation_id: str = "conv-1",
    x_correlation_id: str = "corr-1",
    turn_index: int = 0,
    num_turns: int = 1,
    issued_at_ns: int = 1_000_000,
) -> Credit:
    """Helper to build a Credit with sensible defaults."""
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=issued_at_ns,
    )


@pytest.fixture
async def worker(
    user_config: UserConfig,
    service_config: ServiceConfig,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
) -> Worker:
    """Create a fully initialized and started Worker."""
    w = Worker(
        service_config=service_config,
        user_config=user_config,
        service_id="test-worker",
    )
    await w.initialize()
    await w.start()
    yield w
    await w.stop()


# ============================================================
# receive_credit
# ============================================================


@pytest.mark.asyncio
class TestReceiveCredit:
    """Verify receive_credit delegates to the internal scheduling path."""

    async def test_receive_credit_calls_schedule_credit_drop_task(
        self, worker: Worker, monkeypatch
    ) -> None:
        """receive_credit() should delegate to _schedule_credit_drop_task."""
        mock_schedule = Mock()
        monkeypatch.setattr(worker, "_schedule_credit_drop_task", mock_schedule)

        credit = _make_credit()
        worker.receive_credit(credit)

        mock_schedule.assert_called_once_with(credit)

    async def test_receive_credit_with_valid_credit_struct(
        self, worker: Worker, monkeypatch
    ) -> None:
        """receive_credit() passes the exact Credit struct through to scheduling."""
        captured = []
        monkeypatch.setattr(
            worker, "_schedule_credit_drop_task", lambda c: captured.append(c)
        )

        credit = _make_credit(credit_id=42, conversation_id="special-conv")
        worker.receive_credit(credit)

        assert len(captured) == 1
        assert captured[0].id == 42
        assert captured[0].conversation_id == "special-conv"


# ============================================================
# cancel_credits
# ============================================================


@pytest.mark.asyncio
class TestCancelCredits:
    """Verify cancel_credits creates CancelCredits and delegates."""

    async def test_cancel_credits_creates_cancel_message_and_delegates(
        self, worker: Worker, monkeypatch
    ) -> None:
        """cancel_credits() should build a CancelCredits and call _on_cancel_credits_message."""
        mock_handler = AsyncMock()
        monkeypatch.setattr(worker, "_on_cancel_credits_message", mock_handler)

        await worker.cancel_credits({10, 20, 30})

        mock_handler.assert_called_once()
        msg = mock_handler.call_args[0][0]
        assert msg.credit_ids == {10, 20, 30}


# ============================================================
# Callback setters
# ============================================================


@pytest.mark.asyncio
class TestCallbackSetters:
    """Verify callback setter methods store the callback on the worker."""

    async def test_set_credit_return_callback_stores_callback(
        self, worker: Worker
    ) -> None:
        callback = AsyncMock()
        worker.set_credit_return_callback(callback)
        assert worker._credit_return_callback is callback

    async def test_set_first_token_callback_stores_callback(
        self, worker: Worker
    ) -> None:
        callback = AsyncMock()
        worker.set_first_token_callback(callback)
        assert worker._first_token_callback is callback


# ============================================================
# _send_credit_return routing
# ============================================================


@pytest.mark.asyncio
class TestSendCreditReturnRouting:
    """Verify _send_credit_return routes to callback or ZMQ."""

    async def test_send_credit_return_uses_callback_when_set(
        self, worker: Worker
    ) -> None:
        """When a callback is registered, _send_credit_return should invoke it."""
        callback = AsyncMock()
        worker.set_credit_return_callback(callback)

        credit_return = CreditReturn(credit=_make_credit())
        await worker._send_credit_return(credit_return)

        callback.assert_awaited_once_with(credit_return)

    async def test_send_credit_return_uses_zmq_when_no_callback(
        self, worker: Worker, monkeypatch
    ) -> None:
        """When no callback is registered, _send_credit_return should use the DEALER socket."""
        mock_send = AsyncMock()
        monkeypatch.setattr(worker.credit_dealer_client, "send", mock_send)

        credit_return = CreditReturn(credit=_make_credit())
        await worker._send_credit_return(credit_return)

        mock_send.assert_awaited_once_with(credit_return)

    async def test_send_credit_return_awaits_callback(self, worker: Worker) -> None:
        """The callback should be properly awaited (not just called)."""
        was_awaited = False

        async def slow_callback(cr: CreditReturn) -> None:
            nonlocal was_awaited
            await asyncio.sleep(0)
            was_awaited = True

        worker.set_credit_return_callback(slow_callback)

        await worker._send_credit_return(CreditReturn(credit=_make_credit()))
        assert was_awaited


# ============================================================
# _send_first_token routing
# ============================================================


@pytest.mark.asyncio
class TestSendFirstTokenRouting:
    """Verify _send_first_token routes to callback or ZMQ."""

    async def test_send_first_token_uses_callback_when_set(
        self, worker: Worker
    ) -> None:
        """When a callback is registered, _send_first_token should invoke it."""
        callback = AsyncMock()
        worker.set_first_token_callback(callback)

        first_token = FirstToken(credit_id=1, phase=CreditPhase.PROFILING, ttft_ns=500)
        await worker._send_first_token(first_token)

        callback.assert_awaited_once_with(first_token)

    async def test_send_first_token_uses_zmq_when_no_callback(
        self, worker: Worker, monkeypatch
    ) -> None:
        """When no callback is registered, _send_first_token should use the DEALER socket."""
        mock_send = AsyncMock()
        monkeypatch.setattr(worker.credit_dealer_client, "send", mock_send)

        first_token = FirstToken(credit_id=1, phase=CreditPhase.PROFILING, ttft_ns=500)
        await worker._send_first_token(first_token)

        mock_send.assert_awaited_once_with(first_token)


# ============================================================
# Integration: callback wired correctly through credit flow
# ============================================================


@pytest.mark.asyncio
class TestCreditReturnCallbackIntegration:
    """Verify credit return callback is invoked during actual credit processing."""

    async def test_credit_return_flows_through_callback_after_processing(
        self, worker: Worker, monkeypatch
    ) -> None:
        """After processing a credit, the CreditReturn should be delivered via callback."""
        returned = []
        callback = AsyncMock(side_effect=lambda cr: returned.append(cr))
        worker.set_credit_return_callback(callback)

        # Engine is ready
        worker._engine_ready.set()

        # Stub out _process_credit so the full pipeline completes without needing
        # real inference clients, dataset clients, etc.
        monkeypatch.setattr(worker, "_process_credit", AsyncMock())

        credit = _make_credit(credit_id=99)
        worker.receive_credit(credit)

        # Allow the scheduled task to run
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(returned) == 1
        assert returned[0].credit.id == 99
        assert not returned[0].cancelled

    async def test_first_token_flows_through_callback_during_streaming(
        self, worker: Worker, monkeypatch
    ) -> None:
        """When prefill_concurrency is enabled and streaming yields content,
        the FirstToken callback should fire during _process_credit."""
        first_tokens = []
        ft_callback = AsyncMock(side_effect=lambda ft: first_tokens.append(ft))
        cr_callback = AsyncMock()
        worker.set_first_token_callback(ft_callback)
        worker.set_credit_return_callback(cr_callback)

        # Enable prefill concurrency so first_token_callback is created
        worker._prefill_concurrency_enabled = True

        # Engine is ready
        worker._engine_ready.set()

        # We need _process_credit to actually run (not be mocked) so the
        # first_token_callback closure is exercised.  But we need to mock the
        # deeper dependencies.

        # Mock session manager to return a session
        mock_session = Mock()
        mock_session.x_correlation_id = "corr-1"
        mock_session.turn_index = 0
        mock_session.turn_list = [Turn(role="user", texts=[Text(contents=["hello"])])]
        mock_session.conversation = Conversation(session_id="conv-1", turns=[])
        mock_session.url_index = None
        mock_session.advance_turn = Mock()
        mock_session.store_response = Mock()
        monkeypatch.setattr(
            worker.session_manager, "get", Mock(return_value=mock_session)
        )

        # Mock inference_client.send_request to simulate a streaming request
        # that invokes the first_token_callback
        async def mock_send_request(request_info, first_token_callback=None):
            if first_token_callback is not None:
                # Simulate receiving a meaningful SSE chunk
                msg = SSEMessage(perf_ns=100_000)
                # The callback checks endpoint.parse_response — mock it
                await first_token_callback(50_000, msg)
            return RequestRecord()

        monkeypatch.setattr(worker.inference_client, "send_request", mock_send_request)

        # Mock endpoint.parse_response to return meaningful content
        mock_endpoint = Mock()
        mock_endpoint.parse_response = Mock(
            return_value=Mock(data=Mock())  # non-None data
        )
        mock_endpoint.extract_response_data = Mock(return_value=[])
        monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)

        # Mock _send_inference_result_message
        monkeypatch.setattr(worker, "_send_inference_result_message", AsyncMock())

        credit = _make_credit(credit_id=77)
        worker.receive_credit(credit)

        # Allow the scheduled task to run
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(first_tokens) == 1
        assert first_tokens[0].credit_id == 77
        assert first_tokens[0].phase == CreditPhase.PROFILING
        assert first_tokens[0].ttft_ns == 50_000
