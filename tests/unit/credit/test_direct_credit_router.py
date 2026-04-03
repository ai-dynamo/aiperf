# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DirectCreditRouter.

Focuses on:
- Construction and worker attachment lifecycle
- Direct credit delivery via send_credit / cancel_all_credits
- Callback wiring (before and after worker attachment)
- Credits-complete flag
- End-to-end credit lifecycle with callbacks
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.direct_credit_router import DirectCreditRouter
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit

# ============================================================
# Helpers
# ============================================================


def _make_credit(credit_id: int = 1) -> Credit:
    """Create a minimal Credit for testing."""
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-1",
        x_correlation_id="corr-1",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
    )


def _make_mock_worker(service_id: str = "worker-1") -> MagicMock:
    """Create a mock Worker with the methods DirectCreditRouter requires."""
    worker = MagicMock()
    worker.service_id = service_id
    worker.receive_credit = MagicMock()
    worker.cancel_all_credits = AsyncMock()
    worker.set_credit_return_callback = MagicMock()
    worker.set_first_token_callback = MagicMock()
    return worker


# ============================================================
# Construction & Attachment
# ============================================================


class TestConstructionAndAttachment:
    """Verify worker attachment and worker_id property."""

    def test_attach_worker_stores_reference(self) -> None:
        router = DirectCreditRouter()
        worker = _make_mock_worker()

        router.attach_worker(worker)

        assert router._worker is worker

    def test_worker_id_returns_service_id(self) -> None:
        router = DirectCreditRouter()
        worker = _make_mock_worker(service_id="my-worker-42")
        router.attach_worker(worker)

        assert router.worker_id == "my-worker-42"

    def test_worker_id_raises_without_worker(self) -> None:
        router = DirectCreditRouter()

        with pytest.raises(RuntimeError, match="No worker attached"):
            _ = router.worker_id

    def test_attach_worker_wires_existing_callbacks(self) -> None:
        router = DirectCreditRouter()
        return_cb = AsyncMock()
        first_token_cb = AsyncMock()
        router.set_return_callback(return_cb)
        router.set_first_token_callback(first_token_cb)

        worker = _make_mock_worker()
        router.attach_worker(worker)

        # Return callback is wrapped to inject worker_id, so check it was called with a callable
        worker.set_credit_return_callback.assert_called_once()
        worker.set_first_token_callback.assert_called_once_with(first_token_cb)


# ============================================================
# send_credit
# ============================================================


class TestSendCredit:
    """Verify direct credit delivery to the attached worker."""

    @pytest.mark.asyncio
    async def test_send_credit_calls_worker_receive_credit(self) -> None:
        router = DirectCreditRouter()
        worker = _make_mock_worker()
        router.attach_worker(worker)
        credit = _make_credit()

        await router.send_credit(credit)

        worker.receive_credit.assert_called_once_with(credit)

    @pytest.mark.asyncio
    async def test_send_credit_raises_without_worker(self) -> None:
        router = DirectCreditRouter()

        with pytest.raises(RuntimeError, match="No worker attached"):
            await router.send_credit(_make_credit())


# ============================================================
# cancel_all_credits
# ============================================================


class TestCancelAllCredits:
    """Verify cancel propagation to attached worker."""

    @pytest.mark.asyncio
    async def test_cancel_all_credits_calls_worker(self) -> None:
        router = DirectCreditRouter()
        worker = _make_mock_worker()
        router.attach_worker(worker)

        await router.cancel_all_credits()

        worker.cancel_all_credits.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancel_all_credits_raises_without_worker(self) -> None:
        router = DirectCreditRouter()

        with pytest.raises(RuntimeError, match="No worker attached"):
            await router.cancel_all_credits()


# ============================================================
# mark_credits_complete
# ============================================================


class TestMarkCreditsComplete:
    """Verify credits-complete flag."""

    def test_mark_credits_complete_sets_flag(self) -> None:
        router = DirectCreditRouter()

        assert router._credits_complete is False
        router.mark_credits_complete()
        assert router._credits_complete is True


# ============================================================
# Callback Wiring
# ============================================================


class TestCallbackWiring:
    """Verify callback registration before and after worker attachment."""

    def test_set_return_callback_before_attach_worker(self) -> None:
        router = DirectCreditRouter()
        cb = AsyncMock()

        router.set_return_callback(cb)

        assert router._on_return_callback is cb

    def test_set_return_callback_after_attach_worker(self) -> None:
        router = DirectCreditRouter()
        worker = _make_mock_worker()
        router.attach_worker(worker)
        cb = AsyncMock()

        router.set_return_callback(cb)

        assert router._on_return_callback is cb
        # Return callback is wrapped to inject worker_id
        worker.set_credit_return_callback.assert_called_once()

    def test_set_first_token_callback_before_attach_worker(self) -> None:
        router = DirectCreditRouter()
        cb = AsyncMock()

        router.set_first_token_callback(cb)

        assert router._on_first_token_callback is cb

    def test_set_first_token_callback_after_attach_worker(self) -> None:
        router = DirectCreditRouter()
        worker = _make_mock_worker()
        router.attach_worker(worker)
        cb = AsyncMock()

        router.set_first_token_callback(cb)

        assert router._on_first_token_callback is cb
        worker.set_first_token_callback.assert_called_once_with(cb)

    def test_callbacks_wired_to_worker_on_attach(self) -> None:
        router = DirectCreditRouter()
        return_cb = AsyncMock()
        first_token_cb = AsyncMock()
        router.set_return_callback(return_cb)
        router.set_first_token_callback(first_token_cb)

        worker = _make_mock_worker()
        router.attach_worker(worker)

        # Return callback is wrapped to inject worker_id
        worker.set_credit_return_callback.assert_called_once()
        worker.set_first_token_callback.assert_called_once_with(first_token_cb)

    def test_no_callbacks_wired_when_none_registered(self) -> None:
        router = DirectCreditRouter()
        worker = _make_mock_worker()

        router.attach_worker(worker)

        worker.set_credit_return_callback.assert_not_called()
        worker.set_first_token_callback.assert_not_called()


# ============================================================
# Integration Flow
# ============================================================


class TestIntegrationFlow:
    """Verify end-to-end credit lifecycle through the router."""

    @pytest.mark.asyncio
    async def test_full_credit_lifecycle_send_and_return(self) -> None:
        """Send credit, simulate worker processing, CreditReturn flows back via callback."""
        router = DirectCreditRouter()
        received_returns: list[tuple[str, CreditReturn]] = []

        async def on_return(worker_id: str, msg: CreditReturn) -> None:
            received_returns.append((worker_id, msg))

        router.set_return_callback(on_return)

        # Build a worker mock that invokes the return callback when it receives a credit
        worker = _make_mock_worker(service_id="w-1")
        stored_return_cb: list[Callable] = []

        def capture_return_cb(cb: Callable[[CreditReturn], Awaitable[None]]) -> None:
            stored_return_cb.append(cb)

        worker.set_credit_return_callback = MagicMock(side_effect=capture_return_cb)
        router.attach_worker(worker)

        credit = _make_credit(credit_id=42)
        await router.send_credit(credit)
        worker.receive_credit.assert_called_once_with(credit)

        # Simulate the worker firing the wrapped return callback (worker calls with just credit_return)
        assert len(stored_return_cb) == 1
        credit_return = CreditReturn(credit=credit, first_token_sent=True)
        await stored_return_cb[0](credit_return)

        # The wrapper injects worker_id automatically
        assert len(received_returns) == 1
        assert received_returns[0] == ("w-1", credit_return)

    @pytest.mark.asyncio
    async def test_full_credit_lifecycle_with_first_token(self) -> None:
        """Send credit, FirstToken flows back via callback."""
        router = DirectCreditRouter()
        received_tokens: list[FirstToken] = []

        async def on_first_token(msg: FirstToken) -> None:
            received_tokens.append(msg)

        router.set_first_token_callback(on_first_token)

        worker = _make_mock_worker(service_id="w-2")
        stored_ft_cb: list[Callable] = []

        def capture_ft_cb(cb: Callable[[FirstToken], Awaitable[None]]) -> None:
            stored_ft_cb.append(cb)

        worker.set_first_token_callback = MagicMock(side_effect=capture_ft_cb)
        router.attach_worker(worker)

        credit = _make_credit(credit_id=99)
        await router.send_credit(credit)

        # Simulate the worker firing the first-token callback
        assert len(stored_ft_cb) == 1
        ft = FirstToken(credit_id=99, phase=CreditPhase.PROFILING, ttft_ns=5_000_000)
        await stored_ft_cb[0](ft)

        assert len(received_tokens) == 1
        assert received_tokens[0] is ft
