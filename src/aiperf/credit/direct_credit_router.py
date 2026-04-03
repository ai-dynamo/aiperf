# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero-overhead credit router for direct worker mode.

Delivers credits directly to a co-located Worker instance via method calls
instead of ZMQ sockets. Designed for single-worker benchmarks where the
Worker lives in TimingManager's process.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit

if TYPE_CHECKING:
    from aiperf.workers.worker import Worker


class DirectCreditRouter:
    """Zero-overhead credit router for direct worker mode.

    Implements CreditRouterProtocol by delivering credits directly to a
    co-located Worker instance via method calls instead of ZMQ sockets.
    Designed for single-worker benchmarks where the Worker lives in
    TimingManager's process.
    """

    def __init__(self) -> None:
        self._worker: Worker | None = None
        self._credits_complete: bool = False
        self._in_flight_credits: set[int] = set()
        self._on_return_callback: (
            Callable[[str, CreditReturn], Awaitable[None]] | None
        ) = None
        self._on_first_token_callback: (
            Callable[[FirstToken], Awaitable[None]] | None
        ) = None

    @property
    def worker_id(self) -> str:
        """Return the attached worker's service_id."""
        if self._worker is None:
            raise RuntimeError("No worker attached to DirectCreditRouter")
        return self._worker.service_id

    def attach_worker(self, worker: Worker) -> None:
        """Connect a worker instance and wire callbacks.

        Registers the stored return/first-token callbacks on the worker so
        that CreditReturn and FirstToken events flow back to the caller.

        Args:
            worker: Worker instance to deliver credits to.
        """
        self._worker = worker
        if self._on_return_callback is not None:
            worker.set_credit_return_callback(
                self._wrap_return_callback(self._on_return_callback)
            )
        if self._on_first_token_callback is not None:
            worker.set_first_token_callback(self._on_first_token_callback)

    def _wrap_return_callback(
        self,
        callback: Callable[[str, CreditReturn], Awaitable[None]],
    ) -> Callable[[CreditReturn], Awaitable[None]]:
        """Wrap a (worker_id, credit_return) callback into a (credit_return) callback.

        The orchestrator's callback expects (worker_id, credit_return), but the
        Worker calls its callback with just (credit_return). This wrapper injects
        the worker_id automatically.
        """
        worker_id = self.worker_id
        in_flight = self._in_flight_credits

        async def wrapped(credit_return: CreditReturn) -> None:
            in_flight.discard(credit_return.credit.id)
            await callback(worker_id, credit_return)

        return wrapped

    def set_return_callback(
        self,
        callback: Callable[[str, CreditReturn], Awaitable[None]],
    ) -> None:
        """Register callback for credit returns.

        Args:
            callback: Async function called when credit returns.
                     Signature: (worker_id: str, message: CreditReturn) -> None
        """
        self._on_return_callback = callback
        if self._worker is not None:
            self._worker.set_credit_return_callback(
                self._wrap_return_callback(callback)
            )

    def set_first_token_callback(
        self,
        callback: Callable[[FirstToken], Awaitable[None]],
    ) -> None:
        """Register callback for first token events (prefill concurrency release).

        Args:
            callback: Async function called when first token is received.
                     Signature: (message: FirstToken) -> None
        """
        self._on_first_token_callback = callback
        if self._worker is not None:
            self._worker.set_first_token_callback(callback)

    async def send_credit(self, credit: Credit) -> None:
        """Send credit directly to the attached worker.

        Args:
            credit: Credit to deliver to the worker.

        Raises:
            RuntimeError: If no worker is attached.
        """
        if self._worker is None:
            raise RuntimeError("No worker attached to DirectCreditRouter")
        self._in_flight_credits.add(credit.id)
        self._worker.receive_credit(credit)

    async def cancel_all_credits(self) -> None:
        """Cancel all in-flight credits on the attached worker.

        Raises:
            RuntimeError: If no worker is attached.
        """
        if self._worker is None:
            raise RuntimeError("No worker attached to DirectCreditRouter")
        if self._in_flight_credits:
            await self._worker.cancel_credits(self._in_flight_credits.copy())

    def mark_credits_complete(self) -> None:
        """Mark that all credits have been issued and returned.

        Called by orchestrator when benchmark completes normally.
        """
        self._credits_complete = True
