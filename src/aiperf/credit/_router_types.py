# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared types for the sticky credit router.

Split out of ``sticky_router.py`` to keep that module under the file-size
ergonomics limit. The public re-exports live in ``sticky_router``; new
callers should keep importing from there.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit


@dataclass(slots=True)
class WorkerLoad:
    """Worker load tracking for fair load balancing.

    Note on virtual_sent_credits vs total_sent_credits:
        - total_sent_credits: Actual count of credits sent (for metrics/debugging)
        - virtual_sent_credits: Used for fairness tie-breaking, initialized to
          average when worker joins mid-benchmark to prevent "thundering herd"
          where a new worker with 0 credits gets all requests.

    Note on active_sessions:
        - active_sessions and active_session_ids only represent the number of sticky sessions assigned
          to the worker, which inherently means that it only tracks sessions with MORE turns left. This is
          because sticky sessions are only created when more than 1 turn exists, and are removed when SENDING the final turn.
    """

    worker_id: str
    """Unique identifier for this worker."""

    total_sent_credits: int = 0
    """Actual count of credits sent to this worker."""

    virtual_sent_credits: int = 0
    """Fairness-adjusted credit count, initialized to average on late join."""

    total_completed_credits: int = 0
    """Number of credits that completed successfully."""

    total_cancelled_credits: int = 0
    """Number of credits that were cancelled."""

    total_errors_reported: int = 0
    """Number of credits that reported errors."""

    in_flight_credits: int = 0
    """Credits currently being processed by this worker."""

    active_credit_ids: set[int] = field(default_factory=set)
    """Set of credit IDs currently in flight."""

    active_credits: dict[int, Credit] = field(default_factory=dict)
    """Map of credit ID to Credit for in-flight credits."""

    active_sessions: int = 0
    """Number of sticky multi-turn sessions assigned to this worker."""

    active_session_ids: set[str] = field(default_factory=set)
    """Set of x_correlation_ids for active sticky sessions."""

    last_sent_at_ns: int = 0
    """Monotonic timestamp of last credit send, used for LRU tie-breaking."""


@dataclass(slots=True, frozen=True)
class UnavailableSession:
    """A session whose sticky worker became unavailable before the next turn."""

    worker_id: str
    """ID of the worker that became unavailable."""

    reason: str
    """Human-readable reason the worker became unavailable."""


@runtime_checkable
class CreditRouterProtocol(Protocol):
    """Protocol for routing credits to workers.

    Decouples credit issuing strategies from routing implementation.
    Enables mocking for tests and alternative routing strategies.
    """

    async def send_credit(self, credit: Credit) -> None:
        """Send credit to worker via routing strategy.

        Args:
            credit: Credit to send to worker
        """
        ...

    async def cancel_all_credits(self) -> None:
        """Cancel all in-flight credits.

        Used during phase timeout or system shutdown.
        """
        ...

    def mark_credits_complete(self) -> None:
        """Mark that all credits have been issued and returned.

        Called by orchestrator when benchmark completes normally.
        Suppresses warnings about orphaned sessions during shutdown.
        """
        ...

    def set_return_callback(
        self,
        callback: Callable[[str, CreditReturn], Awaitable[None]],
    ) -> None:
        """Register callback for credit returns.

        Args:
            callback: Async function called when credit returns.
                     Signature: (worker_id: str, message: CreditReturn) -> None
        """
        ...

    def set_first_token_callback(
        self,
        callback: Callable[[FirstToken], Awaitable[None]],
    ) -> None:
        """Register callback for first token events (prefill concurrency release).

        Args:
            callback: Async function called when first token is received.
                     Signature: (message: FirstToken) -> None
        """
        ...
