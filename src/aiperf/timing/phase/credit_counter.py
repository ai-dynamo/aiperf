# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credit counter for lock-free credit tracking.

Provides lock-free operations for credit counting via asyncio serialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.credit.structs import TurnToSend
    from aiperf.timing.config import CreditPhaseConfig


class CreditCounter:
    """Lock-free credit counting via asyncio serialization.

    Tracks credits sent, completed, cancelled, and session counts.
    All public methods are atomic (non-async).

    CRITICAL: All functions must be non-async - would break atomicity.
    """

    def __init__(self, config: CreditPhaseConfig) -> None:
        self._config = config

        # Progress counters
        self._requests_sent: int = 0
        self._requests_completed: int = 0
        self._requests_cancelled: int = 0
        self._request_errors: int = 0
        self._sent_sessions: int = 0
        self._completed_sessions: int = 0
        self._cancelled_sessions: int = 0
        self._total_session_turns: int = 0
        self._prefills_released: int = 0  # TTFTs received + returns without TTFT

        # Final count fields (frozen when phase transitions)
        self._final_requests_sent: int | None = None
        self._final_requests_completed: int | None = None
        self._final_requests_cancelled: int | None = None
        self._final_request_errors: int | None = None
        self._final_sent_sessions: int | None = None
        self._final_completed_sessions: int | None = None
        self._final_cancelled_sessions: int | None = None

    # =========================================================================
    # Properties (read-only access to counters)
    # =========================================================================

    @property
    def requests_sent(self) -> int:
        """Total requests sent."""
        return self._requests_sent

    @property
    def requests_completed(self) -> int:
        """Total requests completed successfully."""
        return self._requests_completed

    @property
    def requests_cancelled(self) -> int:
        """Total requests cancelled."""
        return self._requests_cancelled

    @property
    def request_errors(self) -> int:
        """Total request errors."""
        return self._request_errors

    @property
    def sent_sessions(self) -> int:
        """Total sessions (conversations) started."""
        return self._sent_sessions

    @property
    def completed_sessions(self) -> int:
        """Total sessions (conversations) completed successfully."""
        return self._completed_sessions

    @property
    def cancelled_sessions(self) -> int:
        """Total sessions cancelled (final turn was cancelled)."""
        return self._cancelled_sessions

    @property
    def total_session_turns(self) -> int:
        """Total turns across all started sessions."""
        return self._total_session_turns

    @property
    def in_flight_sessions(self) -> int:
        """Sessions started but not yet finished (no final turn returned)."""
        return self._sent_sessions - self._completed_sessions - self._cancelled_sessions

    @property
    def in_flight(self) -> int:
        """Number of in-flight credits (sent but not yet returned)."""
        return self._requests_sent - self._requests_completed - self._requests_cancelled

    @property
    def prefills_released(self) -> int:
        """Prefill slots released (TTFT received or returned without TTFT)."""
        return self._prefills_released

    @property
    def in_flight_prefills(self) -> int:
        """Requests sent but prefill not yet complete (TTFT not received)."""
        return self._requests_sent - self._prefills_released

    # =========================================================================
    # Final count properties (frozen values)
    # =========================================================================

    @property
    def final_requests_sent(self) -> int | None:
        """Final sent count (frozen when sending completes)."""
        return self._final_requests_sent

    @property
    def final_requests_completed(self) -> int | None:
        """Final completed count (frozen when phase completes)."""
        return self._final_requests_completed

    @property
    def final_requests_cancelled(self) -> int | None:
        """Final cancelled count (frozen when phase completes)."""
        return self._final_requests_cancelled

    @property
    def final_request_errors(self) -> int | None:
        """Final error count (frozen when phase completes)."""
        return self._final_request_errors

    @property
    def final_sent_sessions(self) -> int | None:
        """Final sent sessions count (frozen when sending completes)."""
        return self._final_sent_sessions

    @property
    def final_completed_sessions(self) -> int | None:
        """Final completed sessions count (frozen when phase completes)."""
        return self._final_completed_sessions

    @property
    def final_cancelled_sessions(self) -> int | None:
        """Final cancelled sessions count (frozen when phase completes)."""
        return self._final_cancelled_sessions

    # =========================================================================
    # Freezing Methods (called by PhaseTracker at phase transitions)
    # =========================================================================

    def freeze_sent_counts(self) -> None:
        """Freeze sent counts (called when sending completes)."""
        self._final_requests_sent = self._requests_sent
        self._final_sent_sessions = self._sent_sessions

    def freeze_completed_counts(self) -> None:
        """Freeze completed counts (called when phase completes)."""
        self._final_requests_completed = self._requests_completed
        self._final_completed_sessions = self._completed_sessions
        self._final_cancelled_sessions = self._cancelled_sessions
        self._final_requests_cancelled = self._requests_cancelled
        self._final_request_errors = self._request_errors

    # =========================================================================
    # Atomic Operations (lock-free - no await between read and write)
    # =========================================================================

    def increment_sent(self, turn_to_send: TurnToSend) -> tuple[int, bool]:
        """Atomically increment sent count and return (credit_index, is_final_credit).

        DAG children (``turn_to_send.agent_depth > 0``) count as real HTTP
        requests and DO bump ``_requests_sent`` — the user-visible
        "requests sent" metric must reflect actual wire traffic including
        DAG offspring. They do NOT bump ``_sent_sessions`` or
        ``_total_session_turns`` because they inherit the parent's
        session slot (``CreditIssuer.issue_credit`` skips session-slot
        acquisition for them). Children also never flip
        ``is_final_credit`` — the ``TimingStrategy`` loop's "sending
        complete" signal is root-plan-driven, not wire-volume-driven.

        Lock-free: no async calls.
        """
        credit_index = self._requests_sent
        new_sent_count = self._requests_sent + 1

        if turn_to_send.agent_depth > 0:
            # Children: bump request count only (observability), leave
            # session counters alone (slot is inherited), never signal
            # plan exhaustion.
            self._requests_sent = new_sent_count
            return credit_index, False

        new_sent_sessions_count = self._sent_sessions
        new_total_session_turns = self._total_session_turns

        if turn_to_send.turn_index == 0:
            new_sent_sessions_count += 1
            new_total_session_turns += turn_to_send.num_turns

        is_final_credit = (
            self._config.total_expected_requests is not None
            and new_sent_count >= self._config.total_expected_requests
        ) or (
            self._config.expected_num_sessions is not None
            and new_sent_sessions_count >= self._config.expected_num_sessions
            and new_sent_count >= new_total_session_turns
        )

        self._requests_sent = new_sent_count
        self._sent_sessions = new_sent_sessions_count
        self._total_session_turns = new_total_session_turns

        return credit_index, is_final_credit

    def increment_returned(
        self,
        is_final_turn: bool,
        cancelled: bool,
        *,
        is_child: bool = False,
    ) -> bool:
        """Atomically increment returned count and check phase completion.

        DAG children DO bump ``_requests_completed`` / ``_requests_cancelled``
        — these are user-visible metrics of actual HTTP activity, symmetric
        with ``_requests_sent`` being bumped on the dispatch side. They do
        NOT bump ``_completed_sessions`` / ``_cancelled_sessions`` (session
        slot was inherited, not acquired).

        Lock-free: no async calls.

        Args:
            is_final_turn: Whether the returned turn is the final turn of its session
            cancelled: Whether the credit was cancelled
            is_child: True when ``credit.agent_depth > 0``. Session-level
                counters are skipped for children; request-level counters
                still tick.

        Returns:
            True if ALL sent credits have now been returned or cancelled
            (phase sending must be complete for this to ever return True).
            The DAG deferral lives in ``CreditCallbackHandler``: even when
            this returns True the completion event is held until
            ``BranchOrchestrator.has_pending_branch_work`` drains.
        """
        if cancelled:
            self._requests_cancelled += 1
            if is_final_turn and not is_child:
                self._cancelled_sessions += 1
        else:
            self._requests_completed += 1
            if is_final_turn and not is_child:
                self._completed_sessions += 1

        return self.check_all_returned_or_cancelled()

    def check_all_returned_or_cancelled(self) -> bool:
        """True if all sent credits have been returned or cancelled."""
        if self._final_requests_sent is None:
            return False
        return (
            self._requests_completed + self._requests_cancelled
        ) >= self._final_requests_sent

    def increment_prefill_released(self) -> None:
        """Increment prefill released count (on TTFT or return without TTFT)."""
        self._prefills_released += 1
