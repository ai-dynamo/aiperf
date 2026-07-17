# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credit counter for lock-free credit tracking.

Provides lock-free operations for credit counting via asyncio serialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.plugin.enums import TimingMode

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
        # Graph-IR phases bypass the linear session arithmetic on BOTH sides:
        # ``increment_sent`` detects graph credits per-turn (``trace_id``), but
        # ``increment_returned`` receives no turn/credit info, so the bypass is
        # keyed off the phase's timing mode -- a GRAPH_IR phase issues ONLY
        # trace_id-stamped graph credits (``CreditIssuer.issue_graph_credit``),
        # making the two detections equivalent.
        self._graph_phase = getattr(config, "timing_mode", None) == TimingMode.GRAPH_IR

        # Progress counters
        self._requests_sent: int = 0
        self._root_requests_sent: int = 0
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
        """Total requests sent (root + DAG children)."""
        return self._requests_sent

    @property
    def root_requests_sent(self) -> int:
        """Wire requests sent from root sessions only.

        Diverges from ``requests_sent`` only for DAG runs: children with
        ``agent_depth > 0`` bump ``requests_sent`` (real wire activity)
        but not ``root_requests_sent``. Used by session-completion
        predicates to decide whether the strategy's main loop has
        finished issuing all expected ROOT wires for the planned
        sessions, independently of how many DAG children have fired.
        """
        return self._root_requests_sent

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

        Graph-IR credits (``turn_to_send.trace_id is not None``) bump only
        ``_requests_sent`` (the per-node wire/record count) and ALWAYS return
        ``is_final_credit=False``: they bypass the linear session arithmetic
        entirely because ``GraphIRReplayStrategy`` owns completion and the
        session/request stop semantics for a fan-out DAG.

        DAG children (``turn_to_send.agent_depth > 0``) count as real HTTP
        requests and DO bump ``_requests_sent`` — the user-visible
        "requests sent" metric must reflect actual wire traffic including
        DAG offspring. They do NOT bump ``_sent_sessions`` or
        ``_total_session_turns`` because they inherit the parent's
        session slot (they dispatch via ``CreditIssuer._dispatch_dag_turn``,
        which bypasses session-slot acquisition).

        ``is_final_credit`` flips when the request-count cap is crossed
        on either a root or a child. This is what drives
        ``freeze_sent_counts`` and ``all_credits_sent_event``; with
        children honoring the same cap as roots (see
        ``RequestCountStopCondition.applies_to_dag_children``), the cap
        can be crossed on a child increment, and the issuer must still
        unblock the strategy loop and the phase runner — otherwise the
        run hangs at-cap waiting for a signal that never fires.

        Lock-free: no async calls.
        """
        credit_index = self._requests_sent
        new_sent_count = self._requests_sent + 1

        if turn_to_send.trace_id is not None:
            # Graph-IR credits BYPASS all linear session arithmetic. A weka
            # trace is a fan-out DAG, not a linear ``turn_index==0 ->
            # num_turns-1`` session: every distinct node fires with
            # ``turn_index==0`` on a fresh per-node session key, so the linear
            # path would count each NODE as a fresh session and trip
            # ``is_final_credit`` after the Nth node (freezing the sent-count
            # mid-trace -> lost records). ``GraphIRReplayStrategy`` owns
            # completion and the session/request stop semantics, so here we
            # only bump the wire-request counter (one per node dispatch, which
            # IS the record count) and NEVER flip ``is_final_credit`` from a
            # graph credit -- the strategy freezes counts when its executors
            # drain. ``--request-count`` is still enforced upstream by the
            # ``RequestCountStopCondition`` gate against ``requests_sent``.
            self._requests_sent = new_sent_count
            return credit_index, False

        return self._increment_linear_sent(turn_to_send, credit_index, new_sent_count)

    def _increment_linear_sent(
        self, turn_to_send: TurnToSend, credit_index: int, new_sent_count: int
    ) -> tuple[int, bool]:
        """Linear (non-graph) sent-count arithmetic for root + DAG-child credits.

        Extracted from ``increment_sent`` (the graph-bypass early-return stays
        the leading branch there). Behavior is identical to the prior inline
        body: DAG children bump only ``_requests_sent`` and honor the
        request-count cap; roots also advance session / root-wire / turn
        counters and flip ``is_final_credit`` on the request-count OR
        session-completion predicate.

        Lock-free: no async calls.
        """
        if turn_to_send.agent_depth > 0:
            # Children: bump the wire-request counter only (slot is
            # inherited, sampler-plan counters stay root-only). Flip
            # is_final_credit when the request-count cap is crossed
            # on this child increment so the strategy loop and phase
            # runner unblock the same way they would for a root.
            self._requests_sent = new_sent_count
            is_final_credit = (
                self._config.total_expected_requests is not None
                and new_sent_count >= self._config.total_expected_requests
            )
            return credit_index, is_final_credit

        new_sent_sessions_count = self._sent_sessions
        new_total_session_turns = self._total_session_turns
        new_root_sent = self._root_requests_sent + 1

        if turn_to_send.turn_index == 0:
            new_sent_sessions_count += 1
            new_total_session_turns += turn_to_send.num_turns

        # Use root-only wire count (not global ``new_sent_count``) for the
        # session-completion predicate: BG-fork parents continue running
        # turns AFTER children begin firing, so the global counter would
        # spuriously satisfy the predicate the moment the first child wire
        # lands and the strategy loop would exit before the parent's
        # remaining turns could dispatch.
        is_final_credit = (
            self._config.total_expected_requests is not None
            and new_sent_count >= self._config.total_expected_requests
        ) or (
            self._config.expected_num_sessions is not None
            and new_sent_sessions_count >= self._config.expected_num_sessions
            and new_root_sent >= new_total_session_turns
        )

        self._requests_sent = new_sent_count
        self._root_requests_sent = new_root_sent
        self._sent_sessions = new_sent_sessions_count
        self._total_session_turns = new_total_session_turns

        return credit_index, is_final_credit

    def increment_returned(
        self, is_final_turn: bool, cancelled: bool, errored: bool = False
    ) -> bool:
        """Atomically increment returned count and check phase completion.

        Graph-IR phases (``self._graph_phase``) bypass the session counters
        entirely, mirroring the sent-side graph bypass in
        :meth:`increment_sent`: every graph credit is minted ``turn_index=0,
        num_turns=1`` so ``is_final_turn`` is always True, and counting it here
        would report one "completed session" per NODE (with ``sent_sessions``
        pinned at 0 -> negative ``in_flight_sessions`` and a bogus progress %).
        Trace-level completion is not visible at this callsite, so graph
        session stats stay 0 on both sides (like ``sent_sessions``) and phase
        progress is driven purely by the request counters; per-trace
        completion is owned and reported by ``GraphIRReplayStrategy``
        (``admitted_traces`` / ``completed_traces``).

        Lock-free: no async calls.

        Args:
            is_final_turn: Whether the returned turn is the final turn of its session
            cancelled: Whether the credit was cancelled
            errored: Whether the request returned with a non-None error. Errored
                requests still count as "returned" for the all-returned invariant
                (they are not cancellations), but also bump ``_request_errors``
                so the phase-complete log line reflects fault-injected runs.

        Returns:
            True if ALL sent credits have now been returned or cancelled
            (phase sending must be complete for this to ever return True).
        """
        count_sessions = is_final_turn and not self._graph_phase
        if cancelled:
            self._requests_cancelled += 1
            if count_sessions:
                self._cancelled_sessions += 1
        else:
            self._requests_completed += 1
            if count_sessions:
                self._completed_sessions += 1
            if errored:
                self._request_errors += 1

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
