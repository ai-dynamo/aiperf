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
        # Monotonic dispatch sequence: bumped for EVERY issued credit
        # (roots, DAG children, and ``no_request`` orchestrator credits).
        # Sources ``Credit.id``, which must be unique per dispatched credit
        # because it keys live in-flight tracking (worker ``credit_tasks``,
        # ``StickyCreditRouter.active_credit_ids``, adaptive-scale windows).
        # Decoupled from ``_requests_sent`` so that ``no_request`` credits —
        # which do NOT bump the billable request counter — still receive a
        # distinct id from the next real credit.
        self._dispatch_seq: int = 0
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

    def account_lane_session(self, session_turns: int) -> None:
        """Count a deferred-lane session that acquires its slot directly.

        A gated parent admitted via ``CreditIssuer.acquire_lane_credit`` had its
        turn 0 before t*, so it never dispatches a session-start root credit
        through :meth:`increment_sent` -- yet its join turn DOES reach a terminal
        turn that bumps ``_completed_sessions`` in :meth:`increment_returned`.
        Without this, ``completed_sessions > sent_sessions`` and
        ``in_flight_sessions`` goes negative.

        Mirrors the ``(turn_index == 0 or is_session_start)`` arm of
        :meth:`increment_sent`: bump ``_sent_sessions`` and add the session's
        remaining turns to ``_total_session_turns`` so ``can_send_any_turn``
        (root_requests_sent vs total_session_turns) stays consistent as the
        gated parent's join + tail turns dispatch. Must be called AFTER the lane
        slot is acquired so the session does not gate its own admission via
        ``can_start_new_session`` (sent_sessions < expected_num_sessions).

        Lock-free: no async calls.
        """
        self._sent_sessions += 1
        self._total_session_turns += session_turns

    def increment_sent(self, turn_to_send: TurnToSend) -> tuple[int, bool]:
        """Atomically increment sent count and return (credit_index, is_final_credit).

        DAG children (``turn_to_send.agent_depth > 0``) count as real HTTP
        requests and DO bump ``_requests_sent`` — the user-visible
        "requests sent" metric must reflect actual wire traffic including
        DAG offspring. They do NOT bump ``_sent_sessions`` or
        ``_total_session_turns`` because they inherit the parent's
        session slot (``CreditIssuer.issue_credit`` skips session-slot
        acquisition for them).

        ``is_final_credit`` flips when the request-count cap is crossed
        on either a root or a child. This is what drives
        ``freeze_sent_counts`` and ``all_credits_sent_event``; with
        children honoring the same cap as roots (see
        ``RequestCountStopCondition.applies_to_dag_children``), the cap
        can be crossed on a child increment, and the issuer must still
        unblock the strategy loop and the phase runner — otherwise the
        run hangs at-cap waiting for a signal that never fires. The
        request-count cap fires regardless of ``counts_toward_phase_target``
        because it is a global wire-request cap (dag_jsonl #891 hang fix);
        the session-target predicate is gated on ``counts_toward_phase_target``
        so reactive AGENTIC_REPLAY children/sidecars (which set it False)
        cannot satisfy the sampled root plan.

        Lock-free: no async calls.
        """
        credit_index = self._dispatch_seq
        self._dispatch_seq += 1
        new_sent_count = self._requests_sent + 1

        # The request-count cap is a global wire cap honored by every credit
        # (root or child), so it can flip is_final_credit even for a credit
        # that does not count toward the sampled phase target. A ``no_request``
        # virtual orchestrator credit issues no wire request, so it must NOT
        # count toward the cap (mirrors the ``counts_as_request`` gate below
        # that keeps ``_requests_sent`` unbumped for it); without this gate a
        # spine's virtual credit could spuriously flip is_final_credit at-cap.
        hit_request_cap = (
            not turn_to_send.no_request
            and self._config.total_expected_requests is not None
            and new_sent_count >= self._config.total_expected_requests
        )

        if turn_to_send.agent_depth > 0:
            # Children: bump the wire-request counter only (slot is
            # inherited, sampler-plan counters stay root-only). Flip
            # is_final_credit when the request-count cap is crossed
            # on this child increment so the strategy loop and phase
            # runner unblock the same way they would for a root.
            self._requests_sent = new_sent_count
            return credit_index, hit_request_cap

        new_sent_sessions_count = self._sent_sessions
        new_total_session_turns = self._total_session_turns
        new_root_sent = self._root_requests_sent + 1

        # A root session is counted on its first credit in the phase: turn 0,
        # or an agentic mid-trace resume (is_session_start, turn_index > 0).
        # Without this, a resumed root would bump completed_sessions on its
        # final turn while never bumping sent_sessions, driving
        # completed_sessions > sent_sessions and in_flight_sessions < 0.
        # total_session_turns counts only the turns this session will actually
        # send (num_turns - start index), so a resumed root contributes its
        # remaining tail, keeping it consistent with root_requests_sent.
        if turn_to_send.turn_index == 0 or turn_to_send.is_session_start:
            new_sent_sessions_count += 1
            new_total_session_turns += turn_to_send.num_turns - turn_to_send.turn_index

        # A ``no_request`` orchestrator credit is a virtual root session: it
        # occupies a session slot and counts toward ``--num-conversations``
        # (session + root-plan counters below), but fires no HTTP request, so
        # it must NOT advance the wire-request counter or the
        # ``--request-count`` cap. ``_root_requests_sent`` stays balanced with
        # ``_total_session_turns`` (both bumped) so the session-completion
        # predicate remains satisfiable.
        counts_as_request = not turn_to_send.no_request
        if counts_as_request:
            self._requests_sent = new_sent_count
        else:
            new_sent_count = self._requests_sent

        # Use root-only wire count (not global ``new_sent_count``) for the
        # session-completion predicate: BG-fork parents continue running
        # turns AFTER children begin firing, so the global counter would
        # spuriously satisfy the predicate the moment the first child wire
        # lands and the strategy loop would exit before the parent's
        # remaining turns could dispatch. The session-target arm is gated on
        # ``counts_toward_phase_target`` so reactive (post-sampling) work does
        # not satisfy the sampled plan; the global request-cap arm (above) is
        # not, and already excludes ``no_request`` virtual credits.
        hit_session_target = turn_to_send.counts_toward_phase_target and (
            self._config.expected_num_sessions is not None
            and new_sent_sessions_count >= self._config.expected_num_sessions
            and new_root_sent >= new_total_session_turns
        )
        is_final_credit = hit_request_cap or hit_session_target

        self._root_requests_sent = new_root_sent
        self._sent_sessions = new_sent_sessions_count
        self._total_session_turns = new_total_session_turns

        return credit_index, is_final_credit

    def increment_returned(
        self,
        is_final_turn: bool,
        cancelled: bool,
        errored: bool = False,
        *,
        is_child: bool = False,
        no_request: bool = False,
    ) -> bool:
        """Atomically increment returned count and check phase completion.

        Lock-free: no async calls.

        DAG children DO bump ``_requests_completed`` / ``_requests_cancelled``
        — these are user-visible metrics of actual HTTP activity, symmetric
        with ``_requests_sent`` being bumped on the dispatch side. They do
        NOT bump ``_completed_sessions`` / ``_cancelled_sessions`` (session
        slot was inherited, not acquired).

        Args:
            is_final_turn: Whether the returned turn is the final turn of its session
            cancelled: Whether the credit was cancelled
            errored: Whether the request returned with a non-None error. Errored
                requests still count as "returned" for the all-returned invariant
                (they are not cancellations), but also bump ``_request_errors``
                so the phase-complete log line reflects fault-injected runs.
                Request-level, so it ticks for children too.
            is_child: True when ``credit.agent_depth > 0``. Session-level
                counters are skipped for children; request-level counters
                still tick.
            no_request: Whether the returned credit is a virtual ``no_request``
                orchestrator credit. Such credits do NOT bump the billable
                request counter (``_requests_completed``) — symmetric with
                ``increment_sent``, which does not bump ``_requests_sent`` for
                them. Without this symmetry ``final_requests_completed`` would
                over-count and the records pipeline (which expects exactly one
                record per completed wire request) would wait forever for
                records that a virtual credit never produces. The virtual
                credit still completes its session slot, so it bumps
                ``_completed_sessions`` on its final turn. Orthogonal to
                ``is_child`` (a no_request credit is a root, not a child).

        Returns:
            True if ALL sent credits have now been returned or cancelled
            (phase sending must be complete for this to ever return True).
        """
        if cancelled:
            self._requests_cancelled += 1
            if is_final_turn and not is_child:
                self._cancelled_sessions += 1
        else:
            # Request-level (completed/errors): every real wire request ticks,
            # children included; a ``no_request`` virtual credit does not.
            # Session-level (completed_sessions): gated on ``not is_child``
            # (children inherit the parent's slot); a ``no_request`` root is
            # NOT a child, so its final virtual turn still completes its slot.
            if not no_request:
                self._requests_completed += 1
                if errored:
                    self._request_errors += 1
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
