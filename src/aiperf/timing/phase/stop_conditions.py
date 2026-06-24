# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stop condition checker for phase credit issuance.

Evaluates whether more credits can be sent based on lifecycle state,
counter values, and configuration limits. Pure read-only - never mutates state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.phase.credit_counter import CreditCounter
    from aiperf.timing.phase.lifecycle import PhaseLifecycle

# =============================================================================
# StopCondition implementations
# =============================================================================


class StopCondition(ABC):
    """Abstract base class for a stop condition.

    This is used to evaluate whether more credits can be sent. Concrete subclasses
    implement the should_use() and can_send_any_turn() methods for general checks,
    and may optionally implement the can_start_new_session() method for more restrictive cases.
    """

    # DAG children (``agent_depth > 0``) are spawned reactively by the
    # ``BranchOrchestrator`` at credit-return time — they are NOT driven
    # by the phase's ``TimingStrategy`` loop and do not consume entries
    # from the ``DatasetSampler``. They honor stop conditions that
    # represent user-facing guarantees (cancellation, duration) but
    # bypass ones tied to the TimingStrategy's own loop termination
    # (``is_sending_complete``) or to root-session count targets
    # (``--request-count``, ``--conversation-num``) that were authored
    # for the sampled roots, not their reactive offspring. Concrete
    # conditions set ``applies_to_dag_children = False`` to opt out of
    # child evaluation; all others apply by default.
    applies_to_dag_children: bool = True

    def __init__(
        self,
        config: CreditPhaseConfig,
        lifecycle: PhaseLifecycle,
        counter: CreditCounter,
    ) -> None:
        """Initialize the stop condition. These are all the things that stop conditions have access to."""
        self._config = config
        self._lifecycle = lifecycle
        self._counter = counter

    @classmethod
    @abstractmethod
    def should_use(cls, config: CreditPhaseConfig) -> bool:
        """Returns True if the stop condition should be used for the given configuration.

        This allows dynamically configuring the stop conditions based on which ones are actually relevant.
        For example, if no duration is configured, we don't need to check it.
        """
        pass

    @abstractmethod
    def can_send_any_turn(self) -> bool:
        """True if phase can send ANY turn (first or subsequent)."""
        pass

    def can_start_new_session(self) -> bool:
        """True if phase can start a NEW session.

        Checked in addition to can_send_any_turn() on every first turn.
        Default returns True (no additional restriction). Subclasses like
        SessionCountStopCondition override to prevent new sessions while
        still allowing continuation turns from existing sessions.
        """
        return True


class CancellationStopCondition(StopCondition):
    """Phase-cancelled stop condition.

    Honored by *every* credit, including DAG children — when the user
    cancels (Ctrl-C, explicit API abort, pod eviction), all in-flight
    credit issuance must stop. Separated from the sending-complete
    check so DAG children can bypass the latter without bypassing
    cancellation.
    """

    @classmethod
    def should_use(cls, config: CreditPhaseConfig) -> bool:
        return True

    def can_send_any_turn(self) -> bool:
        return not self._lifecycle.was_cancelled


class SendingCompleteStopCondition(StopCondition):
    """Phase has marked ``is_sending_complete`` on the lifecycle.

    Set by ``PhaseRunner._wait_for_sending_complete`` after
    ``progress.all_credits_sent_event`` fires — which ``CreditIssuer``
    sets as soon as ``CreditCounter.increment_sent`` reports
    ``is_final_credit`` (i.e. the root count / session-turn target has
    been reached).

    DAG children bypass this condition: the flag fires when the
    ``TimingStrategy`` loop has dispatched its last targeted credit,
    which is typically *before* the ``BranchOrchestrator`` has even
    intercepted the root's return to spawn children. Honoring it would
    block every child. DAG completion is tracked separately by
    ``BranchOrchestrator.has_pending_branch_work()``; the callback
    handler defers ``all_credits_returned_event`` until that drains.
    """

    applies_to_dag_children = False

    @classmethod
    def should_use(cls, config: CreditPhaseConfig) -> bool:
        return True

    def can_send_any_turn(self) -> bool:
        return not self._lifecycle.is_sending_complete


class RequestCountStopCondition(StopCondition):
    """Request count based stop condition.

    Honored by EVERY credit, including DAG children — ``--request-count N``
    is a literal cap on total wire requests ("N means N"). Once
    ``requests_sent`` reaches N no further roots OR children are issued; an
    in-flight tree is truncated mid-stream, exactly as a multi-turn session is
    truncated mid-conversation. A child refused at this cap is routed through
    ``BranchOrchestrator.on_child_stopped`` (via the child-issuance chokepoint)
    so the parent's join drains instead of deadlocking. The "run full trees"
    knob is ``--num-conversations`` (``SessionCountStopCondition``), which DOES
    stay bypassed for children.

    INVARIANT — do NOT re-add ``applies_to_dag_children = False`` here. This
    condition inheriting the base ``True`` is one half of the exact-cutoff
    contract; the other half is ``CreditCounter.increment_sent``'s
    unconditional request-count arm (it must flip ``is_final_credit`` when a
    child/join turn crosses the cap). Flipping EITHER re-breaks "N means N" —
    it has been re-broken 3x historically. See the agentx hard-cap spec.
    """

    @classmethod
    def should_use(cls, config: CreditPhaseConfig) -> bool:
        """Returns True if a request count limit is configured."""
        return config.total_expected_requests is not None

    def can_send_any_turn(self) -> bool:
        """Returns True if the request count limit has not been reached."""
        return self._counter.requests_sent < self._config.total_expected_requests


class SessionCountStopCondition(StopCondition):
    """Session count based stop condition.

    Bypassed for DAG children. The counters it reads
    (``sent_sessions``, ``total_session_turns``) correctly exclude
    children (they inherit the parent's session slot and only bump
    the request-level counters — see ``CreditCounter.increment_sent``),
    but the OR comparison still goes at-cap once the root plan
    exhausts. Bypass lets DAG offspring run past it.
    """

    applies_to_dag_children = False

    @classmethod
    def should_use(cls, config: CreditPhaseConfig) -> bool:
        """Returns True if a session count limit is configured."""
        return config.expected_num_sessions is not None

    def can_send_any_turn(self) -> bool:
        """Returns True if more turns can be sent.

        True when either: session limit not reached (can start new sessions),
        OR already-started sessions still have unsent turns remaining.

        The unsent-turns comparison uses ``root_requests_sent`` (not
        ``requests_sent``) to stay in lockstep with
        ``CreditCounter.increment_sent``'s ``is_final_credit`` predicate, which
        also compares root-only sends against ``total_session_turns``. DAG
        children inflate ``requests_sent`` but inherit the parent's session slot
        and add no root turns; using the global count here would close the gate
        on a multi-turn root's remaining continuations (silently dropped in
        ``CreditCallbackHandler``) while ``is_final_credit`` never fires.
        """
        return (
            self._counter.sent_sessions < self._config.expected_num_sessions
            or self._counter.root_requests_sent < self._counter.total_session_turns
        )

    def can_start_new_session(self) -> bool:
        """Returns True if new sessions can be started (limit not reached).

        More restrictive than can_send_any_turn(): prevents starting NEW sessions
        but can_send_any_turn() may still allow turns from already-started sessions.
        """
        return self._counter.sent_sessions < self._config.expected_num_sessions


class DurationStopCondition(StopCondition):
    """Duration based stop condition.

    Honored by DAG children — the user promised a time-bounded run.
    Children that reach ``--benchmark-duration`` stop dispatching
    further turns; in-flight requests drain via their own
    cancellation path.
    """

    @classmethod
    def should_use(cls, config: CreditPhaseConfig) -> bool:
        """Returns True if a benchmark duration is configured."""
        return config.expected_duration_sec is not None

    def can_send_any_turn(self) -> bool:
        """Returns True if the duration has not been reached."""
        time_left = self._lifecycle.time_left_in_seconds()
        return time_left is not None and time_left > 0


# NOTE: The order of these classes will determine the order that the stop conditions are checked in.
_STOP_CONDITION_CLASSES = [
    CancellationStopCondition,  # Always used first — honored by every credit, including DAG children.
    SendingCompleteStopCondition,  # Always used — skipped for DAG children.
    RequestCountStopCondition,
    SessionCountStopCondition,
    DurationStopCondition,
]

# =============================================================================
# StopConditionChecker - Evaluate stop conditions
# =============================================================================


class StopConditionChecker:
    """Evaluates whether more credits can be sent.

    Read-only access to lifecycle and counter - never mutates.
    All decisions are pure functions of current state.

    Used by CreditIssuer to check preconditions before issuing credits.
    The check is performed AFTER acquiring concurrency slots to prevent
    races between slot acquisition and stop condition changes.

    Stop conditions (first one reached wins):
    - Cancelled: Phase was externally cancelled (Ctrl+C)
    - Sending complete: Already marked all credits as sent
    - Timeout: Expected duration elapsed
    - Request count: Sent count >= total_expected_requests
    - Session complete: All sessions started AND all their turns sent
    """

    def __init__(
        self,
        config: CreditPhaseConfig,
        lifecycle: PhaseLifecycle,
        counter: CreditCounter,
    ) -> None:
        """Initialize stop condition checker.

        Args:
            config: Phase configuration with stop thresholds.
            lifecycle: Read-only lifecycle state (was_cancelled, is_sending_complete).
            counter: Read-only counter values (requests_sent, sent_sessions, etc.).
        """
        # Configure and add stop conditions that should be used for the given configuration
        self._stop_conditions: list[StopCondition] = [
            stop_condition_class(config, lifecycle, counter)
            for stop_condition_class in _STOP_CONDITION_CLASSES
            if stop_condition_class.should_use(config)
        ]

        # Cache the stop condition functions to avoid looking them up on every call.
        # micro-optimization for something that will be called a lot
        self._can_send_any_turn_funcs: list[Callable] = [
            stop_condition.can_send_any_turn for stop_condition in self._stop_conditions
        ]
        self._can_start_new_session_funcs: list[Callable] = [
            stop_condition.can_start_new_session
            for stop_condition in self._stop_conditions
        ]
        # Subset of conditions that DAG children must still honor
        # (cancellation, duration, request/session counts). Excludes
        # ``SendingCompleteStopCondition`` — see its docstring.
        self._can_send_child_turn_funcs: list[Callable] = [
            stop_condition.can_send_any_turn
            for stop_condition in self._stop_conditions
            if stop_condition.applies_to_dag_children
        ]

    def can_send_any_turn(self) -> bool:
        """True if phase can send ANY turn (first or subsequent).

        Checked before EVERY credit issuance to prevent races.
        Returns False if:
        - Phase was cancelled
        - Sending already marked complete
        - Timeout elapsed
        - Request count limit reached
        - All sessions complete (session-based mode)
        """
        return all(func() for func in self._can_send_any_turn_funcs)

    def can_send_child_turn(self) -> bool:
        """True if a DAG child credit can be issued.

        Children honor only the stop conditions whose concrete class
        declares ``applies_to_dag_children = True`` (today:
        ``CancellationStopCondition`` and ``DurationStopCondition`` —
        the ones that represent user-facing guarantees). They bypass:

        - ``SendingCompleteStopCondition`` — the ``TimingStrategy``
          loop's "I've dispatched my last targeted credit" flag, which
          flips before DAG children even begin.
        - ``RequestCountStopCondition`` / ``SessionCountStopCondition``
          — the ``<`` comparison goes at-cap the instant the last root
          fires; the counters themselves are already root-only (see
          ``CreditCounter.increment_sent``).

        Called by ``CreditIssuer`` when ``turn.agent_depth > 0``.
        """
        return all(func() for func in self._can_send_child_turn_funcs)

    def can_start_new_session(self) -> bool:
        """True if phase can start a NEW session (more restrictive).

        Used for first turn concurrency acquisition.
        Prevents starting new sessions when near limits.

        Returns False if can_send_any_turn() is False, OR:
        - Session quota reached (can still send subsequent turns of existing sessions)
        """
        # Must pass all general checks first
        if not self.can_send_any_turn():
            return False

        return all(func() for func in self._can_start_new_session_funcs)
