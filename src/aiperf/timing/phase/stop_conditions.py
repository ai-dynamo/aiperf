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
    # from the ``DatasetSampler``. Their stop-condition behavior splits
    # by intent:
    #   - cancellation, duration timeout, ``--request-count``: HONORED.
    #     ``--request-count`` is a literal wire-request cap ("30 means 30")
    #     and time/cancellation are user-facing guarantees that must apply
    #     to every credit on the wire.
    #   - ``is_sending_complete``, ``--num-conversations``: BYPASSED.
    #     The first is a TimingStrategy-loop signal that flips before
    #     children begin; honoring it would block the DAG from draining.
    #     The second targets sampler-plan completion ("run N full
    #     conversations") — children belong to a conversation tree and
    #     should run as part of their parent's session, not be truncated
    #     mid-tree. Concrete conditions set ``applies_to_dag_children = False``
    #     to opt out; ``RequestCountStopCondition`` deliberately stays True.
    # When a child's continuation is blocked by an honored condition,
    # ``CreditCallbackHandler`` notifies ``BranchOrchestrator.on_child_stopped``
    # so the parent's join can drain instead of deadlocking.
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

    Honored by every credit, including DAG children — ``--request-count N``
    is a literal cap on wire requests. Once ``requests_sent`` reaches the
    cap, no further roots OR children are issued; this is consistent with
    how multi-turn continuations on origin/main get truncated mid-stream
    when the cap is hit. A child whose first-turn dispatch is refused
    here is removed from the parent's pending-join during the
    orchestrator's intercept cleanup; a child whose continuation is
    refused triggers ``BranchOrchestrator.on_child_stopped`` from the
    callback handler so the parent's join still drains.
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

    Bypassed for DAG children. ``--num-conversations`` is a sampler
    plan target — "run N full conversations" — and DAG offspring are
    part of the conversation tree they belong to, not separate
    plannable units. The counters this reads (``sent_sessions``,
    ``total_session_turns``) correctly exclude children, but the
    ``OR`` comparison still flips False the instant the last root
    fires its planned turns; honoring the gate for children would
    truncate the DAG mid-tree on every run, producing partial
    conversations that mismatch the stated semantics. Children
    bypass this condition so the planned N conversations actually
    complete; the wire-cap intent is served by ``--request-count``
    instead, which DOES apply to children
    (see ``RequestCountStopCondition``).
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
        """
        return (
            self._counter.sent_sessions < self._config.expected_num_sessions
            or self._counter.requests_sent < self._counter.total_session_turns
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
        # Subset of conditions that DAG children must still honor.
        # Today: cancellation, duration, ``--request-count``. Excludes
        # ``SendingCompleteStopCondition`` (TimingStrategy loop signal
        # that flips before children begin) and
        # ``SessionCountStopCondition`` (``--num-conversations`` is a
        # full-conversation plan target — children should run as part
        # of their parent's session, not be truncated mid-tree). The
        # callback handler turns refusals on this gate into
        # ``BranchOrchestrator.on_child_stopped`` calls so parent
        # joins still drain.
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

        Children honor every stop condition whose concrete class
        declares ``applies_to_dag_children = True``. Today the
        ``False`` opt-outs are:

        - ``SendingCompleteStopCondition`` — flips before children
          begin; honoring it would block every child.
        - ``SessionCountStopCondition`` — ``--num-conversations`` is
          a sampler-plan target; truncating mid-tree would produce
          partial conversations that mismatch the stated semantics.

        Children DO honor cancellation, duration timeout, and
        ``--request-count`` — the wire-request cap. A child blocked
        at this gate gets routed through
        ``BranchOrchestrator.on_child_stopped`` so the parent's join
        drains instead of deadlocking.

        Called by ``CreditIssuer`` when ``turn.agent_depth > 0`` and
        by ``CreditCallbackHandler`` to decide whether to dispatch a
        child's continuation turn.
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
