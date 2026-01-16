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


class LifecycleStopCondition(StopCondition):
    """Lifecycle based stop condition. Checks if the phase is cancelled or has completed sending.

    NOTE: This is always used and is the first in the list of stop conditions.
    """

    @classmethod
    def should_use(cls, config: CreditPhaseConfig) -> bool:
        """Always use this stop condition."""
        return True

    def can_send_any_turn(self) -> bool:
        """Returns True if the phase is not cancelled and has not completed sending."""
        return (
            not self._lifecycle.was_cancelled
            and not self._lifecycle.is_sending_complete
        )


class RequestCountStopCondition(StopCondition):
    """Request count based stop condition."""

    @classmethod
    def should_use(cls, config: CreditPhaseConfig) -> bool:
        """Returns True if a request count limit is configured."""
        return config.total_expected_requests is not None

    def can_send_any_turn(self) -> bool:
        """Returns True if the request count limit has not been reached."""
        return self._counter.requests_sent < self._config.total_expected_requests


class SessionCountStopCondition(StopCondition):
    """Session count based stop condition."""

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
    """Duration based stop condition."""

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
    LifecycleStopCondition,  # Always used first
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
