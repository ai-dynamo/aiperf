# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ConfigDict, Field

from aiperf.common import random_generator as rng
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.models import AIPerfBaseModel
from aiperf.credit.structs import TurnToSend

# =============================================================================
# RequestCancellationConfig - Configuration for request cancellation simulator
# =============================================================================


class RequestCancellationConfig(AIPerfBaseModel):
    """Configuration for request cancellation simulator.

    Controls probabilistic request cancellation for testing how inference servers
    handle client disconnections. The cancellation timer starts after the request
    is fully sent (headers + body written to socket).
    """

    model_config = ConfigDict(frozen=True)

    rate: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of requests to cancel (0-100). None or 0 disables cancellation.",
    )
    delay: float = Field(
        default=0.0,
        ge=0,
        description="Seconds to wait after request is fully sent before cancelling.",
    )


# =============================================================================
# RequestCancellationSimulator - Probabilistic request cancellation simulator
# =============================================================================


class RequestCancellationSimulator:
    """Probabilistic request cancellation simulator for testing client disconnections.

    Tests how inference servers handle client disconnections by sending complete
    requests and then disconnecting before receiving the full response. At credit
    issuance time, this simulator probabilistically decides whether a request should
    be cancelled and sets Credit.cancel_after_ns accordingly.

    Timing Flow:
        1. CreditIssuer calls next_cancellation_delay_ns() at credit issuance
        2. Delay (if any) is stored in Credit.cancel_after_ns
        3. Worker passes this to AioHttpClient via RequestInfo
        4. AioHttpClient starts the cancellation timer AFTER the request is
           fully sent (headers + body written to socket)
        5. If timer expires, request is cancelled and CreditReturn has
           error="RequestCancellationError"

        Timeline::

            T0: Credit issued (issued_at_ns captured for metrics)
                 │
                 │← Worker processing, connection acquired from pool
                 ▼
            T1: Start writing request to socket
                 │
                 │← HTTP headers + body transmitted
                 ▼
            T2: Request fully sent (timer starts here)
                 │
                 │← cancel_after_ns delay
                 ▼
            T3: Request cancelled if still waiting for response

        The timer starts at T2 ("request fully sent") to ensure the server always
        receives the complete request before disconnection. A delay of 0 means
        "send the full request, then immediately disconnect". Longer delays allow
        partial responses to be received before disconnection.

    Behavior:
        - Disabled during warmup to ensure consistency
        - Enabled during profiling based on configured cancellation rate
        - Uses derived RNG for reproducible patterns across runs

    Note:
        This is distinct from "credit cancellation" (CancelCredits message)
        which is used for clean shutdown on Ctrl+C or phase timeout.
    """

    def __init__(self, config: RequestCancellationConfig):
        """Initialize simulator with cancellation rate and delay from config.

        Args:
            config: RequestCancellationConfig containing rate (0-100%)
                   and delay (seconds until cancellation after request sent).
        """
        self.config = config

        self._rng = rng.derive("timing.request.cancellation")

        self._is_cancellation_enabled = bool(config.rate)
        self._cancellation_rate = (
            config.rate / 100.0 if self._is_cancellation_enabled else None
        )
        self._cancellation_delay_ns = int(config.delay * NANOS_PER_SECOND)

    def next_cancellation_delay_ns(
        self, _turn_to_send: TurnToSend | None = None, phase: CreditPhase | None = None
    ) -> int | None:
        """Probabilistically determine cancellation delay for the next credit.

        See class docstring for the complete timing flow.

        Args:
            _turn_to_send: The turn being issued (unused, reserved for future use).
            phase: The current credit phase (WARMUP or PROFILING).

        Returns:
            Cancellation delay in nanoseconds to store in Credit.cancel_after_ns,
            or None if this request should not be cancelled.
        """
        if not self._is_cancellation_enabled:
            return None

        # Don't cancel during warmup
        if phase == CreditPhase.WARMUP:
            return None

        if self._rng.random() < self._cancellation_rate:
            return self._cancellation_delay_ns

        return None

    @property
    def is_cancellation_enabled(self) -> bool:
        """True if config.rate is set and non-zero."""
        return self._is_cancellation_enabled
