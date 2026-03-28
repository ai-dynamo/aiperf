# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for credit router communication.

All over-the-wire structs use tag_field="t" for efficient polymorphic decoding via tagged unions.
Tag values are short strings for minimal wire overhead.
"""

from msgspec import Struct
from typing_extensions import Self

from aiperf.common.enums import CreditPhase

# =============================================================================
# Credit Struct (sent from router to worker)
# =============================================================================


class Credit(
    Struct, omit_defaults=True, frozen=True, kw_only=True, tag_field="t", tag="c"
):
    """Credit representing the right to make a single request to an inference server.

    Sent directly from router to worker (no wrapper message).
    """

    id: int
    """Sequential number of the credit in the credit phase (0-based request index)."""

    phase: CreditPhase
    """Type of credit phase (e.g. warmup, profile)."""

    conversation_id: str
    """Template ID from the dataset."""

    x_correlation_id: str
    """Conversation instance ID for sticky routing (X-Correlation-ID header)."""

    turn_index: int
    """Index of the turn in the conversation (0-based)."""

    num_turns: int
    """Total number of turns in the conversation."""

    issued_at_ns: int
    """Wall clock timestamp when issued (time.time_ns)."""

    cancel_after_ns: int | None = None
    """Delay in nanoseconds after which the request should be cancelled for simulated client disconnections."""

    url_index: int | None = None
    """Index of the URL to use when multiple --url values are configured."""

    allow_worker_migration: bool = False
    """Whether the session can safely continue on a different worker after worker loss."""

    session_num: int | None = None
    """Sequential number of the session/conversation (0-based), shared across all turns."""

    @property
    def is_final_turn(self) -> bool:
        return self.turn_index == self.num_turns - 1


class CreditContext(
    Struct, omit_defaults=True, kw_only=True, tag_field="t", tag="cctx"
):
    """Context for a credit, used by the worker to track processing details."""

    credit: Credit
    """The credit being processed."""

    drop_perf_ns: int
    """Performance timestamp when the credit was dropped."""

    credit_received_ns: int = 0
    """Performance timestamp when the credit was received by the worker."""

    cancelled: bool = False
    """True if the credit was cancelled before completion."""

    returned: bool = False
    """True if the credit was returned after completion."""

    first_token_sent: bool = False
    """True if the first token was sent before this return."""

    error: str | None = None
    """Error message if the request failed (None on success)."""


# =============================================================================
# Turn Structs (pre-credit issuance structs)
# =============================================================================


class TurnToSend(Struct, frozen=True):
    """A turn that needs to be sent."""

    conversation_id: str
    """Template ID from the dataset."""

    x_correlation_id: str
    """Conversation instance ID for sticky routing (X-Correlation-ID header)."""

    turn_index: int
    """Index of the turn in the conversation (0-based)."""

    num_turns: int
    """Total number of turns in the conversation."""

    url_index: int | None = None
    """Preserved backend affinity for multi-URL runs."""

    allow_worker_migration: bool = False
    """Whether a later worker may continue this session after the original worker is lost."""

    @property
    def is_final_turn(self) -> bool:
        return self.turn_index == self.num_turns - 1

    @classmethod
    def from_previous_credit(cls, credit: Credit) -> Self:
        """Create the next turn to send from the previous turn's credit."""
        return cls(
            conversation_id=credit.conversation_id,
            x_correlation_id=credit.x_correlation_id,
            turn_index=credit.turn_index + 1,
            num_turns=credit.num_turns,
            url_index=credit.url_index,
            allow_worker_migration=credit.allow_worker_migration,
        )
