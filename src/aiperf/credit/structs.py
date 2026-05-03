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

    Attributes:
        id: Sequential number of the credit in the credit phase.
        phase: Type of credit phase (e.g., "warmup", "profile").
        conversation_id: Template ID from the dataset.
        x_correlation_id: Conversation instance ID for sticky routing (X-Correlation-ID header).
        turn_index: Index of the turn in the conversation (0-based).
        num_turns: Total number of turns in the conversation.
        issued_at_ns: Wall clock timestamp when issued (time.time_ns).
        cancel_after_ns: Delay in nanoseconds after which the request should be cancelled for simulated client disconnections (optional).
                         Note: this is NOT the same as the credit being cancelled!
        url_index: Index of the URL to use when multiple --url values are configured (optional).
                   None means use the default (first) URL.
    """

    id: int
    phase: CreditPhase
    conversation_id: str
    x_correlation_id: str
    turn_index: int
    num_turns: int
    issued_at_ns: int
    cancel_after_ns: int | None = None
    url_index: int | None = None

    @property
    def is_final_turn(self) -> bool:
        return self.turn_index == self.num_turns - 1


class CreditContext(
    Struct, omit_defaults=True, kw_only=True, tag_field="t", tag="cctx"
):
    """Context for a credit. This is used by the worker to track details of a credit.

    Attributes:
        credit: The credit being processed.
        drop_perf_ns: The performance timestamp when the credit was dropped.
        cancelled: True if the credit was cancelled before completion.
        returned: True if the credit was returned after completion.
        first_token_sent: True if the first token was sent before this return.
        error: The error message if the request failed (None on success).
    """

    credit: Credit
    drop_perf_ns: int
    cancelled: bool = False
    returned: bool = False
    first_token_sent: bool = False
    error: str | None = None


# =============================================================================
# Turn Structs (pre-credit issuance structs)
# =============================================================================


class TurnToSend(Struct, frozen=True):
    """A turn that needs to be sent.

    Attributes:
        conversation_id: Template ID from the dataset.
        x_correlation_id: Conversation instance ID for sticky routing (X-Correlation-ID header).
        turn_index: The index of the turn in the conversation (0-based).
        num_turns: The total number of turns in the conversation.
    """

    conversation_id: str
    x_correlation_id: str
    turn_index: int
    num_turns: int

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
        )
