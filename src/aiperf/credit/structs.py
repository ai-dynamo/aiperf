# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for credit router communication.

All over-the-wire structs use tag_field="t" for efficient polymorphic decoding via tagged unions.
Tag values are short strings for minimal wire overhead.
"""

from typing import TYPE_CHECKING

from msgspec import Struct
from typing_extensions import Self

from aiperf.common.enums import ConversationBranchMode, CreditPhase

if TYPE_CHECKING:
    from aiperf.common.models.dataset_models import TurnMetadata

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
        cancel_after_ns: Delay in nanoseconds after which the request should be cancelled
                         for simulated client disconnections (optional).
                         Note: this is NOT the same as the credit being cancelled!
        url_index: Index of the URL to use when multiple --url values are configured (optional).
                   None means use the default (first) URL.
        agent_depth: DAG nesting level (0 = root session). Stamped onto MetricRecordMetadata
                     for layer-filtering.
        parent_correlation_id: x_correlation_id of the parent session for DAG children;
                               None for root sessions.
        has_forks: True iff the originating turn declares one or more FORK-mode branches;
                   consumed by the sticky router to defer parent-entry eviction until
                   children drain.
        branch_mode: FORK vs SPAWN; ignored when parent_correlation_id is None.
                     FORK = inherit parent turn_list and pin to parent's worker;
                     SPAWN = fresh context, free routing.
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
    agent_depth: int = 0
    parent_correlation_id: str | None = None
    has_forks: bool = False
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK
    """DAG branch mode for this credit. Ignored when parent_correlation_id is None
    (i.e. for root sessions). FORK = inherit parent turn_list; SPAWN =
    fresh context. Default FORK keeps wire footprint small via msgspec omit_defaults."""

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
        agent_depth: DAG nesting level (0 = root); copied into the issued Credit.
        parent_correlation_id: Parent session's x_correlation_id for DAG children;
                               None for root sessions.
        has_forks: True iff this turn declares any FORK-mode branch; the sticky router
                   uses it to defer parent-entry eviction.
        branch_mode: FORK or SPAWN; ignored when parent_correlation_id is None.
    """

    conversation_id: str
    x_correlation_id: str
    turn_index: int
    num_turns: int
    agent_depth: int = 0
    parent_correlation_id: str | None = None
    has_forks: bool = False
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK

    @property
    def is_final_turn(self) -> bool:
        return self.turn_index == self.num_turns - 1

    @classmethod
    def from_previous_credit(
        cls, credit: Credit, next_meta: "TurnMetadata | None" = None
    ) -> Self:
        """Create the next turn to send from the previous turn's credit.

        Args:
            credit: The previous turn's credit.
            next_meta: Metadata for the NEW turn being built. When provided, the
                ``has_forks`` flag is derived from it so the sticky
                router can defer parent-entry eviction until DAG children drain.
        """
        return cls(
            conversation_id=credit.conversation_id,
            x_correlation_id=credit.x_correlation_id,
            turn_index=credit.turn_index + 1,
            num_turns=credit.num_turns,
            agent_depth=credit.agent_depth,
            parent_correlation_id=credit.parent_correlation_id,
            has_forks=next_meta.has_forks if next_meta is not None else False,
            branch_mode=credit.branch_mode,
        )
