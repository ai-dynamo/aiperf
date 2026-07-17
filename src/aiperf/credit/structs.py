# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for credit router communication.

All over-the-wire structs use tag_field="t" for efficient polymorphic decoding via tagged unions.
Tag values are short strings for minimal wire overhead.
"""

from typing import TYPE_CHECKING, Self

from msgspec import Struct

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
        phase: Type of credit phase (e.g., "warmup", "profiling").
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
    root_correlation_id: str | None = None
    """x_correlation_id of the depth-0 root of this credit's session TREE.

    Stable across the whole tree: the root carries its own x_correlation_id
    (left None on the wire when it equals x_correlation_id to keep the struct
    small), and every descendant (child, subchild) inherits the root's id.
    This is the key used for per-tree finality accounting
    (``SessionTreeRegistry``) and is persisted in the export so analysis groups
    a tree under one lane. Effective value is
    ``root_correlation_id or x_correlation_id``."""
    has_forks: bool = False
    is_parent_final: bool | None = None
    """Parent conversation had already returned its final turn at issue time.
    None for roots / when not determinable. Issue-time stamp, never copied."""
    is_tree_final: bool = False
    """Provably the last request the whole session tree will send (conservative
    False when indeterminate). Issue-time stamp from SessionTreeRegistry."""
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK
    """DAG branch mode for this credit. Ignored when parent_correlation_id is None
    (i.e. for root sessions). FORK = inherit parent turn_list; SPAWN =
    fresh context. Default FORK keeps wire footprint small via msgspec omit_defaults."""

    trace_id: str | None = None
    """Graph-IR trace instance identifier this credit addresses
    (``{template}::{nonce}``, e.g. ``t-1::3f2a...``);
    None for non-graph (template/DAG) dispatch."""

    node_ordinal: int | None = None
    """Ordinal of the graph-IR node this credit addresses; the worker
    materializes the node's request from the graph mmap store by this ordinal.
    None for non-graph dispatch."""

    phase_variant: str = "profiling"
    """Graph-IR phase variant label for this credit (e.g. ``"profiling"``,
    ``"warmup"``). Distinct from the credit-router ``phase`` enum."""

    first_token_event: bool = False
    """When True the worker emits a ``FirstToken`` event on TTFT for this credit
    even with prefill-concurrency limiting off (post-TTFT first-token anchoring).
    The graph first-token observer keys off the event's ``trace_id``. Default
    False keeps the wire footprint small via msgspec ``omit_defaults``."""

    @property
    def is_final_turn(self) -> bool:
        return self.turn_index == self.num_turns - 1

    @property
    def effective_root_correlation_id(self) -> str:
        """Tree root id, defaulting to this credit's own ``x_correlation_id``."""
        return self.root_correlation_id or self.x_correlation_id


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
        request_latency_ns: Request latency in nanoseconds using records-pipeline
            semantics.
        inter_token_latency_ns: Inter-token latency in nanoseconds using
            adaptive records-pipeline semantics.
        output_sequence_length: Output sequence length in tokens from usage
            data, when available.
    """

    credit: Credit
    drop_perf_ns: int
    cancelled: bool = False
    returned: bool = False
    first_token_sent: bool = False
    error: str | None = None
    request_latency_ns: int | None = None
    inter_token_latency_ns: float | None = None
    output_sequence_length: int | None = None


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
    root_correlation_id: str | None = None
    """x_correlation_id of the depth-0 root of this turn's session TREE.

    None for a root turn (the root IS its own tree root); set on every
    descendant to the root's id. Propagated onto the issued ``Credit`` and used
    for per-tree finality accounting. Effective value is
    ``root_correlation_id or x_correlation_id``."""
    has_forks: bool = False
    has_branches: bool = False
    """True iff the originating turn declares ANY branch (FORK or SPAWN) in its
    metadata ``branch_ids``. Superset of ``has_forks``, which is FORK-only and
    owned by the sticky router's deferred-eviction logic — do not conflate the
    two. Consumed by finality stamping: a turn that will spawn descendants on
    its return can never be the tree's provably-last request, even when the
    registry shows nothing outstanding yet (SPAWN children register only at
    return-intercept, AFTER issue-time stamping)."""
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK

    trace_id: str | None = None
    """Graph-IR trace instance identifier this turn addresses
    (``{template}::{nonce}``, e.g. ``t-1::3f2a...``); None for non-graph
    (template/DAG) dispatch. Copied into the issued Credit."""

    node_ordinal: int | None = None
    """Ordinal of the graph-IR node this turn addresses; the worker materializes
    the node's request from the graph mmap store by this ordinal. None for
    non-graph dispatch. Copied into the issued Credit."""

    phase_variant: str = "profiling"
    """Graph-IR phase variant label for this turn (e.g. ``"profiling"``,
    ``"warmup"``). Copied into the issued Credit."""

    first_token_event: bool = False
    """When True the issued Credit requests a per-credit ``FirstToken`` event on
    TTFT regardless of prefill-concurrency limiting (post-TTFT first-token
    anchoring). Copied into the issued Credit."""

    @property
    def is_final_turn(self) -> bool:
        return self.turn_index == self.num_turns - 1

    @property
    def effective_root_correlation_id(self) -> str:
        """Tree root id, defaulting to this turn's own ``x_correlation_id``."""
        return self.root_correlation_id or self.x_correlation_id

    @classmethod
    def from_previous_credit(
        cls, credit: Credit, next_meta: "TurnMetadata | None" = None
    ) -> Self:
        """Create the next turn to send from the previous turn's credit.

        Args:
            credit: The previous turn's credit.
            next_meta: Metadata for the NEW turn being built. When provided, the
                ``has_forks`` flag is derived from it so the sticky
                router can defer parent-entry eviction until DAG children drain,
                and ``has_branches`` (any-mode) is derived from its
                ``branch_ids`` so finality stamping stays conservative on
                spawning turns.
        """
        return cls(
            conversation_id=credit.conversation_id,
            x_correlation_id=credit.x_correlation_id,
            turn_index=credit.turn_index + 1,
            num_turns=credit.num_turns,
            agent_depth=credit.agent_depth,
            parent_correlation_id=credit.parent_correlation_id,
            root_correlation_id=credit.root_correlation_id,
            has_forks=next_meta.has_forks if next_meta is not None else False,
            has_branches=bool(next_meta.branch_ids) if next_meta is not None else False,
            branch_mode=credit.branch_mode,
            trace_id=credit.trace_id,
            node_ordinal=credit.node_ordinal,
            phase_variant=credit.phase_variant,
            first_token_event=credit.first_token_event,
        )
