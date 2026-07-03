# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conversation source for sampling and metadata access.

Combines dataset sampling, metadata lookup, x_correlation_id generation,
and helpers for multi-turn decision making.

Terminology:
    conversation_id: Template identifier from the dataset. A conversation can be
        sampled multiple times to create multiple sessions.
    session: A single execution of a conversation template. Has its own
        x_correlation_id and maintains state (worker assignment, turn progress).
    x_correlation_id: Unique session identifier (UUID). Each session is a runtime
        instance of a conversation. Used for sticky routing - all turns in a
        session route to the same worker.
"""

import uuid
from dataclasses import dataclass

from aiperf.common.enums import ConversationBranchMode, ConversationContextMode
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.dataset.protocols import DatasetSamplingStrategyProtocol


@dataclass(slots=True)
class SampledSession:
    """A runtime session instance of a conversation.

    Returned by ConversationSource.next(). Each session is a unique execution
    of a conversation template.
    """

    conversation_id: str
    """Template ID from dataset (can be reused across sessions)."""

    metadata: ConversationMetadata
    """Conversation metadata (turns, prompts, etc.) from the template."""

    x_correlation_id: str
    """Unique session ID (UUID) for sticky routing all turns to the same worker."""

    allow_worker_migration: bool = False
    """Whether later turns can safely continue on a different worker after worker loss."""

    agent_depth: int = 0
    """Static DAG nesting level (0 = root). Mirrors ConversationMetadata.agent_depth."""

    parent_correlation_id: str | None = None
    """Parent session's x_correlation_id when this is a DAG child. None for root sessions.

    Intended as the sticky-pin key for FORK children, but same-worker pinning
    is inert in v1, so this does not currently force children onto the parent's
    worker.
    """

    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK
    """How the child relates to its parent. FORK inherits the parent's accumulated message
    history by seeding from the parent's session; SPAWN starts with a fresh context.
    Ignored when parent_correlation_id is None.
    """

    @property
    def routing_key(self) -> str:
        """Sticky-routing key.

        Returns the parent's correlation_id when set (the key a same-worker
        pin would use for FORK children, though that pin is inert in v1),
        otherwise this session's own x_correlation_id.
        """
        return self.parent_correlation_id or self.x_correlation_id

    def build_first_turn(self, max_turns: int | None = None) -> TurnToSend:
        """Build first turn (turn_index=0) from sampled conversation.

        Args:
            max_turns: The maximum number of turns to send for this user. Simulates a user that is partially through a conversation.
                If None, the number of turns is determined by the conversation metadata.
        """
        first_meta = self.metadata.turns[0] if self.metadata.turns else None
        has_forks = first_meta.has_forks if first_meta is not None else False
        return TurnToSend(
            conversation_id=self.conversation_id,
            x_correlation_id=self.x_correlation_id,
            turn_index=0,
            num_turns=max_turns or len(self.metadata.turns),
            allow_worker_migration=self.allow_worker_migration,
            agent_depth=self.agent_depth,
            parent_correlation_id=self.parent_correlation_id,
            has_forks=has_forks,
            branch_mode=self.branch_mode,
        )


class ConversationSource:
    """Samples conversations from dataset to create session instances.

    Used by timing strategies to get sessions for credit issuance.
    Generates unique x_correlation_id per session for sticky routing.
    """

    def __init__(
        self,
        dataset_metadata: DatasetMetadata,
        dataset_sampler: DatasetSamplingStrategyProtocol,
    ):
        """Initialize conversation source."""
        self._dataset_metadata = dataset_metadata
        self._dataset_sampler = dataset_sampler
        self._metadata_lookup: dict[str, ConversationMetadata] = {
            conv.conversation_id: conv for conv in dataset_metadata.conversations
        }

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        """Dataset metadata."""
        return self._dataset_metadata

    def next(self, x_correlation_id: str | None = None) -> SampledSession:
        """Sample next conversation and return a new session instance."""
        conversation_id = self._dataset_sampler.next_conversation_id()
        metadata = self._metadata_lookup[conversation_id]

        return SampledSession(
            conversation_id=conversation_id,
            metadata=metadata,
            x_correlation_id=x_correlation_id or str(uuid.uuid4()),
            allow_worker_migration=self.get_context_mode(conversation_id)
            in {
                ConversationContextMode.DELTAS_WITH_RESPONSES,
                ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            },
        )

    def get_metadata(self, conversation_id: str) -> ConversationMetadata:
        """Get metadata for a specific conversation."""
        if conversation_id not in self._metadata_lookup:
            raise KeyError(f"No metadata for conversation {conversation_id}")
        return self._metadata_lookup[conversation_id]

    def start_branch_child(
        self,
        parent_correlation_id: str,
        child_conversation_id: str,
        agent_depth: int,
        *,
        branch_mode: ConversationBranchMode = ConversationBranchMode.FORK,
    ) -> SampledSession:
        """Build a SampledSession for a DAG child conversation (FORK or SPAWN-on-parent).

        The returned session carries ``parent_correlation_id`` as its intended
        sticky-pin key, but same-worker pinning is inert in v1, so the child is
        not forced onto the parent's worker. SPAWN-mode children start with a
        fresh context; the sticky-pin key is retained at this layer for when
        the router hooks become active.
        """
        metadata = self._metadata_lookup[child_conversation_id]
        return SampledSession(
            conversation_id=child_conversation_id,
            metadata=metadata,
            x_correlation_id=str(uuid.uuid4()),
            allow_worker_migration=self.get_context_mode(child_conversation_id)
            in {
                ConversationContextMode.DELTAS_WITH_RESPONSES,
                ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            },
            agent_depth=agent_depth,
            parent_correlation_id=parent_correlation_id,
            branch_mode=branch_mode,
        )

    def start_pre_session_child(
        self,
        child_conversation_id: str,
    ) -> SampledSession:
        """Build a SampledSession for a pre-session (turn-0) background SPAWN child.

        Used by ``BranchOrchestrator.dispatch_pre_session_branches`` to fire
        a child before its parent's turn 0 is issued. The child gets a fresh
        correlation id, ``agent_depth=1``, and ``parent_correlation_id=None``
        (no real parent session exists yet). Because ``parent_correlation_id``
        is None, the child's ``routing_key`` naturally equals its own
        ``x_correlation_id`` — the child routes freely (no sticky pin).
        """
        metadata = self._metadata_lookup[child_conversation_id]
        return SampledSession(
            conversation_id=child_conversation_id,
            metadata=metadata,
            x_correlation_id=str(uuid.uuid4()),
            allow_worker_migration=self.get_context_mode(child_conversation_id)
            in {
                ConversationContextMode.DELTAS_WITH_RESPONSES,
                ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            },
            agent_depth=1,
            parent_correlation_id=None,
            branch_mode=ConversationBranchMode.SPAWN,
        )

    def get_context_mode(self, conversation_id: str) -> ConversationContextMode:
        """Resolve context mode for a specific conversation.

        Resolution order matches worker-side session setup:
        conversation override -> dataset default -> DELTAS_WITHOUT_RESPONSES.
        """
        metadata = self.get_metadata(conversation_id)
        return (
            metadata.context_mode
            or self._dataset_metadata.default_context_mode
            or ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        )

    def get_next_turn_metadata(self, credit: Credit) -> TurnMetadata:
        """Get metadata for next turn after completed credit.

        Raises:
            ValueError: If next turn doesn't exist (credit is final turn).
        """
        metadata = self.get_metadata(credit.conversation_id)
        next_index = credit.turn_index + 1

        if next_index >= len(metadata.turns):
            raise ValueError(
                f"No turn {next_index} in conversation {credit.conversation_id} "
                f"(only {len(metadata.turns)} turns exist)"
            )
        return metadata.turns[next_index]
