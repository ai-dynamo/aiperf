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

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.dataset.protocols import DatasetSamplingStrategyProtocol


@dataclass(slots=True)
class SampledSession:
    """A runtime session instance of a conversation.

    Returned by ConversationSource.next(). Each session is a unique execution
    of a conversation template.

    Attributes:
        conversation_id: Template ID from dataset (can be reused across sessions).
        metadata: Conversation metadata (turns, prompts, etc.) from the template.
        x_correlation_id: Unique session ID (UUID). Enables sticky routing so all
            turns in this session route to the same worker.
        agent_depth: Static DAG nesting level (0 = root). Mirrors the loaded
            ConversationMetadata.agent_depth; copied here so the credit issuer
            can stamp it on TurnToSend without re-reading metadata.
        parent_correlation_id: Parent session's x_correlation_id when this is a
            DAG child. None for root sessions. The router uses this for sticky
            pinning so FORK children land on the parent's worker.
        branch_mode: How the child relates to its parent. FORK inherits the
            parent's accumulated message history and pins to the same worker;
            SPAWN starts with a fresh context. Ignored when
            parent_correlation_id is None.
    """

    conversation_id: str
    metadata: ConversationMetadata
    x_correlation_id: str
    agent_depth: int = 0
    parent_correlation_id: str | None = None
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK

    @property
    def routing_key(self) -> str:
        """Sticky-routing key.

        Returns the parent's correlation_id when set (so FORK children share
        a worker with the parent), otherwise this session's own
        x_correlation_id.
        """
        return self.parent_correlation_id or self.x_correlation_id

    def build_first_turn(self, max_turns: int | None = None) -> TurnToSend:
        """Build first turn (turn_index=0) from sampled conversation.

        Args:
            max_turns: The maximum number of turns to send for this user. Simulates a user that is partially through a conversation.
                If None, the number of turns is determined by the conversation metadata.
        """
        first_meta = self.metadata.turns[0] if self.metadata.turns else None
        return TurnToSend(
            conversation_id=self.conversation_id,
            x_correlation_id=self.x_correlation_id,
            turn_index=0,
            num_turns=max_turns or len(self.metadata.turns),
            agent_depth=self.agent_depth,
            parent_correlation_id=self.parent_correlation_id,
            has_forks=first_meta.has_forks if first_meta is not None else False,
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
        """Build a SampledSession for a DAG child conversation.

        Under FORK mode, the returned session inherits sticky-routing from its
        parent via ``parent_correlation_id``; the credit router pins the child
        to the parent's worker, where ``UserSessionManager.create_and_store``
        seeds ``turn_list`` by cloning the parent's in-memory session.
        SPAWN-mode children start with a fresh context, but the sticky pin to
        the parent's correlation_id is preserved at this layer — routing
        freedom is enforced upstream by the orchestrator/router.
        """
        metadata = self._metadata_lookup[child_conversation_id]
        return SampledSession(
            conversation_id=child_conversation_id,
            metadata=metadata,
            x_correlation_id=str(uuid.uuid4()),
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

        Restricted to SPAWN mode with ``dispatch_timing="pre"`` at the validator
        level; FORK pre-dispatch would require inheriting a non-existent
        parent session and is rejected at load time.
        """
        metadata = self._metadata_lookup[child_conversation_id]
        return SampledSession(
            conversation_id=child_conversation_id,
            metadata=metadata,
            x_correlation_id=str(uuid.uuid4()),
            agent_depth=1,
            parent_correlation_id=None,
            branch_mode=ConversationBranchMode.SPAWN,
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
