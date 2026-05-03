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
    """

    conversation_id: str
    metadata: ConversationMetadata
    x_correlation_id: str

    def build_first_turn(self, max_turns: int | None = None) -> TurnToSend:
        """Build first turn (turn_index=0) from sampled conversation.

        Args:
            max_turns: The maximum number of turns to send for this user. Simulates a user that is partially through a conversation.
                If None, the number of turns is determined by the conversation metadata.
        """
        return TurnToSend(
            conversation_id=self.conversation_id,
            x_correlation_id=self.x_correlation_id,
            turn_index=0,
            num_turns=max_turns or len(self.metadata.turns),
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
