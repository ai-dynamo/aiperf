# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User session management for multi-turn conversation optimization."""

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.common.models.dataset_models import Conversation, Turn


class UserSession(AIPerfBaseModel):
    """
    User session for multi-turn processing.

    Stores full conversation data and turn list (including assistant responses) to enable building requests
    with conversation context.
    """

    x_correlation_id: str = Field(
        ..., description="X-Correlation-ID header value. Used for sticky routing."
    )
    num_turns: int = Field(..., ge=0, description="Number of turns in the conversation")
    url_index: int | None = Field(
        default=None,
        description="URL index for multi-URL load balancing. "
        "Set on first turn to ensure all turns in a conversation hit the same backend.",
    )
    conversation: Conversation = Field(
        ..., description="Full conversation data from DatasetManager"
    )
    turn_list: list[Turn] = Field(
        default_factory=list,
        description="Current list of turns in conversation order, including the assistant responses",
    )
    turn_index: int = Field(
        default=0, ge=0, description="The index of the current turn in the conversation"
    )

    def advance_turn(self, turn_index: int) -> Turn:
        """
        Advance the turn list to the next turn.

        Args:
            turn_index: The index of the turn to advance to.

        Returns:
            The turn that was advanced to.
        """
        if turn_index < 0:
            raise ValueError(f"Turn index {turn_index} is negative")
        if turn_index >= self.num_turns:
            raise ValueError(
                f"Turn index {turn_index} is out of range for conversation with {self.num_turns} turns"
            )

        self.turn_list.append(self.conversation.turns[turn_index])
        self.turn_index = turn_index
        return self.turn_list[-1]

    def store_response(self, response_turn: Turn) -> None:
        """
        Store the response for the turn.
        """
        self.turn_list.append(response_turn)


class UserSessionManager:
    """User session manager for multi-turn processing.

    Manages user sessions for multi-turn processing.
    """

    def __init__(self) -> None:
        self._cache: dict[str, UserSession] = {}

    def create_and_store(
        self,
        x_correlation_id: str,
        conversation: Conversation,
        num_turns: int,
        url_index: int | None = None,
    ) -> UserSession:
        """
        Create and store user session.

        Args:
            x_correlation_id: X-Correlation-ID header value
            conversation: Conversation
            num_turns: Number of turns to execute (from Credit.num_turns). May be less than
                len(conversation.turns) for ramp-up users who start mid-session.
            url_index: URL index for multi-URL load balancing. All turns in this session
                will use this index to ensure they hit the same backend server.

        Raises:
            ValueError: If num_turns exceeds the actual conversation length.
        """
        if num_turns > len(conversation.turns):
            raise ValueError(
                f"num_turns ({num_turns}) exceeds conversation length ({len(conversation.turns)})"
            )
        user_session = UserSession(
            x_correlation_id=x_correlation_id,
            num_turns=num_turns,
            url_index=url_index,
            conversation=conversation,
            turn_list=[],
        )
        self.store(x_correlation_id, user_session)
        return user_session

    def store(self, x_correlation_id: str, user_session: UserSession) -> None:
        """
        Store user session.

        Args:
            x_correlation_id: X-Correlation-ID header value
            user_session: User session
        """
        self._cache[x_correlation_id] = user_session

    def get(self, x_correlation_id: str) -> UserSession | None:
        """
        Get user session.

        Args:
            x_correlation_id: X-Correlation-ID header value
        """
        return self._cache.get(x_correlation_id)

    def evict(self, x_correlation_id: str) -> None:
        """
        Evict user session.

        Args:
            x_correlation_id: X-Correlation-ID header value
        """
        self._cache.pop(x_correlation_id, None)
