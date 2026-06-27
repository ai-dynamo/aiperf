# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User session management for multi-turn conversation optimization."""

from collections import OrderedDict

import msgspec

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models.dataset_models import Conversation, Turn

DEFAULT_MAX_SESSIONS = 100_000
"""Default per-worker cap on cached multi-turn sessions.

Sessions are normally evicted by the worker on the final turn or on
cancellation. Abandoned sessions — e.g. a non-final ``CreditReturn`` reclaimed
sticky-router side on worker reconnect/detach, or a ``WITH_RESPONSES`` session
migrated to a new worker leaving the original entry stranded — never receive a
final-turn or cancelled credit on the originating worker, so they would
otherwise accrue in ``_cache`` for the process lifetime. The LRU bound caps that
leak; the limit is high enough that legitimate concurrent multi-turn sessions
on a single worker stay resident.
"""


class UserSession(msgspec.Struct, kw_only=True, omit_defaults=True):
    """
    User session for multi-turn processing.

    Stores full conversation data and turn list (including assistant responses)
    to enable building requests with conversation context. In-process state
    only; never crosses a wire boundary. Mutable — the worker appends to
    ``turn_list`` and advances ``turn_index`` as the session progresses.
    """

    x_correlation_id: str
    num_turns: int
    conversation: Conversation
    url_index: int | None = None
    turn_list: list[Turn] = msgspec.field(default_factory=list)
    turn_index: int = 0
    context_mode: ConversationContextMode = (
        ConversationContextMode.DELTAS_WITHOUT_RESPONSES
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

        turn = self.conversation.turns[turn_index]
        if self.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES:
            self.turn_list = [turn]
        else:
            self.turn_list.append(turn)
        self.turn_index = turn_index
        return turn

    def should_store_response(self) -> bool:
        """Whether assistant responses should be stored based on context mode.

        Responses are stored when the dataset does not include them (WITHOUT_RESPONSES),
        so AIPerf must capture them live.
        """
        return self.context_mode == ConversationContextMode.DELTAS_WITHOUT_RESPONSES

    def store_response(self, response_turn: Turn) -> None:
        """
        Store the response for the turn.
        """
        self.turn_list.append(response_turn)


class UserSessionManager:
    """User session manager for multi-turn processing.

    Manages user sessions for multi-turn processing.
    """

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions ({max_sessions}) must be >= 1")
        self._max_sessions = max_sessions
        self._cache: OrderedDict[str, UserSession] = OrderedDict()
        self._default_context_mode: ConversationContextMode | None = None

    def set_default_context_mode(self, mode: ConversationContextMode | None) -> None:
        """Set the dataset-level default context mode from the loader."""
        self._default_context_mode = mode

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
        context_mode = (
            conversation.context_mode
            or self._default_context_mode
            or ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        )
        user_session = UserSession(
            x_correlation_id=x_correlation_id,
            num_turns=num_turns,
            url_index=url_index,
            conversation=conversation,
            turn_list=[],
            context_mode=context_mode,
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
        self._cache.move_to_end(x_correlation_id)
        # Bound the cache: sessions abandoned without a final-turn or cancelled
        # credit are never evicted by the worker, so cap retention by dropping the
        # least-recently-accessed entry once the limit is exceeded.
        while len(self._cache) > self._max_sessions:
            self._cache.popitem(last=False)

    def get(self, x_correlation_id: str) -> UserSession | None:
        """
        Get user session.

        Args:
            x_correlation_id: X-Correlation-ID header value
        """
        session = self._cache.get(x_correlation_id)
        if session is not None:
            self._cache.move_to_end(x_correlation_id)
        return session

    def evict(self, x_correlation_id: str) -> None:
        """
        Evict user session.

        Args:
            x_correlation_id: X-Correlation-ID header value
        """
        self._cache.pop(x_correlation_id, None)
