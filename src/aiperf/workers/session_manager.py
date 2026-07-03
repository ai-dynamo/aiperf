# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User session management for multi-turn conversation optimization."""

from collections import OrderedDict

import msgspec

from aiperf.common.enums import ConversationBranchMode, ConversationContextMode
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


def _compute_is_fork_parent(conversation: Conversation) -> bool:
    """True if this conversation declares any FORK-mode branch.

    Stamped onto ``UserSession`` at creation rather than recomputed on
    every read because ``conversation.branches`` is dropped on the
    PAYLOAD_BYTES context-mode wire round-trip; a lazy read after that
    would silently flip the flag to ``False``.
    """
    return any(b.mode == ConversationBranchMode.FORK for b in conversation.branches)


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
    is_fork_parent: bool = False
    """Whether this session declares any FORK-mode branch and must therefore be
    pinned in the worker's session cache until all FORK children evict. Stamped
    at ``create_and_store`` time from ``conversation.branches`` so the eviction
    path does not depend on ``conversation`` retaining its branch metadata
    (PAYLOAD_BYTES context-mode round-trips strip ``branches``)."""
    fork_refcount: int = 0
    """Refcount of pending DAG-FORK children that pin this session so its
    history is still resident when each child credit dispatches. Incremented at
    child-seed time by ``pin_for_fork_child``; decremented on child join by
    ``release_fork_child``. Eviction (``evict_if_unpinned``) is a no-op while
    this is non-zero."""
    pending_fork_eviction: bool = False
    """When True, the parent's terminal turn has already fired, but eviction is
    deferred until all FORK-mode children have joined. Used by
    ``release_fork_child`` to auto-evict the session the moment ``fork_refcount``
    reaches 0 (the eviction path that normally fires on the parent's terminal
    turn cannot find any children to pin yet — the orchestrator dispatches them
    on the credit-return path AFTER this terminal eviction runs)."""

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
        is_fork_parent = _compute_is_fork_parent(conversation)
        # FORK seeding hands the parent's accumulated ``turn_list`` to the child.
        # ``MESSAGE_ARRAY_WITH_RESPONSES`` replaces ``turn_list`` on every
        # ``advance_turn`` (see ``advance_turn``), which would discard the seed
        # before the child sends its first request. Defensive rejection:
        # dag_jsonl pins ``DELTAS_WITHOUT_RESPONSES`` for all FORK conversations
        # today, so this only fires for hand-authored configs or future loaders
        # that get the pairing wrong.
        if (
            is_fork_parent
            and context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        ):
            raise NotImplementedError(
                f"conversation '{conversation.session_id}': FORK-mode branches "
                f"are incompatible with context_mode="
                f"{ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES}; "
                "FORK requires DELTAS_WITHOUT_RESPONSES so the parent's captured "
                "assistant turns survive child seeding"
            )
        # Raw-payload turns carry no role/content/raw_messages, so a FORK child
        # seeded from such a parent would replay empty user messages
        # (chat/responses) or drop history entirely (raw endpoint). dag_jsonl
        # never emits raw_payload turns; this guards hand-authored or
        # future-loader configurations.
        if is_fork_parent and any(
            t.raw_payload is not None for t in conversation.turns
        ):
            raise NotImplementedError(
                f"conversation '{conversation.session_id}': FORK-mode branches "
                "are incompatible with raw_payload turns (raw_payload, "
                "inputs_json, mooncake_trace payload mode); FORK requires "
                "structured turn data so the parent context can be replayed to "
                "the child"
            )
        user_session = UserSession(
            x_correlation_id=x_correlation_id,
            num_turns=num_turns,
            url_index=url_index,
            conversation=conversation,
            turn_list=[],
            context_mode=context_mode,
            is_fork_parent=is_fork_parent,
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

    def pin_for_fork_child(self, x_correlation_id: str) -> None:
        """Increment the FORK-pin refcount on the session.

        Called at child-seed time so the parent session stays resident in the
        cache until every FORK child has dispatched. Raises ``KeyError`` if the
        session is unknown — pinning a session that was already evicted is a
        programming error, not a soft failure.
        """
        session = self._cache.get(x_correlation_id)
        if session is None:
            raise KeyError(
                f"pin_for_fork_child: no session for x_correlation_id "
                f"{x_correlation_id!r} (parent already evicted before FORK child arrived)"
            )
        session.fork_refcount += 1

    def seed_from_parent(
        self, child_x_correlation_id: str, parent_x_correlation_id: str
    ) -> None:
        """Seed a freshly-created FORK child's ``turn_list`` with a copy of the
        parent's accumulated turn history.

        FORK-mode children inherit the parent's prompt + captured response
        context. The child's ``turn_list`` starts empty at ``create_and_store``
        time; this copies the parent's current ``turn_list`` (a list of ``Turn``
        objects, including stored assistant responses) into the child so that
        the request-builder prepends the full parent context before the child's
        own messages.

        No-op if either session is already evicted — the FORK-pin refcount
        usually keeps the parent resident, but late-arriving children may race
        past eviction. The request still goes out, just without seed context,
        matching the ``pin_for_fork_child`` "evicted parent" branch.
        """
        parent = self._cache.get(parent_x_correlation_id)
        child = self._cache.get(child_x_correlation_id)
        if parent is None or child is None:
            return
        child.turn_list = list(parent.turn_list)

    def release_fork_child(self, x_correlation_id: str) -> None:
        """Decrement the FORK-pin refcount on the session, floored at 0.

        Called on child join. Releasing an already-zero or unknown session is a
        no-op — releases can race against eviction in practice and must not
        raise.

        When ``pending_fork_eviction`` is set (parent's terminal turn has already
        fired but was waiting for children to land) and the refcount drops to 0,
        the session is evicted in the same call — there is no other code path
        that will collect it.
        """
        session = self._cache.get(x_correlation_id)
        if session is None:
            return
        session.fork_refcount = max(0, session.fork_refcount - 1)
        if session.fork_refcount == 0 and session.pending_fork_eviction:
            self._cache.pop(x_correlation_id, None)

    def evict_if_unpinned(self, x_correlation_id: str) -> None:
        """Evict the session only if its FORK refcount has reached 0.

        Refcount-aware sibling of ``evict``: callers on the FORK path use this
        so pinned parents stay resident until the last child joins. Unknown
        sessions are a no-op.

        Sessions with ``pending_fork_eviction = True`` ALSO stay resident at
        refcount==0 — their parent's terminal turn already fired, but the
        orchestrator's child dispatch happens AFTER this point, so we need to
        keep the session alive for the about-to-arrive children to seed from.
        ``release_fork_child`` handles the eventual cleanup when the last child
        joins.
        """
        session = self._cache.get(x_correlation_id)
        if session is None:
            return
        if session.fork_refcount > 0:
            return
        if session.pending_fork_eviction:
            return
        self._cache.pop(x_correlation_id, None)
