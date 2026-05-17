# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User session management for multi-turn conversation optimization.

Two session types serve two dataset formats:

- ``ContentSession`` backs CONVERSATION-format datasets (synthetic, public,
  file-based, DAG). It holds the full ``Conversation``, accumulates the
  ``turn_list`` (including stored assistant responses), tracks the resolved
  context mode, and participates in FORK seeding.
- ``RawPayloadSession`` backs PAYLOAD_BYTES-format datasets (pre-encoded
  wire bytes in mmap). It carries only ``conversation_id`` + routing
  fields; the body lives in mmap and the records-process resolves it via
  its own ``MemoryMapDatasetClientStore``. FORK + raw_payload is refused at
  dataset-format selection, so fork-related fields stay on ContentSession.

``UserSession`` is a union alias for either type. ``UserSessionManager``
exposes two factory methods (``create_content_session`` /
``create_raw_payload_session``); the worker calls the right one based on
``self._dataset_format``.
"""

from pydantic import Field

from aiperf.common.enums import ConversationBranchMode, ConversationContextMode
from aiperf.common.models import AIPerfBaseModel
from aiperf.common.models.dataset_models import Conversation, Turn


def _compute_is_fork_parent(conversation: Conversation) -> bool:
    """True if this conversation declares any FORK-mode branch.

    Stamped onto ``ContentSession`` at creation rather than recomputed on
    every read because ``conversation.branches`` is dropped on the
    PAYLOAD_BYTES context-mode wire round-trip; a lazy read after that
    would silently flip the flag to ``False``.
    """
    return any(b.mode == ConversationBranchMode.FORK for b in conversation.branches)


class _BaseSession(AIPerfBaseModel):
    """Common routing fields for both session types.

    Subclasses add format-specific fields and advance/store-response
    behavior. Holds nothing format-specific itself so worker-level code
    that only cares about routing (``x_correlation_id``, ``url_index``,
    ``turn_index``, ``num_turns``) can stay polymorphic.
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
    turn_index: int = Field(
        default=0,
        ge=0,
        description="The index of the current turn in the conversation",
    )


class ContentSession(_BaseSession):
    """User session backed by a fully-loaded ``Conversation``.

    Used for CONVERSATION-format datasets (synthetic, public, file-based,
    DAG). Drives the worker's request-builder via ``turn_list`` and stores
    assistant responses live when the context mode requires it. FORK
    seeding copies the parent's ``turn_list`` into the child here.
    """

    conversation: Conversation = Field(
        ..., description="Full conversation data from DatasetManager"
    )
    turn_list: list[Turn] = Field(
        default_factory=list,
        description="Current list of turns in conversation order, including the assistant responses",
    )
    context_mode: ConversationContextMode = Field(
        default=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
        description="Resolved context mode for this session. "
        "Set at creation from conversation-level override, dataset default, or DELTAS_WITHOUT_RESPONSES.",
    )
    is_fork_parent: bool = Field(
        default=False,
        description="Whether this session declares any FORK-mode branch and "
        "must therefore be pinned in the worker's session cache until all FORK "
        "children evict. Stamped at ``create_content_session`` time from "
        "``conversation.branches`` so the eviction path does not depend on "
        "``conversation`` retaining its branch metadata (PAYLOAD_BYTES "
        "context-mode round-trips strip ``branches``).",
    )
    fork_refcount: int = Field(
        default=0,
        ge=0,
        description="Refcount of pending DAG-FORK children that pin this "
        "session in the manager so its history is still resident when "
        "each child credit dispatches. Incremented at child-seed time "
        "by ``pin_for_fork_child``; decremented on child join by "
        "``release_fork_child``. Eviction (``evict_if_unpinned``) is a "
        "no-op while this is non-zero.",
    )
    pending_fork_eviction: bool = Field(
        default=False,
        description="When True, the parent's terminal turn has already "
        "fired, but eviction is deferred until all FORK-mode children "
        "have joined. Used by ``release_fork_child`` to auto-evict the "
        "session the moment ``fork_refcount`` reaches 0 (the eviction "
        "path that normally fires on the parent's terminal turn cannot "
        "find any children to pin yet — orchestrator dispatches them on "
        "the credit-return path AFTER this terminal eviction runs).",
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


class RawPayloadSession(_BaseSession):
    """User session backed by a PAYLOAD_BYTES mmap dataset.

    The wire payload for each turn lives in mmap and is fetched per
    dispatch by the worker (``DatasetClientStoreProtocol.get_payload_bytes``);
    the records-process resolves the same bytes via its own client. There
    is therefore no ``Conversation`` body on the worker side and no
    ``turn_list`` to accumulate.

    FORK + PAYLOAD_BYTES is refused at dataset-format selection
    (``DatasetManager._select_mmap_format``); FORK-related fields and
    response storage live on ``ContentSession`` only. ``advance_turn``
    here is a pure index bump.
    """

    conversation_id: str = Field(
        ...,
        description="Conversation/session ID resolved against DatasetMetadata "
        "and the records-process's mmap client. Substitutes for "
        "``ContentSession.conversation.session_id``.",
    )

    def advance_turn(self, turn_index: int) -> None:
        """Bump the per-session turn cursor; no turn_list to mutate.

        Bound checks mirror ``ContentSession.advance_turn`` so out-of-range
        turn indices fail loudly at the same boundary.
        """
        if turn_index < 0:
            raise ValueError(f"Turn index {turn_index} is negative")
        if turn_index >= self.num_turns:
            raise ValueError(
                f"Turn index {turn_index} is out of range for conversation with {self.num_turns} turns"
            )
        self.turn_index = turn_index


# Polymorphic type alias for callers that don't care which session shape
# they hold (worker storage map, manager get/store/evict). Code that
# reads format-specific fields branches on ``isinstance`` at the use site.
UserSession = ContentSession | RawPayloadSession


class UserSessionManager:
    """User session manager for multi-turn processing.

    Holds the worker's per-correlation session cache and exposes two
    factory methods — one per dataset format. Eviction, FORK pinning, and
    the cache map itself are format-agnostic; only ``create_*`` and
    ``seed_from_parent`` are format-specific (FORK seeding only fires on
    ``ContentSession``).
    """

    def __init__(self) -> None:
        self._cache: dict[str, UserSession] = {}
        self._default_context_mode: ConversationContextMode | None = None

    def set_default_context_mode(self, mode: ConversationContextMode | None) -> None:
        """Set the dataset-level default context mode from the loader."""
        self._default_context_mode = mode

    def create_content_session(
        self,
        x_correlation_id: str,
        conversation: Conversation,
        num_turns: int,
        url_index: int | None = None,
    ) -> ContentSession:
        """Create and store a ``ContentSession`` for CONVERSATION-format datasets.

        Args:
            x_correlation_id: X-Correlation-ID header value
            conversation: Conversation
            num_turns: Number of turns to execute (from Credit.num_turns). May be less than
                len(conversation.turns) for ramp-up users who start mid-session.
            url_index: URL index for multi-URL load balancing. All turns in this session
                will use this index to ensure they hit the same backend server.

        Raises:
            ValueError: If num_turns exceeds the actual conversation length.
            NotImplementedError: If the conversation declares FORK branches paired with an
                incompatible context_mode or raw_payload turns.
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
        # FORK seeding hands the parent's accumulated ``turn_list`` to
        # the child. ``MESSAGE_ARRAY_WITH_RESPONSES`` replaces ``turn_list``
        # on every ``advance_turn`` (see below), which would discard the
        # seed before the child sends its first request. Defensive
        # rejection: dag_jsonl pins ``DELTAS_WITHOUT_RESPONSES`` for all
        # FORK conversations today, so this only fires for hand-authored
        # configs or future loaders that get the pairing wrong.
        if (
            is_fork_parent
            and context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        ):
            raise NotImplementedError(
                f"conversation '{conversation.session_id}': FORK-mode branches "
                f"are incompatible with context_mode="
                f"{ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES.value!r}; "
                "FORK requires DELTAS_WITHOUT_RESPONSES so the parent's "
                "captured assistant turns survive child seeding"
            )
        # Raw-payload turns carry no role/content/raw_messages, so a FORK
        # child seeded from such a parent would replay empty user
        # messages (chat/responses) or drop history entirely (raw
        # endpoint). dag_jsonl never emits raw_payload turns; this guards
        # hand-authored or future-loader configurations.
        if is_fork_parent and any(
            t.raw_payload is not None for t in conversation.turns
        ):
            raise NotImplementedError(
                f"conversation '{conversation.session_id}': FORK-mode branches "
                "are incompatible with raw_payload turns (raw_payload, "
                "inputs_json, mooncake_trace payload mode); FORK requires "
                "structured turn data so the parent context can be replayed "
                "to the child"
            )
        session = ContentSession(
            x_correlation_id=x_correlation_id,
            num_turns=num_turns,
            url_index=url_index,
            conversation=conversation,
            turn_list=[],
            context_mode=context_mode,
            is_fork_parent=is_fork_parent,
        )
        self.store(x_correlation_id, session)
        return session

    def create_raw_payload_session(
        self,
        x_correlation_id: str,
        conversation_id: str,
        num_turns: int,
        url_index: int | None = None,
    ) -> RawPayloadSession:
        """Create and store a ``RawPayloadSession`` for PAYLOAD_BYTES datasets.

        No ``Conversation`` body is loaded — the bytes live in mmap and
        the records-process resolves them through its own client. FORK +
        PAYLOAD_BYTES is refused upstream at dataset-format selection, so
        no FORK seeding hooks fire here.

        Args:
            x_correlation_id: X-Correlation-ID header value
            conversation_id: Session ID resolved against DatasetMetadata and
                the records-process's mmap client.
            num_turns: Number of turns to execute (from Credit.num_turns).
            url_index: URL index for multi-URL load balancing.
        """
        session = RawPayloadSession(
            x_correlation_id=x_correlation_id,
            num_turns=num_turns,
            url_index=url_index,
            conversation_id=conversation_id,
        )
        self.store(x_correlation_id, session)
        return session

    def store(self, x_correlation_id: str, user_session: UserSession) -> None:
        """
        Store user session.

        Refuses to silently replace an existing session with one of a
        different type — that combination indicates a sticky-router
        uniqueness violation (the same correlation id was claimed by two
        different dataset formats / session shapes). Same-type re-stores
        are permitted as a legitimate refresh.

        Args:
            x_correlation_id: X-Correlation-ID header value
            user_session: User session

        Raises:
            RuntimeError: If an existing session of a different concrete type
                is already cached under ``x_correlation_id``.
        """
        existing = self._cache.get(x_correlation_id)
        if existing is not None and type(existing) is not type(user_session):
            raise RuntimeError(
                f"UserSessionManager: x_correlation_id '{x_correlation_id}' "
                f"already stored as {type(existing).__name__}; refusing to "
                f"silently replace with {type(user_session).__name__}. This "
                f"indicates a sticky-router uniqueness violation."
            )
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

    def pin_for_fork_child(self, x_correlation_id: str) -> None:
        """Increment the FORK-pin refcount on the session.

        Called at child-seed time so the parent session stays resident
        in the cache until every FORK child has dispatched. Raises
        ``KeyError`` if the session is unknown — pinning a session that
        was already evicted is a programming error, not a soft failure.

        Only ``ContentSession`` participates in FORK; PAYLOAD_BYTES
        datasets refuse FORK at format-selection time, so this path
        cannot legitimately be reached for a ``RawPayloadSession``.
        """
        session = self._cache.get(x_correlation_id)
        if session is None:
            raise KeyError(
                f"pin_for_fork_child: no session for x_correlation_id "
                f"{x_correlation_id!r} (parent already evicted before FORK child arrived)"
            )
        if not isinstance(session, ContentSession):
            raise TypeError(
                f"pin_for_fork_child: session {x_correlation_id!r} is not a "
                "ContentSession; FORK + PAYLOAD_BYTES is refused at dataset "
                "format selection and should never reach this code path"
            )
        session.fork_refcount += 1

    def seed_from_parent(
        self, child_x_correlation_id: str, parent_x_correlation_id: str
    ) -> None:
        """Seed a freshly-created FORK child's ``turn_list`` with a copy of
        the parent's accumulated turn history.

        FORK-mode children inherit the parent's prompt + captured response
        context. The child's ``turn_list`` starts empty at
        ``create_content_session`` time; this copies the parent's current
        ``turn_list`` (a list of ``Turn`` objects, including stored
        assistant responses) into the child so that the request-builder
        prepends the full parent context before the child's own messages.

        No-op (with a debug-friendly silent return) if either session is
        already evicted or either side is a ``RawPayloadSession`` — the
        latter is a defensive guard; FORK + PAYLOAD_BYTES is refused at
        format selection.
        """
        parent = self._cache.get(parent_x_correlation_id)
        child = self._cache.get(child_x_correlation_id)
        if not isinstance(parent, ContentSession) or not isinstance(
            child, ContentSession
        ):
            return
        child.turn_list = list(parent.turn_list)

    def release_fork_child(self, x_correlation_id: str) -> None:
        """Decrement the FORK-pin refcount on the session, floored at 0.

        Called on child join. Releasing an already-zero or unknown
        session is a no-op — releases can race against eviction in
        practice and must not raise.

        When ``pending_fork_eviction`` is set (parent's terminal turn
        has already fired but was waiting for children to land) and
        the refcount drops to 0, the session is evicted in the same
        call — there is no other code path that will collect it.
        """
        session = self._cache.get(x_correlation_id)
        if not isinstance(session, ContentSession):
            return
        session.fork_refcount = max(0, session.fork_refcount - 1)
        if session.fork_refcount == 0 and session.pending_fork_eviction:
            self._cache.pop(x_correlation_id, None)

    def evict_if_unpinned(self, x_correlation_id: str) -> None:
        """Evict the session only if its FORK refcount has reached 0.

        Refcount-aware sibling of ``evict``: callers on the FORK path
        use this so pinned parents stay resident until the last child
        joins. Unknown or non-``ContentSession`` sessions fall through
        to a plain eviction (RawPayloadSession can never be FORK-pinned).

        Sessions with ``pending_fork_eviction = True`` ALSO stay
        resident at refcount==0 — their parent's terminal turn already
        fired, but the orchestrator's child dispatch happens AFTER
        this point, so we need to keep the session alive for the
        about-to-arrive children to seed from. ``release_fork_child``
        handles the eventual cleanup when the last child joins.
        """
        session = self._cache.get(x_correlation_id)
        if session is None:
            return
        if not isinstance(session, ContentSession):
            self._cache.pop(x_correlation_id, None)
            return
        if session.fork_refcount > 0:
            return
        if session.pending_fork_eviction:
            return
        self._cache.pop(x_correlation_id, None)
