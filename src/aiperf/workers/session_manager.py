# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User session management for multi-turn conversation optimization."""

from pydantic import Field

from aiperf.common.enums import ConversationBranchMode, ConversationContextMode
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
        description="Current list of turns in conversation order, including the assistant responses. "
        "For FORK-mode DAG children, seeded at session creation from the parent's "
        "turn_list so the endpoint sees the full inherited history.",
    )
    turn_index: int = Field(
        default=0, ge=0, description="The index of the current turn in the conversation"
    )
    context_mode: ConversationContextMode = Field(
        default=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
        description="Resolved context mode for this session. "
        "Set at creation from conversation-level override, dataset default, or DELTAS_WITHOUT_RESPONSES.",
    )
    parent_correlation_id: str | None = Field(
        default=None,
        description="Parent session's x_correlation_id when this is a DAG child "
        "(set at ``create_and_store`` time for FORK/SPAWN children). ``None`` "
        "for root sessions. Used by the session manager's FORK-pin refcount "
        "to decrement the parent's live-child count on eviction.",
    )
    branch_mode: ConversationBranchMode | None = Field(
        default=None,
        description="Relationship to the parent (FORK / SPAWN) when this is a DAG "
        "child, else ``None``. FORK children contribute to the parent's "
        "pinned-eviction refcount; SPAWN children do not (they don't seed "
        "from the parent so the parent does not need to stay cached for them).",
    )

    def advance_turn(self, turn_index: int) -> Turn:
        """Append the next turn onto ``turn_list`` and return it.

        Mutates ``turn_list`` in place:
        - Under ``MESSAGE_ARRAY_WITH_RESPONSES`` the list is replaced with
          ``[turn]`` (each turn carries its own full history; prior turns are
          dropped so the endpoint sends the turn's messages as-authored).
        - Under every other mode the turn is appended. Callers that only need
          per-turn overrides can read ``turn_list[-1]`` after this call.
          When jumping ahead past untraversed turns (e.g. agentic_replay's
          mid-trajectory resume at ``k_i > 0``), prior turns 0..turn_index-1
          are seeded first so the endpoint accumulator reproduces the full
          chat prefix from the trace's delta-encoded turns.

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
            # Delta-encoded modes accumulate. For ``DELTAS_WITH_RESPONSES``
            # (e.g. weka agentic_replay), if a caller skips ahead past
            # untraversed turns (agentic_replay warmup resumes at k_i > 0
            # without going through turns 0..k_i-1), seed those first so
            # the endpoint's build_messages reproduces the full chat prefix
            # from the trace's pre-canned delta turns.
            #
            # ``DELTAS_WITHOUT_RESPONSES`` is left unchanged: that mode
            # captures live responses via ``store_response`` and assumes
            # linear traversal; pre-seeding from trace turns would inject
            # placeholder responses that the live-capture flow never wrote.
            if (
                self.context_mode == ConversationContextMode.DELTAS_WITH_RESPONSES
                and len(self.turn_list) < turn_index
            ):
                for missing_idx in range(len(self.turn_list), turn_index):
                    self.turn_list.append(self.conversation.turns[missing_idx])
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
        """Append the captured live assistant response Turn to ``turn_list``."""
        self.turn_list.append(response_turn)


class UserSessionManager:
    """User session manager for multi-turn processing.

    Manages user sessions for multi-turn processing.

    FORK-pin eviction
    -----------------
    FORK-mode DAG children seed their ``turn_list`` from the parent's
    session at ``create_and_store`` time, so the parent must stay cached
    until every FORK child that will ever seed from it has done so.
    The classic bug is: parent's final turn returns → worker calls
    ``evict(parent)`` → child credit arrives later → seed fails with
    ``RuntimeError`` (sticky-routing invariant violated).

    The fix tracks a per-parent live-FORK-child refcount:

    - ``create_and_store`` for a FORK child increments
      ``_fork_child_count[parent]``.
    - ``evict(child)`` decrements the parent's count. When the count hits
      zero and the parent is in ``_pending_eviction``, the parent is
      actually dropped from the cache.
    - ``evict(parent)`` for a session that declared FORK branches checks
      the current count: zero → drop immediately (nothing pending or
      in-flight), non-zero → mark ``_pending_eviction`` and keep cached.

    The live refcount alone is NOT enough: children are counted at
    ``create_and_store`` time, which runs on the worker when each child's
    credit is processed — and those are serialized. A single-turn child
    evicts the instant its only turn completes, so under serial dispatch
    child 0 can be created AND evicted (driving the live count back to zero)
    before child 1 is ever created. Popping the parent on that transient
    zero strands every later sibling with a "parent session not found on
    this worker" invariant violation. The cascade-evict is therefore gated
    on the parent's FULL DECLARED child set having been created
    (``_fork_created >= _fork_expected``), not merely on the live count
    reaching zero.
    """

    def __init__(self) -> None:
        self._cache: dict[str, UserSession] = {}
        self._default_context_mode: ConversationContextMode | None = None
        # FORK-pin refcount: parent x_correlation_id -> number of live
        # FORK children currently cached on this worker that will seed
        # from the parent's turn_list.
        self._fork_child_count: dict[str, int] = {}
        # FORK-pin: total FORK children DECLARED by a parent's branches
        # (parent x_correlation_id -> declared count), set when the parent
        # session is created. The cascade-evict only fires once this many
        # children have been created, so a parent is never dropped on a
        # transient refcount zero between serially-created single-turn
        # children (the wide-fan-out invariant violation).
        self._fork_expected: dict[str, int] = {}
        # FORK-pin: cumulative FORK children CREATED for a parent on this
        # worker (parent x_correlation_id -> count), compared against
        # _fork_expected to know when the full declared child set exists.
        self._fork_created: dict[str, int] = {}
        # Parents whose final turn has completed but still have one or
        # more live FORK children; evicted only when the refcount drops
        # to zero via a child eviction AND the full declared set was created.
        self._pending_eviction: set[str] = set()

    @property
    def default_context_mode(self) -> ConversationContextMode | None:
        """The dataset-level default context mode, if one was set by the loader."""
        return self._default_context_mode

    def set_default_context_mode(self, mode: ConversationContextMode | None) -> None:
        """Set the dataset-level default context mode from the loader."""
        self._default_context_mode = mode

    def create_and_store(
        self,
        x_correlation_id: str,
        conversation: Conversation,
        num_turns: int,
        url_index: int | None = None,
        parent_correlation_id: str | None = None,
        branch_mode: ConversationBranchMode = ConversationBranchMode.FORK,
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
            parent_correlation_id: Parent session's correlation id for DAG
                children; under FORK mode, used to seed ``turn_list`` from the
                parent (sticky routing co-locates them on this worker).
            branch_mode: DAG branch mode. FORK inherits the parent's accumulated
                ``turn_list`` (system/user messages + captured live assistant
                responses). SPAWN starts fresh. Ignored when
                ``parent_correlation_id`` is None.

        Raises:
            ValueError: If num_turns exceeds the actual conversation length.
            RuntimeError: If ``parent_correlation_id`` is set under FORK mode
                but the parent session is not cached on this worker
                (sticky-routing invariant violated).
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

        seed_turn_list: list[Turn] = []
        if (
            parent_correlation_id is not None
            and branch_mode == ConversationBranchMode.FORK
        ):
            parent_session = self.get(parent_correlation_id)
            if parent_session is None:
                raise RuntimeError(
                    f"FORK routing invariant violated: parent session "
                    f"{parent_correlation_id!r} not found on this worker. "
                    f"Sticky routing should have co-located the child "
                    f"{x_correlation_id!r}."
                )
            # Shallow copy: child owns its own list but shares Turn
            # references with the parent. This is a hot path (FORK fanout
            # runs at credit-return time) so deep-copying every turn would
            # cost. Invariant: Turn instances are treated as read-only
            # post-construction — sessions only ``append`` / ``reassign``
            # ``turn_list``, never mutate Turn fields in place. If a future
            # session op needs to mutate a Turn, deep-copy here first.
            seed_turn_list = list(parent_session.turn_list)

        user_session = UserSession(
            x_correlation_id=x_correlation_id,
            num_turns=num_turns,
            url_index=url_index,
            conversation=conversation,
            turn_list=seed_turn_list,
            context_mode=context_mode,
            parent_correlation_id=parent_correlation_id,
            branch_mode=branch_mode if parent_correlation_id is not None else None,
        )
        self.store(x_correlation_id, user_session)
        # A session that DECLARES FORK branches records how many FORK children
        # it will ever seed, so eviction can wait for the full set rather than
        # racing a transient refcount zero between serially-created children.
        fork_declared = sum(
            len(b.child_conversation_ids)
            for b in conversation.branches
            if b.mode == ConversationBranchMode.FORK
        )
        if fork_declared > 0:
            self._fork_expected[x_correlation_id] = fork_declared
        # FORK children bump the parent's live-child refcount AND the
        # cumulative created count so the parent stays cached (pinned) until
        # the whole declared child set has been created and has evicted.
        if (
            parent_correlation_id is not None
            and branch_mode == ConversationBranchMode.FORK
        ):
            self._fork_child_count[parent_correlation_id] = (
                self._fork_child_count.get(parent_correlation_id, 0) + 1
            )
            self._fork_created[parent_correlation_id] = (
                self._fork_created.get(parent_correlation_id, 0) + 1
            )
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

        Three cases, discriminated by the session's FORK-branch topology:

        1. **Plain session (no FORK branches, not a FORK child)**: pop
           immediately — nothing depends on this cache entry surviving.

        2. **FORK child (``parent_correlation_id`` set,
           ``branch_mode == FORK``)**: pop the child, then decrement the
           parent's live-child refcount. If the parent is in
           ``_pending_eviction`` and its refcount has now reached zero,
           drop the parent too (cascade).

        3. **FORK parent (conversation declares any FORK branch)**: mark
           ``_pending_eviction`` and leave cached, regardless of the
           current refcount. The parent's final turn returns on the
           worker's credit-return path BEFORE the orchestrator's child
           credits have been dispatched back to this worker, so the
           refcount is typically still zero at evict time even though
           children are imminent. Popping here would race the children's
           ``create_and_store`` sticky-routing lookup. The last child to
           evict cascades through case 2 and actually drops the parent.

        Note: a FORK parent whose full declared child set never materialises
        (orchestrator dispatch failed, or the ``--request-count`` cap truncated
        some children) keeps ``_fork_created < _fork_expected`` forever, so it
        stays pinned in ``_pending_eviction`` (plus its ``_fork_expected`` /
        ``_fork_created`` entries) for the lifetime of this session manager --
        which is the lifetime of the worker process; there is no per-phase
        reset. This matches the original permanent-pin behavior for that
        pathological case (the pre-refcount code already retained ``_cache`` /
        ``_pending_eviction`` for such parents); the common flow evicts once the
        DAG drains. Under a long run with many truncated DAG trees these entries
        accumulate -- bounding them (a phase-boundary reset or orchestrator-
        signalled GC) is tracked separately.

        Args:
            x_correlation_id: X-Correlation-ID header value
        """
        session = self._cache.get(x_correlation_id)
        if session is None:
            return

        is_fork_child = (
            session.parent_correlation_id is not None
            and session.branch_mode == ConversationBranchMode.FORK
        )
        is_fork_parent = any(
            b.mode == ConversationBranchMode.FORK for b in session.conversation.branches
        )

        if is_fork_parent:
            # Case 3: mark pending, leave cached. Children may not have
            # reached this worker yet; popping now would race their
            # sticky-routing seed lookup.
            self._pending_eviction.add(x_correlation_id)
            return

        # Case 1 or 2: drop from cache.
        self._cache.pop(x_correlation_id, None)
        self._pending_eviction.discard(x_correlation_id)
        self._fork_child_count.pop(x_correlation_id, None)
        self._fork_expected.pop(x_correlation_id, None)
        self._fork_created.pop(x_correlation_id, None)

        if is_fork_child:
            # Case 2: decrement parent's refcount. Only cascade-evict the
            # parent when (a) no live children remain AND (b) the parent's
            # FULL declared child set has been created. Without (b), a single
            # early-finishing child (e.g. a single-turn child under serial
            # dispatch) drives the live count to zero and would pop a parent
            # whose remaining siblings have not yet seeded — the wide-fan-out
            # "parent session not found on this worker" invariant violation.
            parent = session.parent_correlation_id
            remaining = self._fork_child_count.get(parent, 0) - 1
            all_created = self._fork_created.get(parent, 0) >= self._fork_expected.get(
                parent, 0
            )
            if remaining <= 0 and all_created:
                self._fork_child_count.pop(parent, None)
                self._fork_created.pop(parent, None)
                self._fork_expected.pop(parent, None)
                if parent in self._pending_eviction:
                    self._pending_eviction.discard(parent)
                    self._cache.pop(parent, None)
            else:
                # Either siblings are still live, or more declared children
                # are still to be created: keep the parent pinned.
                self._fork_child_count[parent] = max(remaining, 0)
