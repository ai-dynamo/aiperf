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

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode
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
        root_correlation_id: The x_correlation_id of the depth-0 root of this
            session's tree. None on a depth-0 root (it is its own tree root);
            inherited by children/subchildren so per-tree accounting can key on
            ``effective_root_correlation_id``.
        start_turn_index: The turn index this session begins dispatching at.
            The agentic-replay engine starts a session at turn k_i (warmup) or
            resumes at k_i + 1 (profiling) without sending the leading turns.
        cache_bust_marker: Optional cache-bust marker for this session's
            trajectory TREE. The marker is a property of the tree
            (``root_correlation_id``): the depth-0 root and every SPAWN
            descendant (subagents, flat agents) share ONE marker, so the tree
            is a single prefix-cache domain and a session's own turns and
            subagents keep sharing cached prefixes; distinct trees (different
            traces / recycled sessions) get distinct markers. Set on SPAWN
            children to the tree root's shared marker (resolved by
            ``BranchOrchestrator._marker_for_root``); parent sessions populate
            this through the strategy
            (``AgenticReplayStrategy._build_turn_for_session``).
        cache_bust_target: Where to inject the marker. Mirrors the CLI knob;
            NONE when the feature is disabled.
    """

    conversation_id: str
    metadata: ConversationMetadata
    x_correlation_id: str
    agent_depth: int = 0
    parent_correlation_id: str | None = None
    root_correlation_id: str | None = None
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK
    start_turn_index: int = 0
    cache_bust_marker: str | None = None
    cache_bust_target: CacheBustTarget = CacheBustTarget.NONE

    @property
    def routing_key(self) -> str:
        """Sticky-routing key.

        Returns the parent's correlation_id when set (so FORK children share
        a worker with the parent), otherwise this session's own
        x_correlation_id.
        """
        return self.parent_correlation_id or self.x_correlation_id

    @property
    def effective_root_correlation_id(self) -> str:
        """Tree root id, defaulting to this session's own x_correlation_id.

        A root session is its own tree root; a child/subchild inherits the
        depth-0 root's id, set by the spawning code / snapshot seeding so
        per-tree session-slot accounting can key on it.
        """
        return self.root_correlation_id or self.x_correlation_id

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
            root_correlation_id=self.root_correlation_id,
            is_session_start=True,
            has_forks=first_meta.has_forks if first_meta is not None else False,
            has_branches=bool(first_meta.branch_ids)
            if first_meta is not None
            else False,
            branch_mode=self.branch_mode,
            cache_bust_marker=self.cache_bust_marker,
            cache_bust_target=self.cache_bust_target,
        )

    def build_turn_at_index(self, turn_index: int) -> TurnToSend:
        """Build a TurnToSend for an arbitrary turn within this session.

        Used by AgenticReplayStrategy to start a session at turn k_i (warmup)
        or to resume at k_i + 1 (profiling) without dispatching the leading
        turns.

        Raises IndexError if turn_index is out of range.
        """
        if turn_index < 0 or turn_index >= len(self.metadata.turns):
            raise IndexError(
                f"turn_index {turn_index} out of range for conversation "
                f"{self.conversation_id} with {len(self.metadata.turns)} turns"
            )
        meta = self.metadata.turns[turn_index]
        return TurnToSend(
            conversation_id=self.conversation_id,
            x_correlation_id=self.x_correlation_id,
            turn_index=turn_index,
            num_turns=len(self.metadata.turns),
            agent_depth=self.agent_depth,
            parent_correlation_id=self.parent_correlation_id,
            root_correlation_id=self.root_correlation_id,
            # build_turn_at_index is only used to START a session (warmup at
            # k_i, profiling resume at k_i+1, recycled at 0); continuations go
            # through TurnToSend.from_previous_credit. Mark it a session start so
            # the resumed root acquires a session slot + counts even at k_i > 0.
            is_session_start=True,
            has_forks=meta.has_forks if meta is not None else False,
            has_branches=bool(meta.branch_ids) if meta is not None else False,
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
        root_correlation_id: str | None = None,
        branch_mode: ConversationBranchMode = ConversationBranchMode.FORK,
        cache_bust_marker: str | None = None,
        cache_bust_target: CacheBustTarget = CacheBustTarget.NONE,
    ) -> SampledSession:
        """Build a SampledSession for a DAG child conversation.

        Under FORK mode, the returned session inherits sticky-routing from its
        parent via ``parent_correlation_id``; the credit router pins the child
        to the parent's worker, where ``UserSessionManager.create_and_store``
        seeds ``turn_list`` by cloning the parent's in-memory session.
        SPAWN-mode children start with a fresh context, but the sticky pin to
        the parent's correlation_id is preserved at this layer — routing
        freedom is enforced upstream by the orchestrator/router.

        ``root_correlation_id`` is the depth-0 root of the spawning parent's
        tree; the child inherits it so all descendants of one root share a
        single per-tree session-slot key. Defaults to ``parent_correlation_id``
        when not supplied (the live spawn path's parent is always the root).

        ``cache_bust_marker`` / ``cache_bust_target`` are resolved by the caller
        (BranchOrchestrator via ``_marker_for_root``) to the marker of the
        child's trajectory TREE, so every SPAWN descendant of one root shares
        the root's marker (the tree is one prefix-cache domain) while subagents
        in different traces get distinct markers — preventing cross-trace
        sharing of a server KV-cache prefix and artificially inflated hit rates.
        """
        metadata = self._metadata_lookup[child_conversation_id]
        return SampledSession(
            conversation_id=child_conversation_id,
            metadata=metadata,
            x_correlation_id=str(uuid.uuid4()),
            agent_depth=agent_depth,
            parent_correlation_id=parent_correlation_id,
            root_correlation_id=root_correlation_id or parent_correlation_id,
            branch_mode=branch_mode,
            cache_bust_marker=cache_bust_marker,
            cache_bust_target=cache_bust_target,
        )

    def start_pre_session_child(
        self,
        child_conversation_id: str,
        cache_bust_marker: str | None = None,
        cache_bust_target: CacheBustTarget = CacheBustTarget.NONE,
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

        ``cache_bust_marker`` / ``cache_bust_target`` are minted by the caller
        so background SPAWN children get the same per-session unique-marker
        treatment as parents.
        """
        metadata = self._metadata_lookup[child_conversation_id]
        return SampledSession(
            conversation_id=child_conversation_id,
            metadata=metadata,
            x_correlation_id=str(uuid.uuid4()),
            agent_depth=1,
            parent_correlation_id=None,
            branch_mode=ConversationBranchMode.SPAWN,
            cache_bust_marker=cache_bust_marker,
            cache_bust_target=cache_bust_target,
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
