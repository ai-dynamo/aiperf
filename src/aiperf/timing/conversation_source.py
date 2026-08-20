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

from aiperf.common.enums import (
    CacheBustTarget,
    ConversationBranchMode,
    ConversationContextMode,
)
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.dataset.protocols import DatasetSamplingStrategyProtocol
from aiperf.timing.strategies.cache_bust import build_cache_bust_marker


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
    allow_worker_migration: bool = False
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
            allow_worker_migration=self.allow_worker_migration,
            agent_depth=self.agent_depth,
            parent_correlation_id=self.parent_correlation_id,
            root_correlation_id=self.root_correlation_id,
            is_session_start=True,
            has_forks=first_meta.has_forks if first_meta is not None else False,
            no_request=first_meta.no_request if first_meta is not None else False,
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
            allow_worker_migration=self.allow_worker_migration,
            agent_depth=self.agent_depth,
            parent_correlation_id=self.parent_correlation_id,
            root_correlation_id=self.root_correlation_id,
            # build_turn_at_index is only used to START a session (warmup at
            # k_i, profiling resume at k_i+1, recycled at 0); continuations go
            # through TurnToSend.from_previous_credit. Mark it a session start so
            # the resumed root acquires a session slot + counts even at k_i > 0.
            is_session_start=True,
            has_forks=meta.has_forks if meta is not None else False,
            no_request=meta.no_request if meta is not None else False,
            branch_mode=self.branch_mode,
            cache_bust_marker=self.cache_bust_marker,
            cache_bust_target=self.cache_bust_target,
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
        *,
        benchmark_id: str = "unknown",
        cache_bust_target: CacheBustTarget = CacheBustTarget.NONE,
    ) -> None:
        """Initialize conversation source."""
        self._dataset_metadata = dataset_metadata
        self._dataset_sampler = dataset_sampler
        self._benchmark_id = benchmark_id
        self._cache_bust_target = cache_bust_target
        self._cache_bust_pass = 0
        self._cache_bust_markers: dict[str, str | None] = {}
        self._metadata_lookup: dict[str, ConversationMetadata] = {
            conv.conversation_id: conv for conv in dataset_metadata.conversations
        }
        # Monotonic per-sample ordinal. ``next()`` is synchronous (no await), so
        # this increments atomically in the seeded sampler's call order -- a
        # STABLE per-instance id across runs under ``--random-seed``. Recorded
        # only for orchestrator roots (bounded to the graph-instance count), which
        # need a reproducible key for think-time sampling because their random
        # UUID ``x_correlation_id`` cannot seed a reproducible draw.
        self._sample_seq = 0
        self._orchestrator_ordinal: dict[str, int] = {}

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        """Dataset metadata."""
        return self._dataset_metadata

    def _marker_for_session(
        self, conversation_id: str, x_correlation_id: str, *, retain: bool
    ) -> str | None:
        """Mint one deterministic marker for an ordinary session instance.

        ``retain`` gates the ``_cache_bust_markers`` cache, which exists only so
        a caller-supplied correlation id resolves to the SAME marker on every
        lookup (user-centric reuses ``str(user_id)``; ``marker_for_correlation_id``
        reads it back). A correlation id minted internally here is a fresh uuid
        no caller can name, so retaining it would grow the dict without bound —
        one dead entry per sampled session for the length of the run.
        """
        if self._cache_bust_target == CacheBustTarget.NONE:
            return None
        if x_correlation_id in self._cache_bust_markers:
            return self._cache_bust_markers[x_correlation_id]
        marker = build_cache_bust_marker(
            self._benchmark_id,
            self._cache_bust_pass,
            self._cache_bust_pass,
            conversation_id or x_correlation_id,
            target=self._cache_bust_target,
        )
        self._cache_bust_pass += 1
        if retain:
            self._cache_bust_markers[x_correlation_id] = marker
        return marker

    def release_marker_for_correlation_id(self, correlation_id: str) -> None:
        """Drop a retained marker once its session reaches a terminal credit."""
        self._cache_bust_markers.pop(correlation_id, None)

    def next(self, x_correlation_id: str | None = None) -> SampledSession:
        """Sample next conversation and return a new session instance."""
        conversation_id = self._dataset_sampler.next_conversation_id()
        metadata = self._metadata_lookup[conversation_id]
        correlation_id = x_correlation_id or str(uuid.uuid4())
        session = SampledSession(
            conversation_id=conversation_id,
            metadata=metadata,
            x_correlation_id=correlation_id,
            allow_worker_migration=self._can_migrate_worker(conversation_id),
            cache_bust_marker=self._marker_for_session(
                conversation_id, correlation_id, retain=x_correlation_id is not None
            ),
            cache_bust_target=self._cache_bust_target,
        )
        seq = self._sample_seq
        self._sample_seq += 1
        # Store only when a sampled think-time distribution is present -- that is
        # the ONLY consumer of the ordinal. Fixed-think and fire-and-forget
        # orchestrators never read it, so storing them would leak unused entries.
        if metadata.is_orchestrator and metadata.think_time is not None:
            self._orchestrator_ordinal[session.x_correlation_id] = seq
        return session

    def sample_ordinal(self, x_correlation_id: str) -> int | None:
        """Deterministic sampling ordinal for an orchestrator root instance.

        Returns None for non-orchestrator sessions. Stable across runs under the
        same ``--random-seed`` (unlike the random-UUID ``x_correlation_id``), so
        it can key a reproducible per-instance think-time draw.
        """
        return self._orchestrator_ordinal.get(x_correlation_id)

    def forget_ordinal(self, x_correlation_id: str) -> None:
        """Drop a graph instance's stored ordinal once it reaches END, bounding
        the map to in-flight sampled orchestrators (no unbounded growth over a
        long duration run). Idempotent."""
        self._orchestrator_ordinal.pop(x_correlation_id, None)

    def get_metadata(self, conversation_id: str) -> ConversationMetadata:
        """Get metadata for a specific conversation."""
        if conversation_id not in self._metadata_lookup:
            raise KeyError(f"No metadata for conversation {conversation_id}")
        return self._metadata_lookup[conversation_id]

    def session_for_conversation(
        self, conversation_id: str, *, x_correlation_id: str | None = None
    ) -> SampledSession:
        """Build a session for a known conversation, preserving marker policy."""
        metadata = self.get_metadata(conversation_id)
        correlation_id = x_correlation_id or str(uuid.uuid4())
        return SampledSession(
            conversation_id=conversation_id,
            metadata=metadata,
            x_correlation_id=correlation_id,
            allow_worker_migration=self._can_migrate_worker(conversation_id),
            cache_bust_marker=self._marker_for_session(
                conversation_id, correlation_id, retain=x_correlation_id is not None
            ),
            cache_bust_target=self._cache_bust_target,
        )

    def marker_for_correlation_id(self, correlation_id: str) -> str | None:
        """Return the marker minted for a live ordinary session, if any."""
        return self._cache_bust_markers.get(correlation_id)

    def _can_migrate_worker(self, conversation_id: str) -> bool:
        """Whether dataset-authored responses can reconstruct this session."""
        metadata = self.get_metadata(conversation_id)
        context_mode = (
            metadata.context_mode
            or self._dataset_metadata.default_context_mode
            or ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        )
        return context_mode in {
            ConversationContextMode.DELTAS_WITH_RESPONSES,
            ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
        }

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
        """Build a SampledSession for a DAG child conversation."""
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
        """Build a SampledSession for a pre-session background SPAWN child."""
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
        """Get metadata for next turn after completed credit."""
        metadata = self.get_metadata(credit.conversation_id)
        next_index = credit.turn_index + 1

        if next_index >= len(metadata.turns):
            raise ValueError(
                f"No turn {next_index} in conversation {credit.conversation_id} "
                f"(only {len(metadata.turns)} turns exist)"
            )
        return metadata.turns[next_index]
