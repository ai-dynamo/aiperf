# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    MediaType,
    MemoryMapFormat,
)
from aiperf.common.enums.enums import SubagentType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.prerequisites import TurnPrerequisite
from aiperf.common.types import MediaTypeT
from aiperf.plugin.enums import DatasetClientStoreType, DatasetSamplingStrategy


class DatasetClientMetadata(AIPerfBaseModel):
    """Base class for dataset client access metadata.

    Uses discriminated union pattern based on client_type for extensibility.
    Workers receive this metadata to know how to access the dataset backing store.
    """

    discriminator_field: ClassVar[str] = "client_type"

    client_type: DatasetClientStoreType = Field(
        ...,
        description="The type of client store to use for dataset access.",
    )


class MemoryMapClientMetadata(DatasetClientMetadata):
    """Client metadata for memory-mapped dataset access.

    Contains paths to mmap files that workers use for zero-copy,
    O(1) conversation lookups.
    """

    client_type: DatasetClientStoreType = DatasetClientStoreType.MEMORY_MAP

    format: MemoryMapFormat = Field(
        default=MemoryMapFormat.CONVERSATION,
        description="Storage format of the memory-mapped dataset files.",
    )
    data_file_path: Path = Field(
        ...,
        description="Path to the data file. Points to dataset.dat (local) or dataset.dat.zst (k8s).",
    )
    index_file_path: Path = Field(
        ...,
        description="Path to the index file. Points to index.dat (local) or index.dat.zst (k8s).",
    )
    conversation_count: int = Field(
        default=0,
        description="Number of conversations stored in the mmap files.",
    )
    total_size_bytes: int = Field(
        default=0,
        description="Total uncompressed size of the data file in bytes.",
    )
    compressed: bool = Field(
        default=False,
        description="Whether data/index files are zstd-compressed (k8s compress_only mode).",
    )
    compressed_size_bytes: int = Field(
        default=0,
        description="Size of the compressed data file in bytes. 0 when not compressed.",
    )


class Media(AIPerfBaseModel):
    """Base class for all media fields. Contains name and contents of the media data."""

    name: str = Field(default="", description="Name of the media field.")

    contents: list[str] = Field(
        default=[],
        description="List of media contents. Supports batched media payload in a single turn.",
    )


class Text(Media):
    """Media that contains text/prompt data."""

    media_type: ClassVar[MediaTypeT] = MediaType.TEXT


class Image(Media):
    """Media that contains image data."""

    media_type: ClassVar[MediaTypeT] = MediaType.IMAGE


class Audio(Media):
    """Media that contains audio data."""

    media_type: ClassVar[MediaTypeT] = MediaType.AUDIO


class Video(Media):
    """Media that contains video data."""

    media_type: ClassVar[MediaTypeT] = MediaType.VIDEO


class TurnMetadata(AIPerfBaseModel):
    """Metadata of a turn."""

    timestamp_ms: int | float | None = Field(
        default=None,
        description="The absolute timestamp of the turn in milliseconds.",
    )
    delay_ms: int | float | None = Field(
        default=None,
        description="The delay of the turn in the conversation (in milliseconds).",
    )
    branch_ids: list[str] = Field(
        default_factory=list,
        description="Branch IDs triggered after this turn completes (DAG projection).",
    )
    has_forks: bool = Field(
        default=False,
        description="True if this turn triggers any FORK-mode branch. Stamped at load time.",
    )
    prerequisites: list[TurnPrerequisite] = Field(
        default_factory=list,
        description="Conditions gating dispatch of this turn (DAG projection).",
    )


class Turn(AIPerfBaseModel):
    """A dataset representation of a single turn within a conversation.

    A turn is a single interaction between a user and an AI assistant,
    and it contains timestamp, delay, and raw data that user sends in each turn.
    """

    model: str | None = Field(default=None, description="Model name used for the turn.")
    role: str | None = Field(default=None, description="Role of the turn.")
    timestamp: int | float | None = Field(
        default=None,
        description="The absolute timestamp of the turn in milliseconds.",
    )
    delay: int | float | None = Field(
        default=None,
        description="The delay of the turn in the conversation (in milliseconds).",
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate for this turn."
    )
    raw_messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pre-formatted OpenAI-compatible messages array. "
        "When set, bypasses normal turn-based message construction in endpoints.",
    )
    raw_tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pre-formatted OpenAI-compatible tool definitions. "
        "When set alongside raw_messages, injected into the API payload.",
    )
    texts: list[Text] = Field(
        default=[], description="Collection of text data in each turn."
    )
    images: list[Image] = Field(
        default=[], description="Collection of image data in each turn."
    )
    audios: list[Audio] = Field(
        default=[], description="Collection of audio data in each turn."
    )
    videos: list[Video] = Field(
        default=[], description="Collection of video data in each turn."
    )
    raw_payload: dict[str, Any] | None = Field(
        default=None,
        description="Complete pre-built API request payload for verbatim replay. "
        "When set, bypasses all endpoint payload construction (format_payload) "
        "and sends this dict directly to the transport.",
    )
    extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Non-native per-turn request-body fields (temperature, top_p, "
        "seed, stop, vendor tunables like ignore_eos/min_tokens, ...). Merged "
        "into the top level of the chat-completions payload at dispatch time, "
        "matching the OpenAI SDK's extra_body convention.",
    )
    branch_ids: list[str] = Field(
        default_factory=list,
        description="Branch IDs triggered after this turn completes (DAG authoring).",
    )
    prerequisites: list[TurnPrerequisite] = Field(
        default_factory=list,
        description="Conditions gating dispatch of this turn (DAG authoring). "
        "Attached to the gated turn; resolved against branch_ids declared on prior turns.",
    )

    def metadata(self) -> TurnMetadata:
        """Get the metadata of the turn."""
        return TurnMetadata(
            timestamp_ms=self.timestamp,
            delay_ms=self.delay,
            branch_ids=self.branch_ids,
            prerequisites=self.prerequisites,
        )


class ConversationMetadata(AIPerfBaseModel):
    """Metadata of a conversation."""

    conversation_id: str = Field(
        ...,
        description="The ID of the conversation.",
    )
    turns: list[TurnMetadata] = Field(
        default_factory=list,
        description="The metadata of the turns in the conversation.",
    )
    branches: list[ConversationBranchInfo] = Field(
        default_factory=list,
        description="Branch descriptors for this conversation (DAG projection).",
    )
    agent_depth: int = Field(
        default=0,
        description="Static DAG nesting level — 0 for sampleable roots, "
        "``parent_depth + 1`` for fork-spawned descendants. Stamped at "
        "load time by the dag_jsonl loader's topology walk; non-DAG "
        "conversations stay at the default 0. The sampler treats "
        "``agent_depth == 0`` as the root predicate (children are seeded "
        "from their parent's worker context, never sampled directly).",
    )
    subagent_type: SubagentType | None = Field(
        default=None,
        description="Optional sub-agent classification (EXPLORE/GENERAL/PLAN) for metrics/routing.",
    )
    parent_conversation_id: str | None = Field(
        default=None,
        description="For DAG children: the parent conversation ID.",
    )


class DatasetMetadata(AIPerfBaseModel):
    """Metadata of a dataset's structure.

    Contains dataset structure information (conversations, timing) used by
    timing strategies to schedule requests. Does NOT contain data access
    metadata - that's in DatasetClientMetadata (sent separately in
    DatasetConfiguredNotification).
    """

    conversations: list[ConversationMetadata] = Field(
        default_factory=list,
        description="The conversation metadata of the dataset.",
    )
    sampling_strategy: DatasetSamplingStrategy = Field(
        ...,
        description="The sampling strategy to use when choosing conversations from the dataset.",
    )
    has_timing_data: bool = Field(
        default=False,
        description="Whether the dataset has timing data (timestamps/delays in turns).",
    )
    default_context_mode: ConversationContextMode | None = Field(
        default=None,
        description="Dataset-level default for how prior turns are accumulated. "
        "Set by the loader based on dataset format semantics. "
        "Individual conversations can override this via their own context_mode field.",
    )

    @field_validator("default_context_mode")
    @classmethod
    def _reject_unimplemented_context_mode(
        cls,
        v: ConversationContextMode | None,
    ) -> ConversationContextMode | None:
        if v == ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES:
            raise ValueError(
                f"{ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES} is not yet supported"
            )
        return v

    @cached_property
    def total_turn_count(self) -> int:
        """Get the total number of turns in the dataset."""
        return sum(len(conversation.turns) for conversation in self.conversations)

    @cached_property
    def average_turn_count(self) -> float:
        """Get the average number of turns across all conversations in the dataset."""
        if len(self.conversations) == 0:
            return 0
        return self.total_turn_count / len(self.conversations)


class Conversation(AIPerfBaseModel):
    """A dataset representation of a full conversation.

    A conversation is a sequence of turns between a user and an endpoint,
    and it contains the session ID and all the turns that consists the conversation.
    """

    session_id: str = Field(
        default="", description="Unique identifier for the conversation."
    )
    context_mode: ConversationContextMode | None = Field(
        default=None,
        description="How prior turns are accumulated for this conversation. "
        "When None, inherits the dataset-level default.",
    )

    @field_validator("context_mode")
    @classmethod
    def _reject_unimplemented_context_mode(
        cls,
        v: ConversationContextMode | None,
    ) -> ConversationContextMode | None:
        if v == ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES:
            raise ValueError(
                f"{ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES} is not yet supported"
            )
        return v

    turns: list[Turn] = Field(
        default=[], description="List of turns in the conversation."
    )
    system_message: str | None = Field(
        default=None,
        description="Optional shared system message prepended to the first turn. "
        "Identical across all conversations when using --shared-system-prompt-length.",
    )
    user_context_message: str | None = Field(
        default=None,
        description="Optional per-conversation user context prepended to the first turn. "
        "Unique for each conversation when using --user-context-prompt-length.",
    )
    branches: list[ConversationBranchInfo] = Field(
        default_factory=list,
        description="Branch descriptors for this conversation (DAG authoring).",
    )
    agent_depth: int = Field(
        default=0,
        description="Static DAG nesting level — 0 for sampleable roots, "
        "``parent_depth + 1`` for fork-spawned descendants. Stamped at "
        "load time by the dag_jsonl loader's topology walk; non-DAG "
        "conversations stay at the default 0. ``agent_depth == 0`` is "
        "the root predicate (children are seeded from their parent's "
        "worker context, never sampled directly).",
    )
    subagent_type: SubagentType | None = Field(
        default=None,
        description="Optional sub-agent classification (EXPLORE/GENERAL/PLAN) for metrics/routing.",
    )
    parent_conversation_id: str | None = Field(
        default=None,
        description="For DAG children: the parent conversation ID.",
    )

    def metadata(self) -> ConversationMetadata:
        """Get the metadata of the conversation."""
        branches_by_id = {b.branch_id: b for b in self.branches}
        turn_metas: list[TurnMetadata] = []
        for turn in self.turns:
            triggered = [
                branches_by_id[bid] for bid in turn.branch_ids if bid in branches_by_id
            ]
            has_forks = any(b.mode == ConversationBranchMode.FORK for b in triggered)
            turn_metas.append(
                TurnMetadata(
                    timestamp_ms=turn.timestamp,
                    delay_ms=turn.delay,
                    branch_ids=turn.branch_ids,
                    has_forks=has_forks,
                    prerequisites=turn.prerequisites,
                )
            )
        return ConversationMetadata(
            conversation_id=self.session_id,
            turns=turn_metas,
            branches=self.branches,
            agent_depth=self.agent_depth,
            subagent_type=self.subagent_type,
            parent_conversation_id=self.parent_conversation_id,
        )


class SessionPayloads(AIPerfBaseModel):
    """A single session, with its session ID and a list of formatted payloads (one per turn)."""

    session_id: str | None = Field(
        default=None, description="Session ID of the conversation."
    )
    payloads: list[dict[str, Any]] = Field(
        default=[],
        description="List of formatted payloads in the session (one per turn). These have been formatted for the model and endpoint.",
    )


class InputsFile(AIPerfBaseModel):
    """A list of all dataset sessions. Each session contains a list of formatted payloads (one per turn).
    This is similar to the format used by GenAI-Perf for the inputs.json file.
    """

    data: list[SessionPayloads] = Field(
        default=[], description="List of all dataset sessions."
    )
