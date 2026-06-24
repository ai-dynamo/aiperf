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
    TurnInputKind,
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
    api_time_ms: int | float | None = Field(
        default=None,
        description=(
            "Recorded server processing duration of this turn in milliseconds "
            "(the capture's per-request api_time). With timestamp_ms it gives the "
            "turn's recorded interval [timestamp_ms, timestamp_ms + api_time_ms], "
            "which happens-before completion gating uses to derive cross-turn "
            "predecessors and the end-to-start residual. A duration, not warped "
            "(only inter-request idle gaps are compressed). None for loaders "
            "without per-request timing."
        ),
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
    raw_messages_count: int | None = Field(
        default=None,
        description=(
            "Number of OpenAI-compatible raw messages on the source Turn. "
            "None means Turn.raw_messages is None; zero means an explicit empty "
            "messages delta."
        ),
    )
    theoretical_prefix_cache_hit_blocks: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Number of leading hash-id blocks that would be prefix-cache hits "
            "for this turn under an infinite per-session cache. None when the "
            "dataset loader did not provide hash-block metadata."
        ),
    )
    theoretical_prefix_cache_total_blocks: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Number of hash-id blocks considered for theoretical prefix-cache "
            "hit accounting. Pairs with theoretical_prefix_cache_hit_blocks."
        ),
    )
    input_kind: TurnInputKind | None = Field(
        default=None,
        description=(
            "Classification of what produced this turn's new input: genuine "
            "user/agent text input vs tool-result continuation. None when the "
            "dataset loader did not provide the signal."
        ),
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
    api_time_ms: int | float | None = Field(
        default=None,
        description=(
            "Recorded server processing duration of this turn in milliseconds "
            "(capture per-request api_time). Pairs with timestamp to give the "
            "recorded interval used by happens-before completion gating. A "
            "duration (not warped). None for loaders without per-request timing."
        ),
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate for this turn."
    )
    raw_messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pre-formatted OpenAI-compatible messages array. "
        "When set, bypasses normal turn-based message construction in endpoints. "
        "Typed list[dict[str, Any]] rather than a narrower TypedDict because callers "
        "such as MooncakeTrace pass the full OpenAI message spec, which includes "
        "tool-call messages, assistant messages with tool_calls, and multi-modal "
        "content arrays — shapes that do not fit a single narrow TypedDict.",
    )
    raw_tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pre-formatted OpenAI-compatible tool definitions. "
        "When set alongside raw_messages, injected into the API payload.",
    )
    reset_context: bool = Field(
        default=False,
        description=(
            "When True, the endpoint formatter discards messages accumulated "
            "from prior turns in this conversation before applying this turn's "
            "raw_messages. Used by delta-encoded multi-turn conversations to "
            "express a non-monotonic context change (e.g. weka's mid-segment "
            "LCP cut, or any source that needs to rewrite an earlier prefix). "
            "Has no effect when raw_messages is None or when the surrounding "
            "Conversation.context_mode is a MESSAGE_ARRAY mode (each turn "
            "already carries a self-contained array)."
        ),
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
    audio_duration_seconds: float | None = Field(
        default=None,
        description="Duration of the audio content in seconds. Used by ASR-specific "
        "metrics like RTFx. Set by ASR dataset loaders.",
    )
    theoretical_prefix_cache_hit_blocks: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Number of leading hash-id blocks that would hit an infinite "
            "per-session prefix cache for this turn. Set by trace loaders that "
            "already walk hash_ids during reconstruction."
        ),
    )
    theoretical_prefix_cache_total_blocks: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Number of hash-id blocks considered for theoretical prefix-cache "
            "hit accounting for this turn."
        ),
    )
    input_kind: TurnInputKind | None = Field(
        default=None,
        description=(
            "Classification of what produced this turn's new input: genuine "
            "user/agent text input vs tool-result continuation. Set by trace "
            "loaders whose source records the signal (weka input_types/stop); "
            "None otherwise."
        ),
    )

    def metadata(self) -> TurnMetadata:
        """Get the metadata of the turn."""
        return TurnMetadata(
            timestamp_ms=self.timestamp,
            delay_ms=self.delay,
            api_time_ms=self.api_time_ms,
            branch_ids=self.branch_ids,
            prerequisites=self.prerequisites,
            raw_messages_count=None
            if self.raw_messages is None
            else len(self.raw_messages),
            theoretical_prefix_cache_hit_blocks=(
                self.theoretical_prefix_cache_hit_blocks
            ),
            theoretical_prefix_cache_total_blocks=(
                self.theoretical_prefix_cache_total_blocks
            ),
            input_kind=self.input_kind,
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
    system_message: str | None = Field(
        default=None,
        description=(
            "Optional shared system message prepended to the first request. "
            "Timing strategies use this to decide whether an otherwise empty "
            "per-turn raw-message delta can still start a valid request."
        ),
    )
    user_context_message: str | None = Field(
        default=None,
        description=(
            "Optional per-conversation user context prepended to the first request. "
            "Timing strategies use this to decide whether an otherwise empty "
            "per-turn raw-message delta can still start a valid request."
        ),
    )
    branches: list[ConversationBranchInfo] = Field(
        default_factory=list,
        description="Branch descriptors for this conversation (DAG projection).",
    )
    is_root: bool = Field(
        default=True,
        description="Whether this conversation is a DAG root (eligible for sampling). "
        "Non-root DAG children are reachable only via branches from their parent.",
    )
    agent_depth: int = Field(
        default=0,
        description="DAG nesting level (0 = root). Populated by DAG loaders.",
    )
    subagent_type: SubagentType | None = Field(
        default=None,
        description="Optional sub-agent classification (EXPLORE/GENERAL/PLAN) for metrics/routing.",
    )
    parent_conversation_id: str | None = Field(
        default=None,
        description="For DAG children: the parent conversation ID.",
    )
    accuracy_ground_truth: str | None = Field(
        default=None,
        description="Ground-truth answer for this conversation (accuracy mode only). "
        "Set by AccuracyDatasetLoader; None for all other dataset types.",
    )
    accuracy_task: str | None = Field(
        default=None,
        description="Benchmark sub-task name for this conversation (accuracy mode only). "
        "Set by AccuracyDatasetLoader; None for all other dataset types.",
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
    is_root: bool = Field(
        default=True,
        description="Whether this conversation is a DAG root (eligible for sampling). "
        "Non-root DAG children are reachable only via branches from their parent.",
    )
    agent_depth: int = Field(
        default=0,
        description="DAG nesting level (0 = root). Populated by DAG loaders.",
    )
    subagent_type: SubagentType | None = Field(
        default=None,
        description="Optional sub-agent classification (EXPLORE/GENERAL/PLAN) for metrics/routing.",
    )
    parent_conversation_id: str | None = Field(
        default=None,
        description="For DAG children: the parent conversation ID.",
    )
    accuracy_ground_truth: str | None = Field(
        default=None,
        description="Ground-truth answer for this conversation (accuracy mode only). "
        "Propagated to ConversationMetadata so processors receive it via "
        "DatasetConfiguredNotification without re-loading the benchmark.",
    )
    accuracy_task: str | None = Field(
        default=None,
        description="Benchmark sub-task name for this conversation (accuracy mode only). "
        "Propagated to ConversationMetadata so processors receive it via "
        "DatasetConfiguredNotification without re-loading the benchmark.",
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
                turn.metadata().model_copy(update={"has_forks": has_forks})
            )
        return ConversationMetadata(
            conversation_id=self.session_id,
            turns=turn_metas,
            system_message=self.system_message,
            user_context_message=self.user_context_message,
            branches=self.branches,
            is_root=self.is_root,
            agent_depth=self.agent_depth,
            subagent_type=self.subagent_type,
            parent_conversation_id=self.parent_conversation_id,
            accuracy_ground_truth=self.accuracy_ground_truth,
            accuracy_task=self.accuracy_task,
        )

    def to_metadata(self) -> "ConversationMetadata":
        """Project this Conversation into its DatasetMetadata form.

        Used by loaders to invoke validate_for_orchestrator_v1 without
        round-tripping through DatasetManager.
        """
        return ConversationMetadata(
            conversation_id=self.session_id,
            turns=[t.metadata() for t in self.turns],
            system_message=self.system_message,
            user_context_message=self.user_context_message,
            branches=list(self.branches),
            is_root=self.is_root,
            agent_depth=self.agent_depth,
            subagent_type=self.subagent_type,
            parent_conversation_id=self.parent_conversation_id,
            accuracy_ground_truth=self.accuracy_ground_truth,
            accuracy_task=self.accuracy_task,
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
