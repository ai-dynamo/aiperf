# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from aiperf.common.enums import MediaType
from aiperf.common.models.base_models import AIPerfBaseModel
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

    data_file_path: Path = Field(
        ...,
        description="Path to the memory-mapped data file containing serialized conversations.",
    )
    index_file_path: Path = Field(
        ...,
        description="Path to the memory-mapped index file for O(1) conversation lookups.",
    )
    conversation_count: int = Field(
        default=0,
        description="Number of conversations stored in the mmap files.",
    )
    total_size_bytes: int = Field(
        default=0,
        description="Total size of the data file in bytes.",
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

    def metadata(self) -> TurnMetadata:
        """Get the metadata of the turn."""
        return TurnMetadata(
            timestamp_ms=self.timestamp,
            delay_ms=self.delay,
        )

    def copy_with_stripped_media(self) -> "Turn":
        """Create a copy of this turn with multimodal data replaced by placeholders.

        This preserves text data (needed for tokenization) but replaces potentially
        large image/audio/video contents with small placeholder strings. This is
        more efficient than a full deep copy followed by stripping.

        Returns:
            A new Turn with stripped multimodal contents.
        """
        return Turn(
            model=self.model,
            role=self.role,
            timestamp=self.timestamp,
            delay=self.delay,
            max_tokens=self.max_tokens,
            texts=[Text(name=t.name, contents=list(t.contents)) for t in self.texts],
            images=[
                Image(
                    name=img.name,
                    contents=[f"image_{i}" for i in range(len(img.contents))],
                )
                for img in self.images
            ],
            audios=[
                Audio(
                    name=aud.name,
                    contents=[f"audio_{i}" for i in range(len(aud.contents))],
                )
                for aud in self.audios
            ],
            videos=[
                Video(
                    name=vid.name,
                    contents=[f"video_{i}" for i in range(len(vid.contents))],
                )
                for vid in self.videos
            ],
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

    def metadata(self) -> ConversationMetadata:
        """Get the metadata of the conversation."""
        return ConversationMetadata(
            conversation_id=self.session_id,
            turns=[turn.metadata() for turn in self.turns],
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
