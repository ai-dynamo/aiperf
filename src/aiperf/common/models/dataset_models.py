# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar

from pydantic import Field

from aiperf.common.enums import MediaType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import MediaTypeT


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


class Turn(AIPerfBaseModel):
    """A dataset representation of a single turn within a conversation.

    A turn is a single interaction between a user and an AI assistant,
    and it contains timestamp, delay, and raw data that user sends in each turn.
    """

    timestamp: int | None = Field(
        default=None, description="Timestamp of the turn in milliseconds."
    )
    delay: int | None = Field(
        default=None,
        description="Amount of milliseconds to wait before sending the turn.",
    )
    model: str | None = Field(default=None, description="Model name used for the turn.")
    role: str | None = Field(default=None, description="Role of the turn.")
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


class Conversation(AIPerfBaseModel):
    """A dataset representation of a full conversation.

    A conversation is a sequence of turns between a user and an endpoint,
    and it contains the session ID and all the turns that consists the conversation.
    """

    turns: list[Turn] = Field(
        default=[], description="List of turns in the conversation."
    )
    session_id: str = Field(default="", description="Session ID of the conversation.")
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
