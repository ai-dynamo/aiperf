# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.pydantic_utils import AIPerfBaseModel, exclude_if_none


class Text(AIPerfBaseModel):
    name: str = Field(default="text", description="Name of the text field.")

    contents: list[str] = Field(
        default=[],
        description="List of text content. Supports batched text payload in a single turn.",
    )


class Image(AIPerfBaseModel):
    name: str = Field(default="image_url", description="Name of the image field.")

    contents: list[str] = Field(
        default=[],
        description="List of image content. Supports batched image payload in a single turn.",
    )


class Audio(AIPerfBaseModel):
    name: str = Field(default="input_audio", description="Name of the audio field.")

    contents: list[str] = Field(
        default=[],
        description="List of audio content. Supports batched audio payload in a single turn.",
    )


@exclude_if_none(["role"])
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
    role: str | None = Field(default=None, description="Role of the turn.")
    texts: list[Text] = Field(
        default=[], description="Collection of text data in the turn."
    )
    images: list[Image] = Field(
        default=[], description="Collection of image data in the turn."
    )
    audios: list[Audio] = Field(
        default=[], description="Collection of audio data in the turn."
    )


class Conversation(AIPerfBaseModel):
    """A dataset representation of a full conversation.

    A conversation is a sequence of turns between a user and an endpoint,
    and it contains the conversation ID and all the turns that consists
    the conversation.
    """

    turns: list[Turn] = Field(..., description="List of turns in the conversation.")
    id: str = Field(..., description="The conversation ID.")
