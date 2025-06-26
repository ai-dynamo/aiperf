#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


class Text(BaseModel):
    name: str = Field(default="text", description="Name of the text field.")

    content: list[str] = Field(
        default=[],
        description="List of text content. Supports batched text payload in a single turn.",
    )


class Image(BaseModel):
    name: str = Field(default="image_url", description="Name of the image field.")

    content: list[str] = Field(
        default=[],
        description="List of image content. Supports batched image payload in a single turn.",
    )


class Audio(BaseModel):
    name: str = Field(default="input_audio", description="Name of the audio field.")

    content: list[str] = Field(
        default=[],
        description="List of audio content. Supports batched audio payload in a single turn.",
    )


class Turn(BaseModel):
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

    text: list[Text] = Field(
        default=[], description="Collection of text data in each turn."
    )
    image: list[Image] = Field(
        default=[], description="Collection of image data in each turn."
    )
    audio: list[Audio] = Field(
        default=[], description="Collection of audio data in each turn."
    )


class Conversation(BaseModel):
    """A dataset representation of a full conversation.

    A conversation is a sequence of turns between a user and an endpoint,
    and it contains the session ID and all the turns that consists the conversation.
    """

    turns: list[Turn] = Field(
        default=[], description="List of turns in the conversation."
    )
    session_id: str = Field(default="", description="Session ID of the conversation.")
