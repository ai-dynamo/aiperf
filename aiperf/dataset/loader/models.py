# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypeVar

from pydantic import Field, model_validator

from aiperf.common.enums import CustomDatasetType
from aiperf.common.models import AIPerfBaseModel, Audio, Image, Text


class SingleTurn(AIPerfBaseModel):
    """Defines the schema for single-turn data.

    User can use this format to quickly provide a custom single turn dataset.
    Each line in the file will be treated as a single turn conversation.

    The single turn type
      - supports multi-modal (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch_size > 1)
      - DOES NOT support multi-turn features (e.g. session_id)
    """

    type: Literal[CustomDatasetType.SINGLE_TURN] = CustomDatasetType.SINGLE_TURN

    # TODO (TL-89): investigate if we only want to support single field for each modality
    text: str | None = Field(None, description="Simple text string content")
    texts: list[str] | list[Text] | None = Field(
        None,
        description="List of text strings or Text objects format",
    )
    image: str | None = Field(None, description="Simple image string content")
    images: list[str] | list[Image] | None = Field(
        None,
        description="List of image strings or Image objects format",
    )
    audio: str | None = Field(None, description="Simple audio string content")
    audios: list[str] | list[Audio] | None = Field(
        None,
        description="List of audio strings or Audio objects format",
    )
    timestamp: int | None = Field(
        default=None, description="Timestamp of the turn in milliseconds."
    )
    delay: int | None = Field(
        default=None,
        description="Amount of milliseconds to wait before sending the turn.",
    )
    role: str | None = Field(default=None, description="Role of the turn.")

    @model_validator(mode="after")
    def validate_mutually_exclusive_fields(self) -> "SingleTurn":
        """Ensure mutually exclusive fields are not set together"""
        if self.text and self.texts:
            raise ValueError("text and texts cannot be set together")
        if self.image and self.images:
            raise ValueError("image and images cannot be set together")
        if self.audio and self.audios:
            raise ValueError("audio and audios cannot be set together")
        if self.timestamp and self.delay:
            raise ValueError("timestamp and delay cannot be set together")
        return self

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> "SingleTurn":
        """Ensure at least one modality is provided"""
        if not any(
            [self.text, self.texts, self.image, self.images, self.audio, self.audios]
        ):
            raise ValueError("At least one modality must be provided")
        return self


class MultiTurn(AIPerfBaseModel):
    """Defines the schema for multi-turn conversations.

    The multi-turn custom dataset
      - supports multi-modal data (e.g. text, image, audio)
      - supports multi-turn features (e.g. delay, sessions, etc.)
      - supports client-side batching for each data (e.g. batch size > 1)
    """

    type: Literal[CustomDatasetType.MULTI_TURN] = CustomDatasetType.MULTI_TURN

    session_id: str | None = Field(
        None, description="Unique identifier for the conversation session"
    )
    turns: list[SingleTurn] = Field(
        ..., description="List of turns in the conversation"
    )

    @model_validator(mode="after")
    def validate_turns_not_empty(self) -> "MultiTurn":
        """Ensure at least one turn is provided"""
        if not self.turns:
            raise ValueError("At least one turn must be provided")
        return self


class RandomPool(AIPerfBaseModel):
    """Defines the schema for random pool data entry.

    The random pool custom dataset
      - supports multi-modal data (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch size > 1)
      - supports named fields for each modality (e.g. text_field_a, text_field_b, etc.)
      - DOES NOT support multi-turn or its features (e.g. delay, sessions, etc.)
    """

    type: Literal[CustomDatasetType.RANDOM_POOL] = CustomDatasetType.RANDOM_POOL

    text: str | None = Field(None, description="Simple text string content")
    texts: list[str] | list[Text] | None = Field(
        None,
        description="List of text strings or Text objects format",
    )
    image: str | None = Field(None, description="Simple image string content")
    images: list[str] | list[Image] | None = Field(
        None,
        description="List of image strings or Image objects format",
    )
    audio: str | None = Field(None, description="Simple audio string content")
    audios: list[str] | list[Audio] | None = Field(
        None,
        description="List of audio strings or Audio objects format",
    )

    @model_validator(mode="after")
    def validate_mutually_exclusive_fields(self) -> "RandomPool":
        """Ensure mutually exclusive fields are not set together"""
        if self.text and self.texts:
            raise ValueError("text and texts cannot be set together")
        if self.image and self.images:
            raise ValueError("image and images cannot be set together")
        if self.audio and self.audios:
            raise ValueError("audio and audios cannot be set together")
        return self

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> "RandomPool":
        """Ensure at least one modality is provided"""
        if not any(
            [self.text, self.texts, self.image, self.images, self.audio, self.audios]
        ):
            raise ValueError("At least one modality must be provided")
        return self


class MooncakeTrace(AIPerfBaseModel):
    """Defines the schema for Mooncake trace data.

    See https://github.com/kvcache-ai/Mooncake for more details.

    Example:
    ```json
    {"timestamp": 1000, "input_length": 10, "output_length": 4, "hash_ids": [123, 456]}
    ```
    """

    type: Literal[CustomDatasetType.MOONCAKE_TRACE] = CustomDatasetType.MOONCAKE_TRACE

    input_length: int = Field(..., description="The input sequence length of a request")
    output_length: int = Field(
        ..., description="The output sequence length of a request"
    )
    hash_ids: list[int] = Field(..., description="The hash ids of a request")
    timestamp: int = Field(..., description="The timestamp of a request")


CustomDatasetT = TypeVar(
    "CustomDatasetT", bound=SingleTurn | MultiTurn | RandomPool | MooncakeTrace
)
"""A union type of all custom data types."""
