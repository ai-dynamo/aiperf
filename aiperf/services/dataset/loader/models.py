# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from aiperf.common.dataset_models import Audio, Image, Text
from aiperf.common.enums import CustomDatasetType


class SingleTurn(BaseModel):
    """Defines the schema for single-turn data.

    User can use this format to quickly provide a custom single turn dataset.
    Each line in the file will be treated as a single turn conversation.

    The single turn type
      - supports multi-modal (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch_size > 1)
      - DOES NOT support multi-turn features (e.g. delay, sessions, etc.)
    """

    type: Literal[CustomDatasetType.SINGLE_TURN] = CustomDatasetType.SINGLE_TURN

    text: str | list[str] | list[Text] | None = Field(
        None,
        description="Text content - supports simple strings, lists of strings, or named modality format",
    )
    image: str | list[str] | list[Image] | None = Field(
        None,
        description="Image content - supports simple strings, lists of strings, or named modality format",
    )
    audio: str | list[str] | list[Audio] | None = Field(
        None,
        description="Audio content - supports simple strings, lists of strings, or named modality format",
    )
    role: str | None = Field(
        None, description="Role of the turn (e.g., 'user', 'assistant')"
    )
    delay: int | None = Field(
        None, description="Delay in milliseconds before sending this turn"
    )
    timestamp: int | None = Field(
        None, description="Timestamp of the turn in milliseconds"
    )

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> "SingleTurn":
        """Ensure at least one modality is provided"""
        if not any([self.text, self.image, self.audio]):
            raise ValueError(
                "At least one modality (text, image, or audio) must be provided"
            )
        return self

    @model_validator(mode="after")
    def validate_mutually_exclusive_fields(self) -> "SingleTurn":
        """Ensure timestamp and delay cannot be set together"""
        if self.timestamp and self.delay:
            raise ValueError("timestamp and delay cannot be set together")
        return self


class MultiTurn(BaseModel):
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


class MooncakeTrace(BaseModel):
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


CustomData = Annotated[
    SingleTurn | MooncakeTrace | MultiTurn,
    Field(discriminator="type"),
]
"""A union type of all custom data types."""
