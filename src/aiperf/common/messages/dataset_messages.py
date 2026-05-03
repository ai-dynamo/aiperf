# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field, SerializeAsAny, field_validator

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    Conversation,
    DatasetClientMetadata,
    DatasetMetadata,
    Turn,
)
from aiperf.common.types import MessageTypeT


class ConversationRequestMessage(BaseServiceMessage):
    """Message to request a full conversation by ID."""

    message_type: MessageTypeT = MessageType.CONVERSATION_REQUEST

    conversation_id: str = Field(..., description="The dataset conversation ID")
    credit_phase: CreditPhase | None = Field(
        default=None,
        description="The type of credit phase (either warmup or profiling). If not provided, the dataset manager will use the default credit phase.",
    )


class ConversationResponseMessage(BaseServiceMessage):
    """Message containing a full conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_RESPONSE
    conversation: Conversation = Field(..., description="The conversation data")


class ConversationTurnRequestMessage(BaseServiceMessage):
    """Message to request a single turn from a conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_TURN_REQUEST

    conversation_id: str = Field(
        ...,
        description="The ID of the conversation.",
    )
    turn_index: int = Field(
        ...,
        ge=0,
        description="The index of the turn in the conversation.",
    )


class ConversationTurnResponseMessage(BaseServiceMessage):
    """Message containing a single turn from a conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_TURN_RESPONSE

    turn: Turn = Field(..., description="The turn data")


class DatasetConfiguredNotification(BaseServiceMessage):
    """Notification sent to notify other services that the dataset has been configured.

    Contains two separate pieces of information:
    - metadata: Dataset structure (conversations, sampling strategy) for timing strategies
    - client_metadata: Client access info (e.g., mmap paths) for workers to read data
    """

    message_type: MessageTypeT = MessageType.DATASET_CONFIGURED_NOTIFICATION

    metadata: DatasetMetadata = Field(
        ...,
        description="Dataset structure metadata (conversations, timing) for timing strategies.",
    )
    client_metadata: SerializeAsAny[DatasetClientMetadata] = Field(
        ...,
        description="Client access metadata (e.g., mmap file paths) for workers to read dataset.",
    )

    @field_validator("client_metadata", mode="before")
    @classmethod
    def route_client_metadata(cls, v: Any) -> DatasetClientMetadata:
        """Route nested AutoRoutedModel field to correct subclass.

        Pydantic's nested model validation doesn't use AutoRoutedModel.from_json(),
        so we manually route dict inputs to the correct subclass based on client_type.
        """
        if isinstance(v, dict):
            return DatasetClientMetadata.from_json(v)
        return v
