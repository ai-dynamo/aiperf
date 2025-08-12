# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import Conversation, Turn
from aiperf.common.types import MessageTypeT


class ProcessDatasetMessage(BaseServiceMessage):
    """Message for sending dataset processing requests to processors."""

    random_seed: int | None = Field(
        default=None, description="Random seed for the dataset generation"
    )


class ProcessSyntheticDatasetMessage(ProcessDatasetMessage):
    """Message for processing synthetic data."""

    message_type: MessageTypeT = MessageType.PROCESS_SYNTHETIC_DATASET
    num_conversations: int = Field(
        ..., description="Number of conversation to generate"
    )


class ProcessMooncakeTraceDatasetMessage(ProcessDatasetMessage):
    """Message for processing mooncake trace data."""

    message_type: MessageTypeT = MessageType.PROCESS_MOONCAKE_TRACE_DATASET
    dataset: list[tuple[str, Any]] = Field(
        ..., description="The Mooncake trace dataset"
    )


class ProcessMultiTurnDatasetMessage(ProcessDatasetMessage):
    """Message for processing multi-turn data."""

    message_type: MessageTypeT = MessageType.PROCESS_MULTI_TURN_DATASET
    dataset: list[tuple[str, Any]] = Field(..., description="The multi-turn dataset")


class ProcessSingleTurnDatasetMessage(ProcessDatasetMessage):
    """Message for processing single-turn data."""

    message_type: MessageTypeT = MessageType.PROCESS_SINGLE_TURN_DATASET
    dataset: list[tuple[str, Any]] = Field(..., description="The single-turn dataset")


class ProcessRandomPoolDatasetMessage(ProcessDatasetMessage):
    """Message for processing random pool data."""

    message_type: MessageTypeT = MessageType.PROCESS_RANDOM_POOL_DATASET
    dataset: list[tuple[str, Any]] = Field(..., description="The random pool dataset")


class ProcessDatasetResponseMessage(ProcessDatasetMessage):
    """Message for returning dataset processing responses."""

    message_type: MessageTypeT = MessageType.DATASET_RESULT

    generated_data: list[Conversation] = Field(
        default_factory=list, description="Generated conversations"
    )
    error_message: str | None = Field(
        default=None, description="Error message if job failed"
    )
    processing_time_ms: float | None = Field(
        default=None, description="Time taken to process the job in milliseconds"
    )


class ConversationRequestMessage(BaseServiceMessage):
    """Message to request a full conversation by ID."""

    message_type: MessageTypeT = MessageType.CONVERSATION_REQUEST

    conversation_id: str | None = Field(
        default=None, description="The session ID of the conversation"
    )
    credit_phase: CreditPhase | None = Field(
        default=None,
        description="The type of credit phase (either warmup or profiling). If not provided, the timing manager will use the default credit phase.",
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


class DatasetTimingRequest(BaseServiceMessage):
    """Message for a dataset timing request."""

    message_type: MessageTypeT = MessageType.DATASET_TIMING_REQUEST


class DatasetTimingResponse(BaseServiceMessage):
    """Message for a dataset timing response."""

    message_type: MessageTypeT = MessageType.DATASET_TIMING_RESPONSE

    timing_data: list[tuple[int, str]] = Field(
        ...,
        description="The timing data of the dataset. Tuple of (timestamp, conversation_id)",
    )


class DatasetConfiguredNotification(BaseServiceMessage):
    """Notification sent to notify other services that the dataset has been configured."""

    message_type: MessageTypeT = MessageType.DATASET_CONFIGURED_NOTIFICATION
