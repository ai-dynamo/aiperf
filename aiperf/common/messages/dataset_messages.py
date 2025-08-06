# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import AIPerfBaseModel, Conversation, Turn
from aiperf.common.types import MessageTypeT


# NEW: Dataset Processor Messages for parallel data generation
class DatasetJobInfo(AIPerfBaseModel):
    """Specification for a dataset generation job."""

    job_id: str = Field(..., description="Unique identifier for this job")
    num_turns: int = Field(..., description="Number of conversation turns to generate")
    tokens_per_turn: int = Field(..., description="Target number of tokens per turn")
    conversation_id: str | None = Field(
        default=None, description="Conversation ID if extending existing"
    )
    generation_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional generation parameters"
    )


class DatasetJobResult(AIPerfBaseModel):
    """Result of a dataset generation job."""

    job_id: str = Field(
        ..., description="Job identifier that this result corresponds to"
    )
    success: bool = Field(..., description="Whether the job completed successfully")
    generated_data: list[dict[str, Any]] = Field(
        default_factory=list, description="Generated conversation turns or data"
    )
    error_message: str | None = Field(
        default=None, description="Error message if job failed"
    )
    processing_time_ms: float | None = Field(
        default=None, description="Time taken to process the job in milliseconds"
    )


class DatasetJobMessage(BaseServiceMessage):
    """Message for sending dataset generation requests to processors."""

    message_type: MessageTypeT = MessageType.DATASET_JOB

    info: DatasetJobInfo = Field(..., description="The dataset generation job info")


class DatasetResultMessage(BaseServiceMessage):
    """Message for returning dataset generation responses."""

    message_type: MessageTypeT = MessageType.DATASET_RESULT

    result: DatasetJobResult = Field(
        ..., description="The dataset generation job result"
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
