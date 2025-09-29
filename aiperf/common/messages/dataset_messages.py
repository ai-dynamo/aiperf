# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.base_messages import Message
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import Conversation, exclude_if_none
from aiperf.common.models.dataset_models import Turn
from aiperf.common.types import MessageTypeT


@exclude_if_none("data_location", "index_location")
class DatasetInfoMessage(BaseServiceMessage):
    """
    Message containing dataset access information for workers.

    This message is sent by the dataset manager to inform workers about
    available dataset storage (memory-mapped files or shared memory blocks).
    """

    message_type: MessageTypeT = MessageType.DATASET_INFO

    data_location: str | None = Field(
        default=None,
        description="Location of the dataset data (file path or shared memory name)",
    )
    index_location: str | None = Field(
        default=None,
        description="Location of the dataset index (file path or shared memory name)",
    )
    dataset_size: int = Field(
        default=0, description="Number of conversations in the dataset"
    )
    enabled: bool = Field(
        default=False, description="Whether dataset access is enabled and available"
    )


@exclude_if_none("conversation_id", "credit_phase")
class ConversationRequestMessage(Message):
    """Message requesting a conversation from the dataset manager."""

    message_type: MessageTypeT = MessageType.CONVERSATION_REQUEST

    conversation_id: str | None = Field(
        default=None,
        description="ID of the specific conversation to request. If None, returns any conversation.",
    )
    credit_phase: int | None = Field(
        default=None, description="Credit phase associated with this request"
    )


class ConversationResponseMessage(Message):
    """Response message containing a conversation from the dataset manager."""

    message_type: MessageTypeT = MessageType.CONVERSATION_RESPONSE

    conversation: Conversation = Field(..., description="The requested conversation")


@exclude_if_none("turn_index")
class ConversationTurnRequestMessage(Message):
    """Message requesting a specific turn from a conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_TURN_REQUEST

    conversation_id: str = Field(
        ..., description="ID of the conversation containing the turn"
    )
    turn_index: int = Field(..., ge=0, description="Index of the turn to request")


class ConversationTurnResponseMessage(Message):
    """Response message containing a turn from a conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_TURN_RESPONSE

    turn: Turn = Field(..., description="The requested turn")


class DatasetConfiguredNotification(BaseServiceMessage):
    """Notification that the dataset has been configured and is ready."""

    message_type: MessageTypeT = MessageType.DATASET_CONFIGURED_NOTIFICATION


class DatasetTimingRequest(Message):
    """Request for dataset timing information."""

    message_type: MessageTypeT = MessageType.DATASET_TIMING_REQUEST


class DatasetTimingResponse(Message):
    """Response containing dataset timing information."""

    message_type: MessageTypeT = MessageType.DATASET_TIMING_RESPONSE

    timing_data: list[tuple[int, str]] = Field(
        default=[], description="List of (timestamp, conversation_id) tuples"
    )
