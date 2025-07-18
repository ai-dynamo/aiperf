# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from pydantic import (
    Field,
)

from aiperf.common.enums import (
    CreditPhase,
    MessageType,
)
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import Conversation, Turn


class ConversationRequestMessage(BaseServiceMessage):
    """Message to request a full conversation by ID."""

    message_type: Literal[MessageType.CONVERSATION_REQUEST] = (
        MessageType.CONVERSATION_REQUEST
    )

    conversation_id: str | None = Field(
        default=None, description="The session ID of the conversation"
    )
    credit_phase: CreditPhase | None = Field(
        default=None,
        description="The type of credit phase (either warmup or profiling). If not provided, the timing manager will use the default credit phase.",
    )


class ConversationResponseMessage(BaseServiceMessage):
    """Message containing a full conversation."""

    message_type: Literal[MessageType.CONVERSATION_RESPONSE] = (
        MessageType.CONVERSATION_RESPONSE
    )
    conversation: Conversation = Field(..., description="The conversation data")


class ConversationTurnRequestMessage(BaseServiceMessage):
    """Message to request a single turn from a conversation."""

    message_type: Literal[MessageType.CONVERSATION_TURN_REQUEST] = (
        MessageType.CONVERSATION_TURN_REQUEST
    )

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

    message_type: Literal[MessageType.CONVERSATION_TURN_RESPONSE] = (
        MessageType.CONVERSATION_TURN_RESPONSE
    )

    turn: Turn = Field(..., description="The turn data")


class DatasetTimingRequest(BaseServiceMessage):
    """Message for a dataset timing request."""

    message_type: Literal[MessageType.DATASET_TIMING_REQUEST] = (
        MessageType.DATASET_TIMING_REQUEST
    )


class DatasetTimingResponse(BaseServiceMessage):
    """Message for a dataset timing response."""

    message_type: Literal[MessageType.DATASET_TIMING_RESPONSE] = (
        MessageType.DATASET_TIMING_RESPONSE
    )

    timing_data: list[tuple[int, str]] = Field(
        ...,
        description="The timing data of the dataset. Tuple of (timestamp, conversation_id)",
    )
