# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Messages for the phase baseline handshake.

PhaseBaselineRequestMessage is broadcast by SystemController; baseline-collector
services respond with PhaseBaselineAckMessage carrying success/error status.
"""

from pydantic import Field

from aiperf.common.enums import BaselineKind, MessageType
from aiperf.common.messages.base_messages import Message
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.types import MessageTypeT


class PhaseBaselineRequestMessage(Message):
    """Broadcast by SystemController to ask all baseline collectors to scrape."""

    message_type: MessageTypeT = MessageType.PHASE_BASELINE_REQUEST

    phase_id: str = Field(
        ..., description="UUID of the phase being gated; pairs request to ack."
    )
    phase_name: str = Field(
        ..., description="Human-readable phase name (warmup, profiling, ...)."
    )
    kind: BaselineKind = Field(
        ..., description="START before credits are issued; END after returns drain."
    )


class PhaseBaselineAckMessage(BaseServiceMessage):
    """Sent by a baseline collector after attempting collect_baseline()."""

    message_type: MessageTypeT = MessageType.PHASE_BASELINE_ACK

    phase_id: str = Field(..., description="Phase ID this ack is for.")
    kind: BaselineKind = Field(..., description="START or END.")
    success: bool = Field(
        ...,
        description="False if collect_baseline() raised; coordinator still counts as ack.",
    )
    error: str | None = Field(
        default=None, description="Error string when success=False."
    )
