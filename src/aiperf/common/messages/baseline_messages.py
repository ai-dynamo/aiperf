# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Messages for phase-scoped baseline collection."""

from __future__ import annotations

from pydantic import Field

from aiperf.common.enums import BaselineKind, MessageType
from aiperf.common.messages.base_messages import Message
from aiperf.common.types import MessageTypeT, PhaseKind


class PhaseBaselineRequestMessage(Message):
    """Broadcast to ask baseline collectors to scrape near a phase boundary."""

    message_type: MessageTypeT = MessageType.PHASE_BASELINE_REQUEST

    phase_id: str = Field(
        ..., description="Unique runtime ID pairing boundary requests."
    )
    phase_index: int | None = Field(
        default=None, ge=0, description="Absolute index in the ordered phases list."
    )
    profiling_index: int | None = Field(
        default=None, ge=0, description="Index among profiling-kind phases."
    )
    phase_name: str = Field(..., description="User-facing unique phase name.")
    phase_kind: PhaseKind | None = Field(
        default=None, description="Semantic phase kind: warmup or profiling."
    )
    kind: BaselineKind = Field(
        ..., description="START before credits are issued; END after returns drain."
    )
