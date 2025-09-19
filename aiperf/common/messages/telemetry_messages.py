# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails, TelemetryRecord
from aiperf.common.types import MessageTypeT

class TelemetryRecordsMessage(BaseServiceMessage):
    """Message from the telemetry data collector to the records manager to notify it
    of the telemetry records for a batch of GPU samples."""

    message_type: MessageTypeT = MessageType.TELEMETRY_RECORDS

    collector_id: str = Field(
        ..., description="The ID of the telemetry data collector that collected the records."
    )
    records: list[TelemetryRecord] = Field(
        ..., description="The telemetry records collected from GPU monitoring"
    )
    error: ErrorDetails | None = Field(
        default=None, description="The error details if telemetry collection failed."
    )

    @property
    def valid(self) -> bool:
        """Whether the telemetry collection was valid."""

        return self.error is None and len(self.records) > 0