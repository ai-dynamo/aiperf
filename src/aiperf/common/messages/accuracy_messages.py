# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import Field

from aiperf.accuracy.models import AccuracyRecordsData, ProcessAccuracyResult
from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails
from aiperf.common.types import MessageTypeT


class AccuracyRecordsMessage(BaseServiceMessage):
    """Message from a record processor to the records manager carrying graded
    accuracy records on the dedicated ``accuracy`` channel.
    """

    message_type: MessageTypeT = MessageType.ACCURACY_RECORD

    records: list[AccuracyRecordsData] = Field(
        default_factory=list,
        description="The graded accuracy records produced by the record processor",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if grading/record production failed.",
    )


class ProcessAccuracyResultMessage(BaseServiceMessage):
    """Message containing a processed accuracy summary - mirrors ProcessServerMetricsResultMessage."""

    message_type: MessageTypeT = MessageType.PROCESS_ACCURACY_RESULT

    accuracy_result: ProcessAccuracyResult = Field(
        description="The processed accuracy summary results"
    )
