# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import Field

from aiperf.accuracy.models import ProcessAccuracyResult
from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.types import MessageTypeT


class ProcessAccuracyResultMessage(BaseServiceMessage):
    """Message containing a processed accuracy summary - mirrors ProcessServerMetricsResultMessage."""

    message_type: MessageTypeT = MessageType.PROCESS_ACCURACY_RESULT

    accuracy_result: ProcessAccuracyResult = Field(
        description="The processed accuracy summary results"
    )
