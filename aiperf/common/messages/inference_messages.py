# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import (
    Field,
    SerializeAsAny,
)

from aiperf.common.enums import (
    MessageType,
)
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.types import MessageTypeT


class InferenceResultsMessage(BaseServiceMessage):
    """Message for a inference results."""

    message_type: MessageTypeT = MessageType.INFERENCE_RESULTS

    record: SerializeAsAny[RequestRecord] = Field(
        ..., description="The inference results record"
    )


class ParsedInferenceResultsMessage(BaseServiceMessage):
    """Message for a parsed inference results."""

    message_type: MessageTypeT = MessageType.PARSED_INFERENCE_RESULTS

    record: SerializeAsAny[ParsedResponseRecord] = Field(
        ..., description="The post process results record"
    )
