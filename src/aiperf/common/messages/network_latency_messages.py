# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails, NetworkLatencySample
from aiperf.common.types import MessageTypeT


class NetworkLatencyRecordMessage(BaseServiceMessage):
    """Message from the network latency probe manager to the records manager.

    Carries a single TCP-handshake RTT probe sample (success or failure) from
    one probe target. The ``error`` field is populated on a transport-level
    failure to push the sample (mirrors ServerMetricsRecordMessage); a failed
    probe itself is conveyed via ``sample.success == False``.
    """

    message_type: MessageTypeT = MessageType.NETWORK_LATENCY_RECORD

    collector_id: str = Field(
        description="The ID of the probe collector that produced the sample (typically host:port)"
    )
    sample: NetworkLatencySample | None = Field(
        default=None,
        description="The network latency probe sample",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="Transport error details if the sample could not be delivered.",
    )

    @property
    def valid(self) -> bool:
        """Whether a sample was delivered without a transport-level error."""
        return self.error is None and self.sample is not None
