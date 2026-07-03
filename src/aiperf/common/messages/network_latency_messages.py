# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails, NetworkLatencySample


class NetworkLatencyRecordMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.NETWORK_LATENCY_RECORD.value
):
    """Message from the network latency probe manager to the records manager.

    Carries a single TCP-handshake RTT probe sample (success or failure) from
    one probe target. The ``error`` field is populated on a transport-level
    failure to push the sample; a failed probe itself is conveyed via
    ``sample.success == False``.
    """

    collector_id: str
    sample: NetworkLatencySample | None = None
    error: ErrorDetails | None = None

    @property
    def valid(self) -> bool:
        """Whether a sample was delivered without a transport-level error."""
        return self.error is None and self.sample is not None
