# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.common.models import MetricResult, NetworkLatencySample


@runtime_checkable
class NetworkLatencyProcessorProtocol(Protocol):
    """Protocol for results processors that consume network latency RTT samples.

    Separate from ResultsProcessorProtocol because RTT probe samples are
    structurally distinct from inference metric records.
    """

    async def process_network_latency_sample(
        self, sample: NetworkLatencySample
    ) -> None:
        """Process a single TCP-handshake RTT probe sample.

        Args:
            sample: NetworkLatencySample with the probe result (success or failure)
        """
        ...

    async def summarize(self) -> list[MetricResult]: ...
