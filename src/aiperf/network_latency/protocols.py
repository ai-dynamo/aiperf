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

    async def finalize(self) -> None:
        """Flush any buffered samples to disk before results are published.

        Called once by the records-manager BEFORE publishing
        ``ProcessRecordsResultMessage`` so the per-sample JSONL artifact is
        fully flushed before the controller writes the readiness marker.
        Default implementation is a no-op; processors that buffer to disk
        override to flush + close.
        """
        ...
