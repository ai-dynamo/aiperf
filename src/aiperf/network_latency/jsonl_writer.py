# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models import NetworkLatencySample
from aiperf.common.models.record_models import MetricResult
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class NetworkLatencyJSONLWriter(
    BaseMetricsProcessor,
    BufferedJSONLWriterMixin[NetworkLatencySample],
):
    """Exports per-sample TCP-handshake RTT probes to a JSONL artifact.

    Writes one JSON line per probe (success or failure) to
    ``profile_export_network_latency.jsonl`` (respecting --profile-export-prefix
    with the ``_network_latency.jsonl`` suffix). Self-disables unless network
    latency calibration is actively probing (enabled and no mean_ms).
    """

    def __init__(
        self,
        run: BenchmarkRun,
        **kwargs,
    ) -> None:
        if not run.cfg.network_latency.should_probe:
            raise PostProcessorDisabled(
                "Network latency JSONL export disabled: probing not enabled "
                "(requires --network-latency-automatic and no mean_ms)"
            )

        output_file = run.cfg.artifacts.network_latency_export_jsonl_file

        super().__init__(
            run=run,
            output_file=output_file,
            batch_size=Environment.NETWORK_LATENCY.EXPORT_BATCH_SIZE,
            **kwargs,
        )

        self.info(f"Network latency JSONL export enabled: {self.output_file}")

    async def process_network_latency_sample(
        self, sample: NetworkLatencySample
    ) -> None:
        """Write a single probe sample to the JSONL artifact."""
        await self.buffered_write(sample)

    async def summarize(self) -> list[MetricResult]:
        """Summarize result. Not used for this processor."""
        return []
