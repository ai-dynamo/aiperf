# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import ServerMetricsFormat
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import (
    ServerMetricsRecord,
    SlimRecord,
)
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class ServerMetricsJSONLWriter(
    BaseMetricsProcessor,
    BufferedJSONLWriterMixin[SlimRecord],
):
    """Exports per-record server metrics data to JSONL files in slim format.

    This processor converts full ServerMetricsRecord objects to slim format before writing,
    excluding static metadata (metric types, description text) to minimize file size.
    Writes one JSON line per collection cycle.

    Each line contains:
        - timestamp_ns: Collection timestamp in nanoseconds
        - endpoint_latency_ns: Time taken to collect the metrics from the endpoint
        - endpoint_url: Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')
        - metrics: Dict mapping metric names to sample lists (flat structure)
    """

    def __init__(
        self,
        run: BenchmarkRun,
        **kwargs,
    ) -> None:
        if not run.cfg.server_metrics.enabled:
            raise PostProcessorDisabled(
                "Server metrics JSONL export is disabled via --no-server-metrics"
            )

        # Check if JSONL format is enabled
        if ServerMetricsFormat.JSONL not in run.cfg.server_metrics.formats:
            raise PostProcessorDisabled(
                "Server metrics JSONL export disabled: format not selected"
            )

        output_file = run.cfg.artifacts.server_metrics_export_jsonl_file

        super().__init__(
            run=run,
            output_file=output_file,
            batch_size=Environment.SERVER_METRICS.EXPORT_BATCH_SIZE,
            **kwargs,
        )

        self.info(f"Server metrics JSONL export enabled: {self.output_file}")

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record by converting to slim and writing to JSONL.

        Converts full record to slim format to reduce file size by excluding static metadata.
        Skips duplicate records to avoid cluttering the JSONL file.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics snapshot and metadata
        """
        # Skip duplicate records - they're already filtered in time series aggregation
        if record.is_duplicate:
            return

        # Convert to slim format before writing to reduce file size
        slim_record = record.to_slim()
        await self.buffered_write(slim_record)

    async def summarize(self) -> list[MetricResult]:
        """Summarize result. Not used for this processor"""
        return []
