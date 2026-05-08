# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.post_processors.strategies.core import (
    OTelResultData,
    OTelResultsStrategyProtocol,
    OTelStrategyContextProtocol,
)
from aiperf.post_processors.strategies.genai_semconv import (
    convert_metric_value,
    translate,
)


class MetricResultsStrategy(OTelResultsStrategyProtocol):
    """Streams per-request metric records as histogram observations."""

    def __init__(self, context: OTelStrategyContextProtocol) -> None:
        self._context = context

    def supports(self, record_data: OTelResultData) -> bool:
        return isinstance(record_data, MetricRecordsData)

    async def process(self, record_data: OTelResultData) -> None:
        if not isinstance(record_data, MetricRecordsData):
            return

        aiperf_attributes = self._context.build_record_attributes(record_data)
        for metric_name, metric_value in record_data.metrics.items():
            numeric_values = self._context.coerce_metric_values(
                metric_name, metric_value
            )
            if not numeric_values:
                continue

            # translate() uses the first value only for metadata lookup (unit,
            # attributes, bucket boundaries). The actual value is irrelevant for
            # determining the emission struct — all list values share the same spec name.
            emission = translate(
                metric_name,
                numeric_values[0],
                record_data,
                user_config=self._context.user_config,
            )

            if emission is not None:
                instrument = await self._context.get_or_create_histogram(
                    emission.spec_metric_name,
                    unit=emission.unit,
                    description=f"GenAI semconv metric: {emission.spec_metric_name}",
                    explicit_bucket_boundaries=emission.explicit_bucket_boundaries,
                )
                merged_attrs = {**aiperf_attributes, **dict(emission.attributes)}
                for value in numeric_values:
                    converted = convert_metric_value(metric_name, value)
                    instrument.record(converted, merged_attrs)
            else:
                instrument = await self._context.get_or_create_histogram(metric_name)
                for value in numeric_values:
                    instrument.record(value, aiperf_attributes)
