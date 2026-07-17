# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate counter for runtime context-overflow detections.

Companion to ``aiperf.common.scenario.is_context_overflow_response`` and the
``RequestRecord.context_overflow`` flag set by the inference result parser.
Increments by 1 per record whose request was tagged as a context-overflow
error; otherwise contributes 0. Used by the InferenceX AgentX scenario
(RFC §7) to flip ``submission_valid=false`` when the overflow rate exceeds the
configured threshold.
"""

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class ContextOverflowCountMetric(BaseAggregateCounterMetric[int]):
    """Counts records flagged as context-overflow by the runtime classifier.

    Formula:
        ```
        Context Overflow Count = Sum(1 if request.context_overflow else 0)
        ```

    Marked ``ERROR_ONLY`` because context-overflow records are by definition
    error responses, and ``NO_INDIVIDUAL_RECORDS`` because the count is an
    aggregate-only signal that doesn't make sense on a per-record export.
    """

    tag = "context_overflow_count"
    header = "Context Overflow Count"
    short_header = "Ctx Overflow"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    flags = MetricFlags.ERROR_ONLY | MetricFlags.NO_INDIVIDUAL_RECORDS
    required_metrics = None

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        """Return 1 iff the underlying RequestRecord was flagged as overflow."""
        return 1 if getattr(record.request, "context_overflow", False) else 0
