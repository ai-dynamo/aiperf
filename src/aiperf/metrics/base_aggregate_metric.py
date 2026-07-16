# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import ClassVar, Generic

from aiperf.common.enums import AggregationKind, MetricType, MetricValueTypeVarT
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class BaseAggregateMetric(
    Generic[MetricValueTypeVarT], BaseMetric[MetricValueTypeVarT], ABC
):
    """A base class for aggregate metrics.

    Each distributed RecordProcessor emits the per-record value via
    ``_parse_record`` (this record alone; it must not fold across records). The
    MetricsAccumulator stores those values in a numpy column and folds them into
    a single scalar according to the subclass's ``aggregation_kind``
    (SUM/MAX/MIN, default SUM) — no per-instance running state is kept.

    Examples:
    ```python
    class RequestCountMetric(BaseAggregateMetric[int]):
        aggregation_kind = AggregationKind.SUM

        def _parse_record(self, record: ParsedResponseRecord, record_metrics: MetricRecordDict) -> int:
            # One per request; the accumulator sums the column.
            return 1
    ```
    """

    type = MetricType.AGGREGATE
    aggregation_kind: ClassVar[AggregationKind] = AggregationKind.SUM

    def parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Parse the record and return the individual value.

        Raises:
            ValueError: If the metric cannot be computed for the given inputs.
        """
        self._require_valid_record(record)
        self._check_metrics(record_metrics)
        return self._parse_record(record, record_metrics)

    @abstractmethod
    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Parse the record and *return* the individual value based on this record,
        and this record alone. Implemented by subclasses.

        NOTE: Do not fold across records here — the accumulator combines the
        per-record values via ``aggregation_kind``.

        Called after the required metrics and record are validated, so it can
        assume both are available/valid.

        Raises:
            ValueError: If the metric cannot be computed for the given inputs.
        """
        raise NotImplementedError("Subclasses must implement this method")
