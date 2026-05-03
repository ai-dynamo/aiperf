# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_aggregate_metric import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class AccuracyCorrectSumMetric(BaseAggregateMetric[float]):
    """Running sum of per-record accuracy.correct values (1.0 correct, 0.0 incorrect).

    AccuracyRecordProcessor writes this tag to MetricRecordDict for every record.
    Registered here so MetricResultsProcessor can aggregate it without warnings.
    AccuracyResultsProcessor and AccuracyConsoleExporter own display; this metric
    is marked NO_CONSOLE | INTERNAL so it does not appear in the standard table.
    """

    tag = "accuracy.correct"
    header = "Accuracy Correct"
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.NO_CONSOLE | MetricFlags.INTERNAL
    required_metrics = None

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        value = record_metrics.get("accuracy.correct")
        if value is None:
            raise NoMetricValue("accuracy.correct not in record_metrics")
        return float(value)

    def _aggregate_value(self, value: float) -> None:
        self._value += value


class AccuracyUnparsedSumMetric(BaseAggregateMetric[float]):
    """Running sum of per-record accuracy.unparsed values (1.0 unparsed, 0.0 conforming).

    AccuracyRecordProcessor writes this tag when the model output required the
    regex fallback (e.g. 'The answer is B.' instead of 'B').
    Marked NO_CONSOLE | INTERNAL so it does not appear in the standard table.
    """

    tag = "accuracy.unparsed"
    header = "Accuracy Unparsed"
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.NO_CONSOLE | MetricFlags.INTERNAL
    required_metrics = None

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        value = record_metrics.get("accuracy.unparsed")
        if value is None:
            raise NoMetricValue("accuracy.unparsed not in record_metrics")
        return float(value)

    def _aggregate_value(self, value: float) -> None:
        self._value += value
