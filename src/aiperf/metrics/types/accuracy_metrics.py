# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.accuracy.models import (
    ACCURACY_RECORD_CORRECT_KEY,
    ACCURACY_RECORD_UNPARSED_KEY,
)
from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_aggregate_metric import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class AccuracyCorrectSumMetric(BaseAggregateMetric[float]):
    """Registration for the per-record ``accuracy_correct`` transport key.

    AccuracyRecordProcessor writes ``accuracy_correct`` (1.0 correct / 0.0
    incorrect) into every record's MetricRecordDict to hand the grade to
    AccuracyResultsProcessor. MetricsAccumulator only ingests registered tags
    (it warns and drops anything else), so this exists purely to register it.

    It is a transport key, NOT the accuracy display: ``INTERNAL`` +
    ``console_group=NONE`` keep it out of the console table and file exports.
    The user-facing accuracy summary is produced separately by
    AccuracyResultsProcessor.summarize() under the ``accuracy.`` (dot) namespace,
    which this ``accuracy_`` (underscore) tag deliberately stays out of so the
    two can never collide.
    """

    tag = ACCURACY_RECORD_CORRECT_KEY
    header = "Accuracy Correct"
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.INTERNAL
    console_group = MetricConsoleGroup.NONE
    required_metrics = None

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        value = record_metrics.get(ACCURACY_RECORD_CORRECT_KEY)
        if value is None:
            raise NoMetricValue(f"{ACCURACY_RECORD_CORRECT_KEY} not in record_metrics")
        return float(value)


class AccuracyUnparsedSumMetric(BaseAggregateMetric[float]):
    """Registration for the per-record ``accuracy_unparsed`` transport key.

    Same role as AccuracyCorrectSumMetric: registers the per-record
    ``accuracy_unparsed`` (1.0 when the grader could not cleanly extract the
    answer) transport key so MetricsAccumulator accepts it. Transport only --
    ``INTERNAL`` + ``console_group=NONE``; the displayed unparsed counts come
    from AccuracyResultsProcessor.summarize() in the ``accuracy.`` namespace.
    Registering this under the dot tag ``accuracy.unparsed`` used to collide
    with that summary tag and drop the overall unparsed count.
    """

    tag = ACCURACY_RECORD_UNPARSED_KEY
    header = "Accuracy Unparsed"
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.INTERNAL
    console_group = MetricConsoleGroup.NONE
    required_metrics = None

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        value = record_metrics.get(ACCURACY_RECORD_UNPARSED_KEY)
        if value is None:
            raise NoMetricValue(f"{ACCURACY_RECORD_UNPARSED_KEY} not in record_metrics")
        return float(value)
