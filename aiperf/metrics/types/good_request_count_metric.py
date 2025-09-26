# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.enums import GenericMetricUnit, MetricFlags, MetricTimeUnit
from aiperf.common.enums.metric_enums import MetricTimeUnitInfo, MetricUnitT
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry


class GoodRequestCountMetric(BaseAggregateCounterMetric):
    """
    Counts requests that satisfy all user-provided SLO thresholds.
    """

    tag = "good_request_count"
    header = "GoodRequestCount"
    short_header = "GoodRequestCount"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    flags = MetricFlags.GOODPUT | MetricFlags.HIDDEN
    required_metrics: set[str] | None = None

    _thresholds: dict[str, float] = {}

    @classmethod
    def set_slos(cls, slos: dict[str, float] | None) -> None:
        """
        Configure SLO thresholds and update dependencies.
        """
        slos = slos or {}
        for metric_tag in slos:
            MetricRegistry.get_class(metric_tag)

        cls._thresholds = {k: float(v) for k, v in slos.items()}

        cls.required_metrics = set(cls._thresholds.keys()) if cls._thresholds else None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Returns 1 if the record meets all SLOs; otherwise 0.
        If no SLOs were configured, returns 0.
        """
        if not type(self)._thresholds:
            return 0
        return 1 if self._is_good(record, record_metrics) else 0

    def _normalize_threshold(self, metric_cls, raw_threshold: float) -> float:
        """
        Convert the user-provided SLO (which is in ms for time-based metrics) into the metric's unit.
        For non-time metrics, return as-is.
        """
        unit: MetricUnitT = metric_cls.unit
        if isinstance(unit, (MetricTimeUnit | MetricTimeUnitInfo)):
            return MetricTimeUnit.MILLISECONDS.convert_to(unit, raw_threshold)
        return float(raw_threshold)

    def _passes(self, metric_cls, record_value: float, threshold_value: float) -> bool:
        if metric_cls.flags.has_flags(MetricFlags.LARGER_IS_BETTER):
            return record_value >= threshold_value
        return record_value <= threshold_value

    def _is_good(self, record, record_metrics) -> bool:
        for metric_tag, raw_threshold in type(self)._thresholds.items():
            metric_cls = MetricRegistry.get_class(metric_tag)
            unit = metric_cls.unit
            try:
                value = record_metrics.get_converted_or_raise(metric_cls, unit)
            except Exception:
                return False

            threshold = self._normalize_threshold(metric_cls, raw_threshold)

            if not self._passes(metric_cls, value, threshold):
                return False

        return True
