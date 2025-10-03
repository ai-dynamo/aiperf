# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import MetricTypeError, NoMetricValue
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
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    flags = MetricFlags.GOODPUT | MetricFlags.NO_CONSOLE
    required_metrics: set[str] | None = None

    _thresholds: ClassVar[dict[str, float]] = {}

    @classmethod
    def set_slos(cls, slos: dict[str, float] | None) -> None:
        """
        Configure SLO thresholds and update dependencies.
        """
        slos = slos or {}
        if not slos:
            cls._thresholds = {}
            cls.required_metrics = None
            return

        normalized_slos: dict[str, float] = {}

        for metric_tag, value in slos.items():
            try:
                metric_cls = MetricRegistry.get_class(metric_tag)
            except MetricTypeError as e:
                raise ValueError(
                    f"Unknown metric tag(s) in --goodput: {metric_tag}."
                ) from e
            unit = getattr(metric_cls, "unit", None)
            display_unit = getattr(metric_cls, "display_unit", None)
            if display_unit is not None and unit is not None and display_unit != unit:
                try:
                    value = display_unit.convert_to(unit, float(value))
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert from {display_unit} to "
                        f"{unit} for '{metric_tag}': {e}"
                    ) from e
            normalized_slos[metric_tag] = value

        cls._thresholds = normalized_slos
        cls.required_metrics = set(cls._thresholds) if cls._thresholds else None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Returns 1 if the record meets all SLOs; otherwise 0.
        If no SLOs were configured, returns 0.
        """
        if not self._thresholds:
            return 0
        return 1 if self._is_good(record_metrics) else 0

    def _passes(self, metric_cls, record_value: float, threshold_value: float) -> bool:
        """Compare a record value against its SLO using the metric's directionality."""
        if metric_cls.flags.has_flags(MetricFlags.LARGER_IS_BETTER):
            return record_value >= threshold_value
        return record_value <= threshold_value

    def _is_good(self, record_metrics: MetricRecordDict) -> bool:
        """Check if the record satisfies all configured SLOs."""
        for metric_tag, threshold in self._thresholds.items():
            metric_cls = MetricRegistry.get_class(metric_tag)

            target_unit = metric_cls.unit
            try:
                value = record_metrics.get_converted_or_raise(metric_cls, target_unit)
            except NoMetricValue:
                return False

            if not self._passes(metric_cls, value, float(threshold)):
                return False

        return True
