# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from aiperf.common.accumulator_protocols import MetricSeriesProtocol
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    MetricDictValueTypeT,
    MetricTimeUnit,
    MetricType,
    MetricUnitT,
    MetricValueTypeT,
)
from aiperf.common.exceptions import MetricTypeError, MetricUnitError, NoMetricValue
from aiperf.common.models.record_models import MetricResult, MetricValue
from aiperf.common.types import MetricTagT

if TYPE_CHECKING:
    from aiperf.metrics.base_metric import BaseMetric
    from aiperf.metrics.metric_registry import MetricRegistry


__all__ = [
    "BaseMetricDict",
    "MetricAggregator",
    "MetricDictValueTypeVarT",
    "MetricRecordDict",
    "MetricResultsDict",
    "MetricSeriesProtocol",
    "metric_result_from_array",
]

# Back-compat alias: ``MetricAggregator`` is the historical name for
# :class:`MetricSeriesProtocol`. Both point at the same Protocol; the alias
# keeps existing imports working.
MetricAggregator = MetricSeriesProtocol


MetricDictValueTypeVarT = TypeVar(
    "MetricDictValueTypeVarT", bound="MetricValueTypeT | MetricDictValueTypeT"
)

# Standard percentile band shared by ``metric_result_from_array`` and the
# derived-latency builders. Kept here (vs inlined) so the byte-for-byte
# layout of the resulting MetricResult matches across all callers.
_PERCENTILE_QS = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)

_logger = AIPerfLogger(__name__)


def metric_result_from_array(
    tag: MetricTagT,
    header: str,
    unit: str,
    clean: NDArray[np.float64],
    arr_sum: float,
    *,
    ddof: int = 0,
) -> MetricResult:
    """Compute MetricResult directly from a clean (no-NaN) numpy array.

    Sorts ``clean`` in-place (safe — callers always pass a fresh copy from
    fancy indexing). Extracts min/max from sorted endpoints, avg from
    arr_sum / n, std from np.std. Vectorized linear interpolation for 9
    percentiles.

    Args:
        ddof: Delta degrees of freedom for std. 0 = population (inference
              metrics), 1 = sample with Bessel's correction (telemetry
              time-series).
    """
    n = len(clean)
    clean.sort()  # in-place sort

    virtual_idx = _PERCENTILE_QS / 100.0 * (n - 1)
    lo = virtual_idx.astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = virtual_idx - lo
    pcts = clean[lo] + frac * (clean[hi] - clean[lo])

    std = float(np.std(clean, ddof=ddof)) if n > ddof else 0.0

    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        min=float(clean[0]),
        max=float(clean[-1]),
        avg=arr_sum / n,
        sum=arr_sum,
        std=std,
        p1=float(pcts[0]),
        p5=float(pcts[1]),
        p10=float(pcts[2]),
        p25=float(pcts[3]),
        p50=float(pcts[4]),
        p75=float(pcts[5]),
        p90=float(pcts[6]),
        p95=float(pcts[7]),
        p99=float(pcts[8]),
        count=n,
    )


class BaseMetricDict(
    Generic[MetricDictValueTypeVarT], dict[MetricTagT, MetricDictValueTypeVarT]
):
    """Base class for all metric dicts."""

    def get_or_raise(self, metric: type["BaseMetric"]) -> MetricDictValueTypeT:
        """Get the value of a metric, or raise NoMetricValue if it is not available."""
        value = self.get(metric.tag)
        if value is None:
            raise NoMetricValue(f"Metric {metric.tag} is not available for the record.")
        return value

    def get_converted_or_raise(
        self, metric: type["BaseMetric"], other_unit: MetricUnitT
    ) -> float:
        """Get the value of a metric, but converted to a different unit, or raise NoMetricValue if it is not available."""
        return metric.unit.convert_to(other_unit, self.get_or_raise(metric))  # type: ignore


class MetricRecordDict(BaseMetricDict[MetricValueTypeT]):
    """
    A dict of metrics for a single record. This is used to store the current values
    of all metrics that have been computed for a single record.

    This will include:
    - The current value of any `BaseRecordMetric` that has been computed for this record.
    - The new value of any `BaseAggregateMetric` that has been computed for this record.
    - No `BaseDerivedMetric`s will be included.
    """

    def to_display_dict(
        self,
        registry: "type[MetricRegistry]",
        show_internal: bool = False,
        show_experimental: bool = False,
    ) -> dict[str, MetricValue]:
        """Convert to display units with filtering applied.
        NOTE: This will not include metrics with the `NO_INDIVIDUAL_RECORDS` flag.

        Args:
            registry: MetricRegistry class for looking up metric definitions
            show_internal: If True, include experimental/internal metrics

        Returns:
            Dictionary of {tag: MetricValue} for export
        """
        from aiperf.common.enums import MetricFlags

        result = {}
        for tag, value in self.items():
            try:
                metric_class = registry.get_class(tag)
            except MetricTypeError:
                _logger.warning(f"Metric {tag} not found in registry")
                continue

            if (
                metric_class.has_flags(MetricFlags.EXPERIMENTAL)
                and not show_experimental
            ):
                continue
            if metric_class.has_flags(MetricFlags.INTERNAL) and not show_internal:
                continue
            if metric_class.has_flags(MetricFlags.NO_INDIVIDUAL_RECORDS):
                continue

            display_unit = metric_class.display_unit or metric_class.unit
            if display_unit != metric_class.unit:
                try:
                    if isinstance(value, list):
                        value = [
                            metric_class.unit.convert_to(display_unit, v) for v in value
                        ]
                    else:
                        value = metric_class.unit.convert_to(display_unit, value)
                except MetricUnitError as e:
                    _logger.warning(
                        f"Error converting {tag} from {metric_class.unit} to {display_unit}: {e}"
                    )

            result[tag] = MetricValue(
                value=value,
                unit=str(display_unit),
            )

        return result


class MetricResultsDict(BaseMetricDict[MetricDictValueTypeT]):
    """
    A dict of metrics over an entire run. This is used to store the final values
    of all metrics that have been computed for an entire run.

    This will include:
    - All ``BaseRecordMetric`` values as a run-level metric series implementing
      ``MetricSeriesProtocol`` (numpy column, ragged CSR, growable array, etc.).
    - The most recent value of each ``BaseAggregateMetric``.
    - The value of any ``BaseDerivedMetric`` that has already been computed.
    """

    def __init__(self, *args: ..., **kwargs: ...) -> None:
        super().__init__(*args, **kwargs)
        self.window_start_ns: int | None = None
        """Inclusive start of the analysis window (ns since epoch); ``None``
        means the dict spans the full run."""
        self.window_end_ns: int | None = None
        """Exclusive end of the analysis window (ns since epoch); ``None``
        means the dict spans the full run."""

    def observation_duration(self, target_unit: MetricUnitT) -> float:
        """Return the observation duration converted to *target_unit*.

        If explicit window bounds are set (``window_start_ns`` / ``window_end_ns``
        — populated for timeslice and analyzer-windowed scalar dicts), uses
        ``(window_end_ns - window_start_ns)``. Otherwise falls back to
        ``BenchmarkDurationMetric``.

        Raises ``NoMetricValue`` when the resolved duration is zero.
        """
        from aiperf.metrics.types.benchmark_duration_metric import (
            BenchmarkDurationMetric,
        )

        if self.window_start_ns is not None and self.window_end_ns is not None:
            duration_ns = self.window_end_ns - self.window_start_ns
            duration = MetricTimeUnit.NANOSECONDS.convert_to(target_unit, duration_ns)
        else:
            duration = self.get_converted_or_raise(BenchmarkDurationMetric, target_unit)
        if duration == 0:
            raise NoMetricValue("Observation duration is zero")
        return duration

    def get_converted_or_raise(
        self, metric: type["BaseMetric"], other_unit: MetricUnitT
    ) -> float:
        """Get the value of a metric, but converted to a different unit, or raise NoMetricValue if it is not available."""
        if metric.type == MetricType.RECORD:
            # Record metrics are a run-level metric series, so we can't convert them directly.
            raise ValueError(
                f"Cannot convert a record metric to a different unit: {metric.tag}"
            )
        return super().get_converted_or_raise(metric, other_unit)
