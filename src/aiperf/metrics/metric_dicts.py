# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    MetricDictValueTypeT,
    MetricType,
    MetricUnitT,
    MetricValueTypeT,
    MetricValueTypeVarT,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import MetricTypeError, MetricUnitError, NoMetricValue
from aiperf.common.growable_array import GrowableArray
from aiperf.common.models.record_models import MetricResult, MetricValue
from aiperf.common.types import MetricTagT

if TYPE_CHECKING:
    from aiperf.metrics.base_metric import BaseMetric
    from aiperf.metrics.metric_registry import MetricRegistry


_PERCENTILE_QS = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)


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

    Sorts ``clean`` in-place (safe — callers always pass a fresh copy from fancy indexing).
    Extracts min/max from sorted endpoints, avg from arr_sum / n, std from np.std.
    Vectorized linear interpolation for 9 percentiles.

    Args:
        ddof: Delta degrees of freedom for std. 0 = population (inference metrics),
              1 = sample with Bessel's correction (telemetry time-series).
    """
    n = len(clean)
    clean.sort()

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
        min=clean[0],
        max=clean[-1],
        avg=arr_sum / n,
        sum=arr_sum,
        std=std,
        p1=pcts[0],
        p5=pcts[1],
        p10=pcts[2],
        p25=pcts[3],
        p50=pcts[4],
        p75=pcts[5],
        p90=pcts[6],
        p95=pcts[7],
        p99=pcts[8],
        count=n,
    )


@runtime_checkable
class MetricAggregator(Protocol):
    """Run-level aggregator that produces a :class:`MetricResult`.

    Implemented by :class:`MetricArray` (exact, ``np.ndarray``-backed),
    :class:`aiperf.metrics.list_metric_aggregation.TDigestListMetricAggregator`
    (bounded-memory t-digest sketch), and :class:`ScalarSumAggregator`
    (precomputed sum/count over columnar storage). All maintain an exact
    running ``sum`` so derived-sum metrics work uniformly across them.
    """

    @property
    def sum(self) -> int | float: ...

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult: ...


class ScalarSumAggregator:
    """:class:`MetricAggregator` over a precomputed sum and count.

    Used by :class:`aiperf.metrics.accumulator.MetricsAccumulator`, where the
    per-record values already live in columnar storage: derived-sum metrics
    (e.g. ``total_osl`` feeding ``output_token_throughput``) only need the
    exact ``sum``, so copying the column into a :class:`MetricArray` would
    allocate for nothing. Created once per tag per results pass, never per
    record.

    Example:
        >>> agg = ScalarSumAggregator(total=4096.0, count=8)
        >>> agg.sum
        4096.0
    """

    __slots__ = ("_sum", "_count")

    def __init__(self, total: int | float, count: int) -> None:
        self._sum = total
        self._count = count

    @property
    def sum(self) -> int | float:
        """Exact sum of the per-record values this aggregator stands in for."""
        return self._sum

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Build a sum/avg/count-only result (distribution stats live in the
        columnar path — see ``metric_result_from_array``)."""
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            sum=self._sum,
            avg=self._sum / self._count if self._count else None,
            count=self._count,
        )


MetricDictValueTypeVarT = TypeVar(
    "MetricDictValueTypeVarT", bound="MetricValueTypeT | MetricDictValueTypeT"
)

_logger = AIPerfLogger(__name__)


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
    - All `BaseRecordMetric`s as a MetricArray of their values.
    - The most recent value of each `BaseAggregateMetric`.
    - The value of any `BaseDerivedMetric` that has already been computed.

    Optional ``window_start_ns`` / ``window_end_ns`` attributes carry the
    steady-state / timeslice window bounds when this dict represents a
    sub-range of the full run. They drive :meth:`observation_duration` so
    derived metrics divide by the windowed elapsed time instead of the full
    benchmark duration.
    """

    window_start_ns: int | None = None
    window_end_ns: int | None = None

    def observation_duration(self, target_unit: MetricUnitT) -> float:
        """Return the observation duration converted to ``target_unit``.

        If explicit window bounds are set, uses
        ``window_end_ns - window_start_ns``; otherwise falls back to
        :class:`BenchmarkDurationMetric`. Raises :class:`NoMetricValue` when
        the resulting duration is zero.
        """
        from aiperf.common.enums import MetricTimeUnit
        from aiperf.common.exceptions import NoMetricValue
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
            # Record metrics are a MetricArray of values, so we can't convert them directly.
            raise ValueError(
                f"Cannot convert a record metric to a different unit: {metric.tag}"
            )
        return super().get_converted_or_raise(metric, other_unit)


class MetricArray(Generic[MetricValueTypeVarT]):
    """NumPy backed array for metric data.

    This is used to store the values of a metric over time.
    Uses GrowableArray internally for efficient storage with automatic growth.
    """

    def __init__(
        self, initial_capacity: int = Environment.METRICS.ARRAY_INITIAL_CAPACITY
    ):
        """Initialize the array with the given initial capacity."""
        self._array = GrowableArray(
            initial_capacity=initial_capacity,
            dtype=np.float64,
            track_sum=True,
        )

    def extend(self, values: list[MetricValueTypeVarT]) -> None:
        """Extend the array with a list of values."""
        self._array.extend(np.asarray(values, dtype=np.float64))

    def append(self, value: MetricValueTypeVarT) -> None:
        """Append a value to the array."""
        self._array.append(value)

    @property
    def sum(self) -> MetricValueTypeVarT:
        """Get the sum of the array."""
        return self._array.sum  # type: ignore

    @property
    def data(self) -> np.ndarray:
        """Return view of actual data."""
        return self._array.data

    @property
    def capacity(self) -> int:
        """Return current capacity."""
        return self._array.capacity

    def __len__(self) -> int:
        """Return number of elements."""
        return len(self._array)

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Compute metric stats with zero-copy"""

        arr = self.data
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
            arr, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            avg=float(np.mean(arr)),
            sum=float(self.sum),
            std=float(np.std(arr)),
            p1=float(p1),
            p5=float(p5),
            p10=float(p10),
            p25=float(p25),
            p50=float(p50),
            p75=float(p75),
            p90=float(p90),
            p95=float(p95),
            p99=float(p99),
            count=len(self._array),
        )
