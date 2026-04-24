# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import ConfigDict

from aiperf.common.constants import STAT_KEYS
from aiperf.common.enums import MetricValueTypeT
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.common.types import MetricTagT, TimeSliceT


@dataclass(slots=True, kw_only=True)
class MetricResult:
    """The result values of a single metric.

    Slotted dataclass — shared type for msgspec envelopes
    (``RealtimeMetricsMessage.metrics``, ``ProfileResults.records``) and
    Pydantic (``ProfileResults`` under ``BenchmarkResultsResponse``) via
    ``__pydantic_config__``.

    Carries every JsonMetricResult percentile/stat directly — historically
    inherited, but a msgspec.Struct cannot subclass Pydantic BaseModel, so the
    fields are duplicated here (see ``to_json_result`` for the conversion
    back to the Pydantic JSON-export shape).
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    tag: MetricTagT
    header: str
    unit: str
    count: int | None = None
    # The most recent value of the metric (realtime dashboard display only).
    current: float | int | None = None
    sum: int | float | None = None
    avg: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    std: float | None = None

    def __post_init__(self) -> None:
        # Callers sometimes pass a BaseMetricUnit enum (str-backed but with a
        # custom __hash__) where a plain str is expected. Pydantic used to
        # coerce this on validation; msgspec does not. Collapse to str so
        # downstream set/dict comparisons keyed on the unit continue to work.
        if type(self.unit) is not str and isinstance(self.unit, str):
            self.unit = str.__str__(self.unit)

    def to_display_unit(self) -> MetricResult:
        """Convert the metric result to its display unit."""
        from aiperf.metrics.display_units import to_display_unit
        from aiperf.metrics.metric_registry import MetricRegistry

        return to_display_unit(self, MetricRegistry)

    def to_json_result(self) -> JsonMetricResult:
        """Convert the metric result to a JsonMetricResult."""
        result = JsonMetricResult(unit=self.unit)
        for stat in [
            s for s in STAT_KEYS if s != "sum"
        ]:  # sum is not included in the JsonMetricResult
            setattr(result, stat, getattr(self, stat, None))
        return result


@dataclass(frozen=True, slots=True)
class MetricValue:
    """The value of a metric converted to display units for export."""

    value: MetricValueTypeT
    """The numeric metric value in display units."""

    unit: str
    """The display unit label (e.g. 'ms', 'tokens/s')."""


@dataclass(slots=True, kw_only=True)
class ProfileResults:
    """The results of a profile run.

    Slotted dataclass — shared type for msgspec
    (``ProfileResultsMessage.profile_results``, the /api/results HTTP
    payload encoded via msgspec) and Pydantic
    (``BenchmarkResultsResponse.results`` via ``ProcessRecordsResult``).

    Every field including ``was_cancelled=False`` and ``error_summary=[]``
    is serialized on the wire (historical ``omit_defaults=False`` semantics)
    because downstream consumers expect them — dataclasses always emit
    every field.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    completed: int
    start_ns: int
    end_ns: int
    records: list[MetricResult] | None = None
    timeslice_metric_results: dict[TimeSliceT, list[MetricResult]] | None = None
    total_expected: int | None = None
    was_cancelled: bool = False
    error_summary: list[ErrorDetailsCount] = field(default_factory=list)

    def get(self, tag: MetricTagT) -> MetricResult | None:
        """Get a metric result by tag, if it exists."""
        for record in self.records or []:
            if record.tag == tag:
                return record
        return None


@dataclass(slots=True, kw_only=True)
class ProcessRecordsResult:
    """Result of the process records command.

    Slotted dataclass — shared natively between msgspec
    (``ProcessRecordsResultMessage.results``) and Pydantic
    (``BenchmarkResultsResponse.results``).
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    results: ProfileResults
    errors: list[ErrorDetails] = field(default_factory=list)

    def get(self, tag: MetricTagT) -> MetricResult | None:
        """Get a metric result by tag, if it exists."""
        return self.results.get(tag)
