# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

from pydantic import ConfigDict

from aiperf.common.constants import STAT_KEYS
from aiperf.common.enums import MetricValueTypeT
from aiperf.common.models.branch_stats import BranchStats
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.common.types import MetricTagT


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
        """Convert the metric result to a JsonMetricResult.

        `count` is omitted for non-RECORD metrics (derived/aggregate scalars),
        where it would trivially be 1 and risks being misread as the request
        count. Tags from other registries (e.g. GPU telemetry) are not in
        MetricRegistry; those keep `count` as-is. Future MetricType members
        also keep `count` by default — opt them in here explicitly.
        """
        from aiperf.common.enums import MetricType
        from aiperf.metrics.metric_registry import MetricRegistry

        metric_class = MetricRegistry.get_class_or_none(self.tag)
        is_scalar = metric_class is not None and metric_class.type in {
            MetricType.AGGREGATE,
            MetricType.DERIVED,
        }

        result = JsonMetricResult(
            unit=self.unit,
            count=None if is_scalar else self.count,
        )
        for stat in STAT_KEYS:
            setattr(result, stat, getattr(self, stat, None))
        return result


@dataclass(frozen=True, slots=True)
class MetricValue:
    """The value of a metric converted to display units for export."""

    value: MetricValueTypeT
    """The numeric metric value in display units."""

    unit: str
    """The display unit label (e.g. 'ms', 'tokens/s')."""


class TimesliceWindow(NamedTuple):
    """Per-timeslice temporal boundaries.

    ``is_complete`` is ``None`` for fully-closed windows (space-efficient
    default matching ``BaseTimeslice``) and ``False`` for the trailing
    partial window when the benchmark stopped before the next slice
    boundary.
    """

    start_ns: int
    end_ns: int
    is_complete: bool | None = None


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
    timeslice_metric_results: list[dict[MetricTagT, MetricResult]] | None = None
    timeslice_windows: list[TimesliceWindow] | None = None
    multi_turn_ttft_trend: dict[int, MetricResult] | None = None
    """Per-``turn_index`` TTFT distribution across all conversations.
    Surfaces KV-cache effectiveness — turn N+1 should drop below turn N.
    Populated by ``MetricsAccumulator.summarize()`` only when records have
    ``turn_index`` metadata; ``None`` for single-turn workloads."""
    total_expected: int | None = None
    was_cancelled: bool = False
    error_summary: list[ErrorDetailsCount] = field(default_factory=list)
    branch_stats: BranchStats | None = None
    """DAG branch-orchestrator counters (children spawned/completed/errored,
    parents suspended/resumed, joins suppressed, children truncated). None on
    non-DAG runs."""

    def get(self, tag: MetricTagT) -> MetricResult | None:
        """Get a metric result by tag, if it exists."""
        for record in self.records or []:
            if record.tag == tag:
                return record
        return None

    def model_dump_json(self) -> str:
        """Pydantic-compat JSON serializer (msgspec ``json.encode`` under the hood).

        Pydantic-model fields (``branch_stats: BranchStats``) are bridged via a
        local ``enc_hook`` so msgspec can serialize them without a registered
        Pydantic encoder.
        """
        import msgspec

        from aiperf.common.models.base_models import _msgspec_enc_hook
        from aiperf.common.models.branch_stats import BranchStats

        def _enc(obj: Any) -> Any:
            if isinstance(obj, BranchStats):
                return obj.model_dump()
            return _msgspec_enc_hook(obj)

        return msgspec.json.encode(self, enc_hook=_enc).decode("utf-8")

    @classmethod
    def model_validate_json(cls, value: str | bytes) -> ProfileResults:
        """Pydantic-compat constructor from a JSON string / bytes.

        Bridges nested Pydantic ``BranchStats`` via a ``dec_hook``.
        """
        import msgspec

        from aiperf.common.models.branch_stats import BranchStats

        def _dec(typ: type, obj: Any) -> Any:
            if typ is BranchStats and isinstance(obj, dict):
                return BranchStats(**obj)
            raise NotImplementedError

        return msgspec.json.decode(value, type=cls, dec_hook=_dec)


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

    def model_dump_json(self) -> str:
        """Pydantic-compat JSON serializer (msgspec ``json.encode`` under the hood).

        Mirrors ``ProfileResults.model_dump_json``: nested Pydantic
        ``BranchStats`` (inside ``results``) is bridged via a local ``enc_hook``
        so the producer encode path stays symmetric with
        ``model_validate_json``'s ``dec_hook``.
        """
        import msgspec

        from aiperf.common.models.base_models import _msgspec_enc_hook
        from aiperf.common.models.branch_stats import BranchStats

        def _enc(obj: Any) -> Any:
            if isinstance(obj, BranchStats):
                return obj.model_dump()
            return _msgspec_enc_hook(obj)

        return msgspec.json.encode(self, enc_hook=_enc).decode("utf-8")

    @classmethod
    def model_validate_json(cls, value: str | bytes) -> ProcessRecordsResult:
        """Pydantic-compat constructor from a JSON string / bytes.

        Mirrors ``ProfileResults.model_validate_json``: this is a slotted
        dataclass, not a Pydantic model, so the stdlib ``model_validate`` does
        not exist. Bridges the nested Pydantic ``BranchStats`` (inside
        ``results``) via a ``dec_hook``.
        """
        import msgspec

        from aiperf.common.models.branch_stats import BranchStats

        def _dec(typ: type, obj: Any) -> Any:
            if typ is BranchStats and isinstance(obj, dict):
                return BranchStats(**obj)
            raise NotImplementedError

        return msgspec.json.decode(value, type=cls, dec_hook=_dec)
