# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from aiperf.common.enums.metric_enums import MetricValueTypeVarT

if TYPE_CHECKING:
    from aiperf.common.enums import CreditPhase
    from aiperf.common.models.error_models import ErrorDetailsCount
    from aiperf.common.models.record_models import MetricResult
    from aiperf.common.types import MetricTagT
    from aiperf.exporters.exporter_config import FileExportInfo
    from aiperf.plugin.enums import AccumulatorType


@runtime_checkable
class AccumulatorResult(Protocol):
    """Protocol for typed results from accumulator summarize()."""

    def to_json(self) -> Any:
        """Serialize to JSON-compatible structure."""
        ...

    def to_csv(self) -> list[dict[str, Any]]:
        """Serialize to list of CSV-compatible row dicts."""
        ...


@runtime_checkable
class MetricSeriesProtocol(Protocol[MetricValueTypeVarT]):
    """Shared interface for run-level record metric series consumers.

    Implemented by any in-memory accumulator that exposes a running sum, a
    record count, and a finalized ``MetricResult`` summary. Used by the
    per-tag dispatch path in MetricsAccumulator and by ColumnStore-backed
    series wrappers so that derived metrics can read values without caring
    about the underlying storage shape (numpy column, ragged CSR, growable
    array, etc.).
    """

    @property
    def sum(self) -> MetricValueTypeVarT:
        """Return the accumulated sum of all observed values."""

    def __len__(self) -> int:
        """Return the number of observed values."""

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Summarize the accumulated values as a MetricResult."""


@dataclass(frozen=True, slots=True)
class ExportContext:
    """Context passed to domain-specific export_results() methods.

    Bundles the profiling time window and error summary so that export_results
    signatures stay stable as new fields are added.
    """

    start_ns: int | None = None
    """Inclusive start of the export time window (ns since epoch), or None for unbounded."""

    end_ns: int | None = None
    """Exclusive end of the export time window (ns since epoch), or None for unbounded."""

    phase: CreditPhase | None = None
    """Credit phase represented by this export, or None when phase-agnostic."""

    phase_index: int | None = None
    """Concrete runtime phase index for phase-local exports, when available."""

    phase_name: str | None = None
    """User-provided phase name for diagnostics/export metadata."""

    phase_kind: str | None = None
    """Semantic phase kind for diagnostics/export metadata."""

    is_phase_scoped: bool = False
    """True when exporting an individual concrete phase rather than aggregate results."""

    error_summary: list[ErrorDetailsCount] | None = None
    """De-duplicated profile-run error counts to surface in the export, if any."""

    cancelled: bool = False
    """True when the profile run was cancelled — exporters may emit partial artifacts."""

    warmup_start_ns: int | None = None
    """Inclusive start of the warmup window (ns), for accumulators that export a
    separate warmup summary alongside the profiling one (e.g. server metrics)."""

    warmup_end_ns: int | None = None
    """Exclusive end of the warmup window (ns); see ``warmup_start_ns``."""


@dataclass(slots=True)
class SummaryContext:
    """Typed cross-accumulator communication context for summarize-time analyzers.

    NOT a Pydantic model — never serialized over the wire. Created by
    RecordsManager during summarization: accumulators run first and register
    their instances in ``accumulators`` and their summaries in
    ``accumulator_outputs``; then ``analyzer`` plugins read peer state via
    ``get_accumulator()`` / ``get_output()`` to compute cross-accumulator
    metrics (e.g. energy efficiency joins GPU telemetry to inference tokens).

    Dependencies are currently one level deep (analyzers depend on accumulators,
    not on each other), so a flat two-stage run suffices; a topological sort
    over analyzer-to-analyzer dependencies is the extension point if that changes.
    """

    accumulators: dict[AccumulatorType, Any] = field(default_factory=dict)
    """Live accumulator instances keyed by AccumulatorType — accumulators use this to query peer state."""

    accumulator_outputs: dict[str, Any] = field(default_factory=dict)
    """Already-computed summary payloads keyed by accumulator name — populated as topo-order completes."""

    start_ns: int = 0
    """Inclusive start of the summarization window (ns since epoch); 0 means full range."""

    end_ns: int = 0
    """Exclusive end of the summarization window (ns since epoch); 0 means full range."""

    phase: CreditPhase | None = None
    """Credit phase to scope this summary to (e.g. PROFILING for realtime metrics), or None for phase-agnostic full-range summarization."""

    phase_index: int | None = None
    """Absolute phase index to scope this summary to when ``phase`` is set."""

    cancelled: bool = False
    """True when the profile run was cancelled — analyzers may short-circuit."""

    def get_accumulator(self, accumulator_type: AccumulatorType) -> Any | None:
        """Look up an accumulator by its type. Returns None if not present."""
        return self.accumulators.get(accumulator_type)

    def get_output(self, accumulator_type: str) -> Any | None:
        """Look up a previously-computed accumulator output. Returns None if not yet available."""
        return self.accumulator_outputs.get(accumulator_type)


@runtime_checkable
class AccumulatorProtocol(Protocol):
    """Protocol for accumulators that ingest records, support time-range queries, and produce summaries.

    Accumulators are the primary data stores in the records pipeline. Each accumulator
    owns exactly one record type and is fully self-contained — no cross-accumulator
    dependencies.
    """

    async def process_record(self, record: Any) -> None:
        """Ingest a single record into this accumulator's internal storage."""
        ...

    def query_time_range(self, start_ns: int, end_ns: int) -> NDArray[np.bool_]:
        """Return a boolean mask where True marks records in [start_ns, end_ns).

        The mask length equals the accumulator's record count. Callers can use
        ``mask.sum()`` for the count or ``np.where(mask)[0]`` for indices.
        """
        ...

    async def summarize(self, ctx: SummaryContext | None = None) -> AccumulatorResult:
        """Compute and return aggregated metric results.

        Args:
            ctx: Optional SummaryContext for reading dependency outputs.
                 None when called for realtime metrics (no cross-processor deps).
        """
        ...

    async def export_results(self, ctx: ExportContext) -> Any:
        """Export final results for this accumulator.

        Called once after profiling completes. Each accumulator returns its own
        typed result (AccumulatorMetricsSummary, TelemetryExportData, ServerMetricsResults)
        which is consumed by typed fields on the unified results message.

        Args:
            ctx: ExportContext with profiling time window, error summary, and cancelled flag.
        """
        ...


@runtime_checkable
class AnalyzerProtocol(Protocol):
    """Protocol for summarize-time analyzers that join across accumulators.

    Analyzers run once after every accumulator has summarized. Unlike an
    accumulator, an analyzer stores no records — it reads peer accumulator state
    from the ``SummaryContext`` (via ``get_accumulator()`` / ``get_output()``)
    and returns derived ``MetricResult`` rows that are merged into the profiling
    summary. The first analyzer is energy efficiency, which joins GPU-telemetry
    energy to inference token totals.

    An analyzer declares its ``required_accumulators`` in plugin metadata; the
    RecordsManager skips it when any declared accumulator is absent (e.g. energy
    efficiency is skipped when GPU telemetry is disabled).
    """

    async def analyze(self, ctx: SummaryContext) -> list[MetricResult]:
        """Compute cross-accumulator metrics from ``ctx`` and return them."""
        ...


@runtime_checkable
class StreamExporterProtocol(Protocol):
    """Protocol for processors that stream each record to an external sink (e.g. JSONL files).

    Stream exporters have no summarization dependencies and are flushed after
    all accumulators complete.
    """

    async def process_record(self, record: Any) -> None:
        """Write a single record to the export sink."""
        ...

    async def finalize(self) -> None:
        """Flush any buffered data. Called once after all records are processed."""
        ...

    def get_export_info(self) -> FileExportInfo:
        """Return metadata about the file this exporter writes to."""
        ...
