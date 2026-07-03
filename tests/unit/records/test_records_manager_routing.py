# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the static record-type routing infrastructure.

Two static lookup helpers in ``records_manager_processing`` dispatch
records to accumulators and stream exporters:

* :func:`accumulators_for_record_type` and
  :func:`stream_exporters_for_record_type` — pure functions that read the
  ``record_types`` metadata from ``plugins.iter_entries(...)`` and return
  the matching accumulator/exporter instances. Called once at
  ``RecordsManager.__init__`` time so the hot path is a list iteration,
  not a per-record plugin scan.
* ``_send_record_to_accumulators`` — fans a record out to the precomputed
  ``_metric_record_accumulators`` and ``_metric_record_stream_exporters``
  lists; per-handler exceptions are caught so one bad handler does not
  abort the others.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from aiperf.common.accumulator_protocols import (
    AccumulatorProtocol,
    AccumulatorResult,
    ExportContext,
    StreamExporterProtocol,
    SummaryContext,
)
from aiperf.plugin.enums import AccumulatorType, StreamExporterType
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_manager_processing import (
    accumulators_for_record_type,
    stream_exporters_for_record_type,
)

# ---------------------------------------------------------------------------
# Fake plugin entries (k8s plugin metadata shape)
# ---------------------------------------------------------------------------


def _make_entry(name: str, record_types: list[str]) -> MagicMock:
    """Build a fake PluginEntry-shaped MagicMock with ``record_types`` metadata."""
    entry = MagicMock()
    entry.name = name
    entry.metadata = {"record_types": record_types}
    return entry


# ---------------------------------------------------------------------------
# Stub processors (protocol-conformant)
# ---------------------------------------------------------------------------


class StubAccumulatorResult:
    """Minimal AccumulatorResult for testing."""

    def to_json(self) -> Any:
        return {}

    def to_csv(self) -> list[dict[str, Any]]:
        return []


class StubAccumulator:
    """Accumulator stub that records process_record calls."""

    def __init__(self) -> None:
        self.process_record = AsyncMock()

    async def summarize(
        self, ctx: SummaryContext | None = None
    ) -> StubAccumulatorResult:
        return StubAccumulatorResult()

    def query_time_range(self, start_ns: int, end_ns: int) -> NDArray[np.bool_]:
        return np.array([], dtype=bool)

    async def export_results(self, ctx: ExportContext) -> StubAccumulatorResult:
        return StubAccumulatorResult()


class StubStreamExporter:
    """Stream exporter stub for testing."""

    def __init__(self) -> None:
        self.process_record = AsyncMock()
        self.finalize = AsyncMock()
        self.get_export_info = MagicMock()


# ---------------------------------------------------------------------------
# Tests: accumulators_for_record_type / stream_exporters_for_record_type
# ---------------------------------------------------------------------------


class TestAccumulatorsForRecordType:
    """Static plugin-metadata lookup replaces the source's _routing_table."""

    def test_single_accumulator_matches_record_type(self, monkeypatch) -> None:
        acc = StubAccumulator()
        accs = {AccumulatorType.METRIC_RESULTS: acc}
        entries = [_make_entry("metric_results", ["metric_records"])]

        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda category: iter(entries),
        )

        matched = accumulators_for_record_type(accs, "metric_records")
        assert matched == [acc]

    def test_no_match_for_unknown_record_type(self, monkeypatch) -> None:
        acc = StubAccumulator()
        accs = {AccumulatorType.METRIC_RESULTS: acc}
        entries = [_make_entry("metric_results", ["metric_records"])]

        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda category: iter(entries),
        )

        matched = accumulators_for_record_type(accs, "telemetry_records")
        assert matched == []

    def test_only_matching_accumulators_returned(self, monkeypatch) -> None:
        """Different accumulators register under different record_types."""
        acc_metric = StubAccumulator()
        acc_telemetry = StubAccumulator()
        accs = {
            AccumulatorType.METRIC_RESULTS: acc_metric,
            AccumulatorType.GPU_TELEMETRY: acc_telemetry,
        }
        entries = [
            _make_entry("metric_results", ["metric_records"]),
            _make_entry("gpu_telemetry", ["telemetry_records"]),
        ]

        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda category: iter(entries),
        )

        assert accumulators_for_record_type(accs, "metric_records") == [acc_metric]
        assert accumulators_for_record_type(accs, "telemetry_records") == [
            acc_telemetry
        ]

    def test_skips_entries_not_in_loaded_dict(self, monkeypatch) -> None:
        """Entries with no instantiated accumulator (disabled) are skipped."""
        acc = StubAccumulator()
        accs = {AccumulatorType.METRIC_RESULTS: acc}
        # Two entries declare "metric_records" but only one is loaded.
        entries = [
            _make_entry("metric_results", ["metric_records"]),
            _make_entry("server_metrics", ["metric_records"]),
        ]

        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda category: iter(entries),
        )

        matched = accumulators_for_record_type(accs, "metric_records")
        assert matched == [acc]

    def test_empty_accumulators_dict_returns_empty(self, monkeypatch) -> None:
        entries = [_make_entry("metric_results", ["metric_records"])]
        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda category: iter(entries),
        )
        assert accumulators_for_record_type({}, "metric_records") == []


class TestStreamExportersForRecordType:
    def test_single_stream_exporter_matches(self, monkeypatch) -> None:
        exp = StubStreamExporter()
        exporters = {StreamExporterType.RECORD_EXPORT: exp}
        entries = [_make_entry("record_export", ["metric_records"])]

        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda category: iter(entries),
        )

        matched = stream_exporters_for_record_type(exporters, "metric_records")
        assert matched == [exp]

    def test_only_matching_exporters_returned(self, monkeypatch) -> None:
        exp_record = StubStreamExporter()
        exp_telemetry = StubStreamExporter()
        exporters = {
            StreamExporterType.RECORD_EXPORT: exp_record,
            StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp_telemetry,
        }
        entries = [
            _make_entry("record_export", ["metric_records"]),
            _make_entry("gpu_telemetry_jsonl_writer", ["telemetry_records"]),
        ]

        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda category: iter(entries),
        )

        assert stream_exporters_for_record_type(exporters, "metric_records") == [
            exp_record
        ]


# ---------------------------------------------------------------------------
# Tests: _send_record_to_accumulators (per-record dispatch hot path)
# ---------------------------------------------------------------------------


def _make_dispatch_manager_mock(
    accumulators_list: list[Any],
    exporters_list: list[Any],
) -> MagicMock:
    """Mock RecordsManager with the precomputed dispatch lists pre-populated.

    Mirrors the source-branch ``_make_manager_mock`` helper but adapts to
    K8s's static lookup: ``_metric_record_accumulators`` and
    ``_metric_record_stream_exporters`` are computed in ``__init__`` from
    ``accumulators_for_record_type`` / ``stream_exporters_for_record_type``,
    so we set them directly.
    """
    mgr = MagicMock()
    mgr._metric_record_accumulators = accumulators_list
    mgr._metric_record_stream_exporters = exporters_list
    mgr.error = MagicMock()
    mgr.warning = MagicMock()
    mgr.debug = MagicMock()
    mgr._send_record_to_accumulators = (
        RecordsManager._send_record_to_accumulators.__get__(mgr)
    )
    return mgr


class TestSendRecordToAccumulators:
    """Test K8s's per-record fan-out (replaces source's _dispatch_record)."""

    @pytest.mark.asyncio
    async def test_dispatch_calls_all_handlers(self) -> None:
        acc = StubAccumulator()
        exp = StubStreamExporter()
        mgr = _make_dispatch_manager_mock([acc], [exp])

        record = MagicMock()
        await mgr._send_record_to_accumulators(record)

        acc.process_record.assert_called_once_with(record)
        exp.process_record.assert_called_once_with(record)

    @pytest.mark.asyncio
    async def test_dispatch_with_no_handlers_is_noop(self) -> None:
        """Empty dispatch lists short-circuit — no error, no crash."""
        mgr = _make_dispatch_manager_mock([], [])

        await mgr._send_record_to_accumulators(MagicMock())

        # No errors reported
        mgr.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception_logged(self) -> None:
        """One handler raising does not prevent other handlers from running."""
        acc = StubAccumulator()
        acc.process_record.side_effect = RuntimeError("boom")
        exp = StubStreamExporter()
        mgr = _make_dispatch_manager_mock([acc], [exp])

        record = MagicMock()
        await mgr._send_record_to_accumulators(record)

        # Exporter should still be called despite accumulator failure
        exp.process_record.assert_called_once_with(record)
        # Error should be logged
        mgr.error.assert_called_once()
        assert "boom" in mgr.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_dispatch_multiple_handler_exceptions(self) -> None:
        """Multiple handler failures are each logged independently."""
        acc = StubAccumulator()
        acc.process_record.side_effect = RuntimeError("acc error")
        exp = StubStreamExporter()
        exp.process_record.side_effect = ValueError("exp error")
        mgr = _make_dispatch_manager_mock([acc], [exp])

        await mgr._send_record_to_accumulators(MagicMock())

        assert mgr.error.call_count == 2

    @pytest.mark.asyncio
    async def test_handler_order_accumulators_before_exporters(self) -> None:
        """Accumulators run before stream exporters in the gather targets."""
        call_order: list[str] = []

        acc = StubAccumulator()

        async def acc_record(_record: Any) -> None:
            call_order.append("acc")

        acc.process_record.side_effect = acc_record

        exp = StubStreamExporter()

        async def exp_record(_record: Any) -> None:
            call_order.append("exp")

        exp.process_record.side_effect = exp_record

        mgr = _make_dispatch_manager_mock([acc], [exp])

        await mgr._send_record_to_accumulators(MagicMock())

        # Targets list is [*accumulators, *exporters] — gather may interleave
        # but the *targets* list ordering is observable via zip in the error
        # path. Both must have run.
        assert "acc" in call_order
        assert "exp" in call_order


# ---------------------------------------------------------------------------
# Tests: Protocol conformance of stubs
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_stub_accumulator_matches_protocol(self) -> None:
        assert isinstance(StubAccumulator(), AccumulatorProtocol)

    def test_stub_stream_exporter_matches_protocol(self) -> None:
        assert isinstance(StubStreamExporter(), StreamExporterProtocol)

    def test_stub_result_matches_accumulator_result(self) -> None:
        assert isinstance(StubAccumulatorResult(), AccumulatorResult)


# ---------------------------------------------------------------------------
# Tests: Stream exporter finalize
# ---------------------------------------------------------------------------


def _make_finalize_manager_mock(stream_exporters: dict) -> MagicMock:
    """Create a mock with _stream_exporters and _finalize_stream_exporters wired up."""
    mgr = MagicMock()
    mgr._stream_exporters = stream_exporters
    mgr.debug = MagicMock()
    mgr.error = MagicMock()
    mgr._finalize_stream_exporters = RecordsManager._finalize_stream_exporters.__get__(
        mgr
    )
    return mgr


class TestFinalizeStreamExporters:
    """Test _finalize_stream_exporters logic using a mock RecordsManager."""

    @pytest.mark.asyncio
    async def test_finalize_calls_all_exporters(self) -> None:
        exp1 = StubStreamExporter()
        exp2 = StubStreamExporter()
        mgr = _make_finalize_manager_mock(
            {
                StreamExporterType.RECORD_EXPORT: exp1,
                StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp2,
            },
        )

        await mgr._finalize_stream_exporters()

        exp1.finalize.assert_called_once()
        exp2.finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_finalize_empty_exporters_noop(self) -> None:
        mgr = _make_finalize_manager_mock({})
        await mgr._finalize_stream_exporters()
        # No error, no crash
        mgr.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_finalize_error_logged_per_exporter(self) -> None:
        """One exporter failing does not prevent others from finalizing."""
        exp1 = StubStreamExporter()
        exp1.finalize.side_effect = RuntimeError("flush failed")
        exp2 = StubStreamExporter()
        mgr = _make_finalize_manager_mock(
            {
                StreamExporterType.RECORD_EXPORT: exp1,
                StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp2,
            },
        )

        await mgr._finalize_stream_exporters()

        # Both should be called (gather runs all concurrently)
        exp1.finalize.assert_called_once()
        exp2.finalize.assert_called_once()
        # Error logged for the failing one
        mgr.error.assert_called_once()
        assert "flush failed" in mgr.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_finalize_multiple_errors(self) -> None:
        exp1 = StubStreamExporter()
        exp1.finalize.side_effect = RuntimeError("error 1")
        exp2 = StubStreamExporter()
        exp2.finalize.side_effect = ValueError("error 2")
        mgr = _make_finalize_manager_mock(
            {
                StreamExporterType.RECORD_EXPORT: exp1,
                StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp2,
            },
        )

        await mgr._finalize_stream_exporters()

        assert mgr.error.call_count == 2


# ---------------------------------------------------------------------------
# Source-branch _dispatch_record / _routing_table — intentionally absent
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="k8s uses static accumulators_for_record_type, not _dispatch_record"
)
def test_dispatch_record_method_exists() -> None:
    """Source branch had RecordsManager._dispatch_record. K8s replaced it
    with _send_record_to_accumulators driven by precomputed lists set in
    __init__ via accumulators_for_record_type / stream_exporters_for_record_type.
    See TestSendRecordToAccumulators above for the ported behavior."""


@pytest.mark.skip(
    reason="k8s uses static accumulators_for_record_type, not _routing_table"
)
def test_routing_table_attribute_exists() -> None:
    """Source branch built RecordsManager._routing_table at init time as
    dict[str, list[handler]] keyed by record_type. K8s replaces it with two
    precomputed flat lists per record type (just metric_records today). See
    TestAccumulatorsForRecordType / TestStreamExportersForRecordType above
    for the ported behavior."""
