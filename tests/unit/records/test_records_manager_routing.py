# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for metadata-driven record routing in ``RecordsManager``.

``RecordsManager`` builds a ``record_type -> handlers`` table from the
``record_types`` metadata on accumulator and stream-exporter plugins. The hot path
then dispatches each typed record to the handlers for its ``record_type``.
"""

from __future__ import annotations

import asyncio
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
from aiperf.common.exceptions import PluginDisabled
from aiperf.plugin.enums import AccumulatorType, PluginType, StreamExporterType
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_manager_processing import (
    load_accumulators,
)

# ---------------------------------------------------------------------------
# Fake plugin entries
# ---------------------------------------------------------------------------


def _make_entry(name: str, record_types: list[str]) -> MagicMock:
    """Build a fake PluginEntry-shaped MagicMock with ``record_types`` metadata."""
    entry = MagicMock()
    entry.name = name
    entry.metadata = {"record_types": record_types}
    return entry


# ---------------------------------------------------------------------------
# Stub handlers (protocol-conformant)
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


def _set_plugin_entries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    accumulator_entries: list[MagicMock] | None = None,
    stream_exporter_entries: list[MagicMock] | None = None,
) -> None:
    accumulator_entries = accumulator_entries or []
    stream_exporter_entries = stream_exporter_entries or []

    def _iter_entries(category: PluginType):
        if category == PluginType.ACCUMULATOR:
            return iter(accumulator_entries)
        if category == PluginType.STREAM_EXPORTER:
            return iter(stream_exporter_entries)
        return iter(())

    monkeypatch.setattr(
        "aiperf.records.records_manager.plugins.iter_entries",
        _iter_entries,
    )


# ---------------------------------------------------------------------------
# Tests: _build_routing_table
# ---------------------------------------------------------------------------


class TestBuildRoutingTable:
    """Plugin metadata controls which handlers receive each record type."""

    def test_single_accumulator_matches_record_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acc = StubAccumulator()
        manager = RecordsManager.__new__(RecordsManager)
        manager._accumulators = {AccumulatorType.METRIC_RESULTS: acc}
        manager._stream_exporters = {}
        _set_plugin_entries(
            monkeypatch,
            accumulator_entries=[_make_entry("metric_results", ["metric_records"])],
        )

        table = manager._build_routing_table()

        assert table == {"metric_records": [acc]}

    def test_only_matching_handlers_are_routed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        metric_acc = StubAccumulator()
        telemetry_acc = StubAccumulator()
        exporter = StubStreamExporter()
        manager = RecordsManager.__new__(RecordsManager)
        manager._accumulators = {
            AccumulatorType.METRIC_RESULTS: metric_acc,
            AccumulatorType.GPU_TELEMETRY: telemetry_acc,
        }
        manager._stream_exporters = {
            StreamExporterType.SERVER_METRICS_JSONL_WRITER: exporter,
        }
        _set_plugin_entries(
            monkeypatch,
            accumulator_entries=[
                _make_entry("metric_results", ["metric_records"]),
                _make_entry("gpu_telemetry", ["gpu_telemetry"]),
            ],
            stream_exporter_entries=[
                _make_entry("server_metrics_jsonl_writer", ["metric_records"]),
            ],
        )

        table = manager._build_routing_table()

        assert table["metric_records"] == [metric_acc, exporter]
        assert table["gpu_telemetry"] == [telemetry_acc]

    def test_skips_entries_not_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        acc = StubAccumulator()
        manager = RecordsManager.__new__(RecordsManager)
        manager._accumulators = {AccumulatorType.METRIC_RESULTS: acc}
        manager._stream_exporters = {}
        _set_plugin_entries(
            monkeypatch,
            accumulator_entries=[
                _make_entry("metric_results", ["metric_records"]),
                _make_entry("server_metrics", ["metric_records"]),
            ],
        )

        table = manager._build_routing_table()

        assert table["metric_records"] == [acc]

    def test_empty_loaded_handlers_returns_empty_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._accumulators = {}
        manager._stream_exporters = {}
        _set_plugin_entries(
            monkeypatch,
            accumulator_entries=[_make_entry("metric_results", ["metric_records"])],
        )

        assert manager._build_routing_table() == {}


# ---------------------------------------------------------------------------
# Tests: _dispatch_record
# ---------------------------------------------------------------------------


class TestDispatchRecord:
    """Per-record fan-out uses the metadata-derived routing table."""

    def _manager(self, routing_table: dict[str, list[Any]]) -> RecordsManager:
        manager = RecordsManager.__new__(RecordsManager)
        manager._routing_table = routing_table
        manager._warned_unrouted_record_types = set()
        manager.error = MagicMock()
        manager.debug = MagicMock()
        manager.warning = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_dispatch_calls_all_handlers(self) -> None:
        acc = StubAccumulator()
        exp = StubStreamExporter()
        manager = self._manager({"metric_records": [acc, exp]})
        record = MagicMock(record_type="metric_records")

        errors = await manager._dispatch_record(record)

        assert errors == []
        acc.process_record.assert_awaited_once_with(record)
        exp.process_record.assert_awaited_once_with(record)

    @pytest.mark.asyncio
    async def test_dispatch_with_no_handlers_warns_once(self) -> None:
        manager = self._manager({})
        record = MagicMock(record_type="metric_records")

        errors = await manager._dispatch_record(record)
        # A second unrouted record of the same type must not re-warn.
        errors += await manager._dispatch_record(record)

        assert errors == []
        manager.warning.assert_called_once()
        manager.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_unrouted_control_plane_does_not_warn(self) -> None:
        # Control-plane records (e.g. credit_phase_stats) are consumed elsewhere
        # and only OPTIONALLY streamed, so an absent handler is expected -- passing
        # warn_if_unrouted=False must suppress the misleading "records dropped" warning.
        manager = self._manager({})
        record = MagicMock(record_type="credit_phase_stats")

        errors = await manager._dispatch_record(record, warn_if_unrouted=False)

        assert errors == []
        manager.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_missing_record_type_returns_error(self) -> None:
        manager = self._manager({})
        record = object()

        errors = await manager._dispatch_record(record)

        assert len(errors) == 1
        assert isinstance(errors[0], TypeError)
        manager.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception_logged_and_returned(self) -> None:
        acc = StubAccumulator()
        acc.process_record.side_effect = RuntimeError("boom")
        exp = StubStreamExporter()
        manager = self._manager({"metric_records": [acc, exp]})
        record = MagicMock(record_type="metric_records")

        errors = await manager._dispatch_record(record)

        exp.process_record.assert_awaited_once_with(record)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        manager.error.assert_called_once()
        assert "boom" in manager.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_dispatch_best_effort_handler_exception_not_counted(self) -> None:
        # A best-effort handler (streaming telemetry like OTel/MLflow) that raises
        # must be logged but NOT counted -- a downed collector is not an inference
        # failure and must not pollute the benchmark's phase error summary.
        best_effort = StubStreamExporter()
        best_effort.is_best_effort = True
        best_effort.process_record.side_effect = RuntimeError("collector down")
        strict = StubStreamExporter()
        strict.process_record.side_effect = ValueError("real failure")
        manager = self._manager({"metric_records": [best_effort, strict]})

        errors = await manager._dispatch_record(MagicMock(record_type="metric_records"))

        # Only the strict handler's failure is returned; both are logged.
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert manager.error.call_count == 2

    @pytest.mark.asyncio
    async def test_dispatch_multiple_handler_exceptions(self) -> None:
        acc = StubAccumulator()
        acc.process_record.side_effect = RuntimeError("acc error")
        exp = StubStreamExporter()
        exp.process_record.side_effect = ValueError("exp error")
        manager = self._manager({"metric_records": [acc, exp]})

        errors = await manager._dispatch_record(MagicMock(record_type="metric_records"))

        assert len(errors) == 2
        assert manager.error.call_count == 2

    @pytest.mark.asyncio
    async def test_handler_cancelled_error_is_counted_not_reraised(self) -> None:
        # A handler-level CancelledError (captured by gather's return_exceptions)
        # means one handler's coroutine was cancelled, not this task. It must be
        # returned as an error -- not re-raised -- so the caller still advances the
        # tracker update and the (timeout-less) completion barrier. Genuine task
        # cancellation makes the gather itself raise and never reaches this code.
        acc = StubAccumulator()
        acc.process_record.side_effect = asyncio.CancelledError
        manager = self._manager({"metric_records": [acc]})

        errors = await manager._dispatch_record(MagicMock(record_type="metric_records"))

        assert len(errors) == 1
        assert isinstance(errors[0], asyncio.CancelledError)
        manager.error.assert_called_once()


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
                StreamExporterType.SERVER_METRICS_JSONL_WRITER: exp1,
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
        mgr.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_finalize_error_logged_per_exporter(self) -> None:
        """One exporter failing does not prevent others from finalizing."""
        exp1 = StubStreamExporter()
        exp1.finalize.side_effect = RuntimeError("flush failed")
        exp2 = StubStreamExporter()
        mgr = _make_finalize_manager_mock(
            {
                StreamExporterType.SERVER_METRICS_JSONL_WRITER: exp1,
                StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp2,
            },
        )

        await mgr._finalize_stream_exporters()

        exp1.finalize.assert_called_once()
        exp2.finalize.assert_called_once()
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
                StreamExporterType.SERVER_METRICS_JSONL_WRITER: exp1,
                StreamExporterType.GPU_TELEMETRY_JSONL_WRITER: exp2,
            },
        )

        await mgr._finalize_stream_exporters()

        assert mgr.error.call_count == 2


# ---------------------------------------------------------------------------
# Tests: load_accumulators construction-failure policy
# ---------------------------------------------------------------------------


def _make_loader_host() -> MagicMock:
    """Minimal ``_LoaderHost``-shaped MagicMock for load_accumulators."""
    host = MagicMock()
    host.service_id = "records-manager"
    host.run = MagicMock()
    host.pub_client = MagicMock()
    host.attach_child_lifecycle = MagicMock()
    host.debug = MagicMock()
    host.error = MagicMock()
    return host


class _RaisingAccumulator:
    """Accumulator whose construction always fails with a generic error."""

    def __init__(self, **kwargs: Any) -> None:
        raise RuntimeError("accumulator __init__ exploded")


class _DisabledAccumulator:
    """Accumulator whose construction opts out via PluginDisabled."""

    def __init__(self, **kwargs: Any) -> None:
        raise PluginDisabled("metric_results disabled")


class TestLoadAccumulatorsConstructionFailure:
    """The load-bearing metric_results accumulator must fail fast on a
    construction error (no fallback summary producer), while optional
    accumulators degrade gracefully and the explicit disable opt-out stays
    a silent skip."""

    def test_load_accumulators_metric_results_construction_failure_reraises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entries = [_make_entry("metric_results", ["metric_records"])]
        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda _plugin_type: entries,
        )
        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.get_class",
            lambda _plugin_type, _name: _RaisingAccumulator,
        )
        host = _make_loader_host()

        with pytest.raises(RuntimeError, match="accumulator __init__ exploded"):
            load_accumulators(host)

    def test_load_accumulators_optional_accumulator_failure_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entries = [_make_entry("gpu_telemetry", ["gpu_telemetry"])]
        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda _plugin_type: entries,
        )
        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.get_class",
            lambda _plugin_type, _name: _RaisingAccumulator,
        )
        host = _make_loader_host()

        accumulators = load_accumulators(host)

        assert AccumulatorType.GPU_TELEMETRY not in accumulators
        host.error.assert_called_once()

    def test_load_accumulators_metric_results_disabled_is_silent_skip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entries = [_make_entry("metric_results", ["metric_records"])]
        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.iter_entries",
            lambda _plugin_type: entries,
        )
        monkeypatch.setattr(
            "aiperf.records.records_manager_processing.plugins.get_class",
            lambda _plugin_type, _name: _DisabledAccumulator,
        )
        host = _make_loader_host()

        accumulators = load_accumulators(host)

        assert AccumulatorType.METRIC_RESULTS not in accumulators
        host.error.assert_not_called()
