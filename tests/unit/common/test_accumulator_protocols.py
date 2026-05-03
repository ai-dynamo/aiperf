# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from aiperf.common.accumulator_protocols import (
    AccumulatorProtocol,
    AccumulatorResult,
    AnalyzerProtocol,
    ExportContext,
    StreamExporterProtocol,
    SummaryContext,
)
from aiperf.plugin.enums import AccumulatorType

# ---------------------------------------------------------------------------
# Stub AccumulatorResult implementation
# ---------------------------------------------------------------------------


@dataclass
class StubResult:
    values: list[int]

    def to_json(self) -> list[int]:
        return self.values

    def to_csv(self) -> list[dict[str, Any]]:
        return [{"value": v} for v in self.values]


# ---------------------------------------------------------------------------
# Stub implementations for isinstance checks
# ---------------------------------------------------------------------------


class StubAccumulator:
    async def process_record(self, record: Any) -> None:
        pass

    def query_time_range(self, start_ns: int, end_ns: int) -> NDArray[np.bool_]:
        return np.array([], dtype=bool)

    async def summarize(self, ctx: SummaryContext | None = None) -> StubResult:
        return StubResult(values=[])

    async def export_results(self, ctx: ExportContext) -> StubResult:
        return StubResult(values=[])


class StubAnalyzer:
    required_accumulators: ClassVar[set[AccumulatorType]] = set()
    summary_dependencies: ClassVar[list[AccumulatorType]] = []

    async def summarize(self, ctx: SummaryContext) -> Any:
        return []


class StubStreamExporter:
    async def process_record(self, record: Any) -> None:
        pass

    async def finalize(self) -> None:
        pass

    def get_export_info(self) -> Any:
        return None


class NotAnAccumulator:
    """Missing required methods — should NOT satisfy any protocol."""

    pass


# ---------------------------------------------------------------------------
# Protocol isinstance checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance, protocol, expected",
    [
        pytest.param(
            StubAccumulator(), AccumulatorProtocol, True, id="accumulator-matches"
        ),
        pytest.param(StubAnalyzer(), AnalyzerProtocol, True, id="analyzer-matches"),
        pytest.param(
            StubStreamExporter(),
            StreamExporterProtocol,
            True,
            id="stream-exporter-matches",
        ),
        pytest.param(
            NotAnAccumulator(), AccumulatorProtocol, False, id="not-accumulator"
        ),
        pytest.param(NotAnAccumulator(), AnalyzerProtocol, False, id="not-analyzer"),
        pytest.param(
            NotAnAccumulator(), StreamExporterProtocol, False, id="not-stream-exporter"
        ),
        pytest.param(
            StubStreamExporter(),
            AccumulatorProtocol,
            False,
            id="exporter-is-not-accumulator",
        ),
        # StreamExporterProtocol now uses finalize() instead of summarize(),
        # so accumulators no longer structurally match the exporter protocol.
        pytest.param(
            StubAccumulator(),
            StreamExporterProtocol,
            False,
            id="accumulator-does-not-match-exporter",
        ),
    ],
)
def test_protocol_isinstance_check(
    instance: object, protocol: type, expected: bool
) -> None:
    assert isinstance(instance, protocol) is expected


def test_protocols_are_unambiguous() -> None:
    """Accumulators and stream exporters are now fully distinct.

    StreamExporterProtocol uses finalize() while AccumulatorProtocol uses
    summarize(), so there is no structural overlap between the two protocols.
    """
    acc = StubAccumulator()
    exp = StubStreamExporter()

    assert isinstance(acc, AccumulatorProtocol) is True
    assert isinstance(acc, StreamExporterProtocol) is False
    assert isinstance(exp, AccumulatorProtocol) is False
    assert isinstance(exp, StreamExporterProtocol) is True


# ---------------------------------------------------------------------------
# AccumulatorResult protocol tests
# ---------------------------------------------------------------------------


class TestAccumulatorResult:
    def test_stub_result_satisfies_protocol(self) -> None:
        result = StubResult(values=[1, 2, 3])
        assert isinstance(result, AccumulatorResult)

    def test_to_json(self) -> None:
        result = StubResult(values=[1, 2, 3])
        assert result.to_json() == [1, 2, 3]

    def test_to_csv(self) -> None:
        result = StubResult(values=[10, 20])
        assert result.to_csv() == [{"value": 10}, {"value": 20}]

    def test_missing_to_json_does_not_satisfy(self) -> None:
        class NoJson:
            def to_csv(self) -> list[dict[str, Any]]:
                return []

        assert not isinstance(NoJson(), AccumulatorResult)

    def test_missing_to_csv_does_not_satisfy(self) -> None:
        class NoCsv:
            def to_json(self) -> Any:
                return {}

        assert not isinstance(NoCsv(), AccumulatorResult)

    def test_plain_object_does_not_satisfy(self) -> None:
        assert not isinstance(object(), AccumulatorResult)


# ---------------------------------------------------------------------------
# SummaryContext tests
# ---------------------------------------------------------------------------


class TestSummaryContext:
    def test_default_construction(self) -> None:
        ctx = SummaryContext()
        assert ctx.accumulators == {}
        assert ctx.accumulator_outputs == {}
        assert ctx.start_ns == 0
        assert ctx.end_ns == 0
        assert ctx.cancelled is False

    def test_get_accumulator_present(self) -> None:
        sentinel = object()
        ctx = SummaryContext(accumulators={AccumulatorType.METRIC_RESULTS: sentinel})
        assert ctx.get_accumulator(AccumulatorType.METRIC_RESULTS) is sentinel

    def test_get_accumulator_missing(self) -> None:
        ctx = SummaryContext()
        assert ctx.get_accumulator(AccumulatorType.METRIC_RESULTS) is None

    def test_get_output_present(self) -> None:
        sentinel = object()
        ctx = SummaryContext(
            accumulator_outputs={AccumulatorType.GPU_TELEMETRY: sentinel}
        )
        assert ctx.get_output(AccumulatorType.GPU_TELEMETRY) is sentinel

    def test_get_output_missing(self) -> None:
        ctx = SummaryContext()
        assert ctx.get_output(AccumulatorType.GPU_TELEMETRY) is None

    def test_cancelled_flag(self) -> None:
        ctx = SummaryContext(cancelled=True)
        assert ctx.cancelled is True

    def test_time_range(self) -> None:
        ctx = SummaryContext(start_ns=1_000_000, end_ns=2_000_000)
        assert ctx.start_ns == 1_000_000
        assert ctx.end_ns == 2_000_000

    def test_accumulator_outputs_mutable(self) -> None:
        ctx = SummaryContext()
        ctx.accumulator_outputs[AccumulatorType.METRIC_RESULTS] = [1, 2, 3]
        assert ctx.get_output(AccumulatorType.METRIC_RESULTS) == [1, 2, 3]


# ---------------------------------------------------------------------------
# ExportContext tests
# ---------------------------------------------------------------------------


class TestExportContext:
    def test_default_construction(self) -> None:
        ctx = ExportContext()
        assert ctx.start_ns is None
        assert ctx.end_ns is None
        assert ctx.error_summary is None
        assert ctx.cancelled is False

    def test_cancelled_flag(self) -> None:
        ctx = ExportContext(cancelled=True)
        assert ctx.cancelled is True

    def test_with_time_range(self) -> None:
        ctx = ExportContext(start_ns=1_000, end_ns=2_000)
        assert ctx.start_ns == 1_000
        assert ctx.end_ns == 2_000
