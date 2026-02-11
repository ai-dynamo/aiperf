# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.exceptions import MetricUnitError
from aiperf.common.models import MetricResult
from aiperf.metrics.display_units import (
    _logger,
    to_display_unit,
)


class FakeUnit:
    def __init__(self, name: str, raise_on_convert: bool = False):
        self.value = name
        self._raise = raise_on_convert

    def __eq__(self, other):
        return isinstance(other, FakeUnit) and self.value == other.value

    def convert_to(self, target: "FakeUnit", v: float) -> float:
        if self._raise:
            raise MetricUnitError("Exception raised")
        if self.value == target.value:
            return v
        if self.value == "ns" and target.value == "ms":
            return v / NANOS_PER_MILLIS
        if self.value == "ms" and target.value == "ns":
            return v * NANOS_PER_MILLIS
        raise AssertionError(f"unsupported conversion {self.value}->{target.value}")


class FakeMetric:
    def __init__(self, base: FakeUnit, display: FakeUnit | None):
        self.unit = base
        self.display_unit = display or base
        self.display_order = 0


class FakeRegistry:
    def __init__(
        self,
        base_unit: str,
        display_unit: str | None = None,
        raise_on_convert: bool = False,
    ):
        base = FakeUnit(base_unit, raise_on_convert=raise_on_convert)
        disp = FakeUnit(display_unit) if display_unit else None
        self._metric = FakeMetric(base, disp)

    def get_class(self, _tag):
        return self._metric


class TestDisplayUnitsUtils:
    def test_noop_when_display_equals_base(self):
        reg = FakeRegistry(base_unit="ms", display_unit="ms")
        src = MetricResult(
            tag="request_latency", unit="ms", header="RL", avg=10.0, p90=12.0
        )
        out = to_display_unit(src, reg)
        # No conversion -> same object to keep it cheap
        assert out is src
        assert out.avg == 10.0
        assert out.unit == "ms"

    def test_converts_ns_to_ms_and_returns_copy(self):
        reg = FakeRegistry(base_unit="ns", display_unit="ms")
        src = MetricResult(
            tag="time_to_first_token",
            unit="ns",
            header="TTFT",
            avg=1_500_000.0,
            min=None,
            max=2_000_000.0,
            p90=1_550_000.0,
            p75=1_230_000.0,
            count=7,
        )
        out = to_display_unit(src, reg)
        assert out is not src
        assert out.unit == "ms"
        assert out.avg == pytest.approx(1.5)
        assert out.max == pytest.approx(2.0)
        assert out.p90 == pytest.approx(1.55)
        assert out.p75 == pytest.approx(1.23)
        # count isn't in STAT_KEYS and must not be converted/touched
        assert out.count == 7
        assert src.avg == 1_500_000.0  # original left untouched

    def test_preserves_none_fields(self):
        reg = FakeRegistry(base_unit="ns", display_unit="ms")
        src = MetricResult(
            tag="time_to_first_token",
            unit="ns",
            header="TTFT",
            avg=1_000_000.0,
            p95=None,
        )
        out = to_display_unit(src, reg)
        assert out.p95 is None
        assert out.avg == pytest.approx(1.0)

    def test_logs_error_on_unit_mismatch(self, monkeypatch):
        err_mock = Mock()
        monkeypatch.setattr(_logger, "error", err_mock)
        reg = FakeRegistry(base_unit="ns", display_unit="ms")
        # record claims "ms" but base is "ns"
        src = MetricResult(
            tag="time_to_first_token", unit="ms", header="TTFT", avg=1_000_000.0
        )
        to_display_unit(src, reg)
        assert err_mock.call_count == 1
        msg = err_mock.call_args[0][0]
        assert "does not match the expected unit (ns)" in msg

    def test_warns_and_continues_when_convert_raises(self, monkeypatch):
        warn_mock = Mock()
        monkeypatch.setattr(_logger, "warning", warn_mock)
        # Force convert_to to raise
        reg = FakeRegistry(base_unit="ns", display_unit="ms", raise_on_convert=True)
        src = MetricResult(
            tag="time_to_first_token", unit="ns", header="TTFT", avg=1_000_000.0
        )
        out = to_display_unit(src, reg)
        # Unit string still updated to display (ms), value left as original (since conversion failed)
        assert out.unit == "ms"
        assert out.avg == 1_000_000.0
        assert warn_mock.call_count == 1
