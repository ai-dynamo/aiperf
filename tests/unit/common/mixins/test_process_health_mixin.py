# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ProcessHealthMixin."""

from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.common.mixins.process_health_mixin import ProcessHealthMixin
from aiperf.common.models import CPUTimes, CtxSwitches, ProcessHealth

# psutil returns these named tuples
_PsutilCpuTimes = namedtuple(
    "pcputimes", ["user", "system", "children_user", "children_system", "iowait"]
)
_PsutilCpuTimesShort = namedtuple(
    "pcputimes", ["user", "system", "children_user", "children_system"]
)
_PsutilCtxSwitches = namedtuple("pctxsw", ["voluntary", "involuntary"])
_PsutilMemInfo = namedtuple("pmem", ["rss", "vms"])
_PsutilIOCounters = namedtuple(
    "pio",
    [
        "read_count",
        "write_count",
        "read_bytes",
        "write_bytes",
        "read_chars",
        "write_chars",
    ],
)


def _make_mock_process(
    *,
    pid: int = 42,
    create_time: float = 1000.0,
    cpu_percent: float = 12.5,
    cpu_times: tuple = _PsutilCpuTimes(1.0, 0.5, 0.0, 0.0, 0.1),
    memory_rss: int = 100_000_000,
    io_counters: tuple | None = _PsutilIOCounters(10, 20, 1000, 2000, 3000, 4000),
    ctx_switches: tuple = _PsutilCtxSwitches(500, 50),
    num_threads: int = 8,
    has_io_counters: bool = True,
) -> MagicMock:
    """Build a mock psutil.Process with realistic return values."""
    proc = MagicMock()
    proc.pid = pid
    proc.create_time.return_value = create_time
    proc.cpu_percent.return_value = cpu_percent
    proc.cpu_times.return_value = cpu_times
    proc.memory_info.return_value = _PsutilMemInfo(rss=memory_rss, vms=200_000_000)
    proc.num_ctx_switches.return_value = ctx_switches
    proc.num_threads.return_value = num_threads
    if has_io_counters:
        proc.io_counters.return_value = io_counters
    else:
        del proc.io_counters
    return proc


class _Host(ProcessHealthMixin):
    """Minimal host class satisfying the mixin's MRO (BaseMixin expects **kwargs)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TestProcessHealthMixinInit:
    """Tests for __init__ behavior."""

    def test_init_calls_cpu_percent_to_prime(self) -> None:
        """cpu_percent() is called once during init to discard the first 0-result."""
        mock_proc = _make_mock_process()
        with patch("psutil.Process", return_value=mock_proc):
            _Host()
        mock_proc.cpu_percent.assert_called_once()

    def test_init_stores_create_time(self) -> None:
        mock_proc = _make_mock_process(create_time=999.0)
        with patch("psutil.Process", return_value=mock_proc):
            host = _Host()
        assert host._create_time == 999.0

    def test_previous_and_health_start_none(self) -> None:
        mock_proc = _make_mock_process()
        with patch("psutil.Process", return_value=mock_proc):
            host = _Host()
        assert host._process_health is None
        assert host._previous is None


class TestGetProcessHealth:
    """Tests for get_process_health()."""

    @pytest.fixture
    def host(self) -> _Host:
        mock_proc = _make_mock_process()
        with patch("psutil.Process", return_value=mock_proc):
            h = _Host()
        # Reset cpu_percent call count from __init__
        mock_proc.cpu_percent.reset_mock()
        return h

    def test_returns_process_health(self, host: _Host) -> None:
        health = host.get_process_health()
        assert isinstance(health, ProcessHealth)

    def test_populates_all_fields(self, host: _Host) -> None:
        health = host.get_process_health()
        assert health.pid == 42
        assert health.create_time == 1000.0
        assert health.cpu_usage == 12.5
        assert health.memory_usage == 100_000_000
        assert health.num_threads == 8
        assert health.cpu_times == CPUTimes(user=1.0, system=0.5, iowait=0.1)
        assert health.num_ctx_switches == CtxSwitches(voluntary=500, involuntary=50)
        assert health.io_counters is not None

    def test_uptime_is_positive(self, host: _Host) -> None:
        health = host.get_process_health()
        assert health.uptime > 0

    def test_previous_tracks_last_health(self, host: _Host) -> None:
        first = host.get_process_health()
        second = host.get_process_health()
        assert host._previous is first
        assert host._process_health is second

    @pytest.mark.parametrize(
        "cpu_times_tuple,expected_iowait",
        [
            param(
                _PsutilCpuTimes(2.0, 1.0, 0.0, 0.0, 0.3),
                0.3,
                id="linux_with_iowait",
            ),
            param(
                _PsutilCpuTimesShort(2.0, 1.0, 0.0, 0.0),
                0.0,
                id="macos_no_iowait",
            ),
        ],
    )  # fmt: skip
    def test_iowait_fallback_when_missing(
        self, cpu_times_tuple: tuple, expected_iowait: float
    ) -> None:
        mock_proc = _make_mock_process(cpu_times=cpu_times_tuple)
        with patch("psutil.Process", return_value=mock_proc):
            host = _Host()
        health = host.get_process_health()
        assert health.cpu_times.iowait == expected_iowait

    def test_io_counters_none_when_not_available(self) -> None:
        mock_proc = _make_mock_process(has_io_counters=False)
        with patch("psutil.Process", return_value=mock_proc):
            host = _Host()
        health = host.get_process_health()
        assert health.io_counters is None


class TestGetPssMemory:
    """Tests for get_pss_memory()."""

    def test_returns_pss_value(self) -> None:
        mock_proc = _make_mock_process()
        with patch("psutil.Process", return_value=mock_proc):
            host = _Host()
        with patch(
            "aiperf.common.mixins.process_health_mixin.read_pss_self",
            return_value=50_000_000,
        ):
            assert host.get_pss_memory() == 50_000_000

    def test_returns_none_when_unavailable(self) -> None:
        mock_proc = _make_mock_process()
        with patch("psutil.Process", return_value=mock_proc):
            host = _Host()
        with patch(
            "aiperf.common.mixins.process_health_mixin.read_pss_self",
            return_value=None,
        ):
            assert host.get_pss_memory() is None
