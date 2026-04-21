# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the memory tracker module."""

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.memory_tracker import (
    MemoryPhase,
    MemoryReading,
    MemorySnapshot,
    MemoryTracker,
    read_memory_self,
    read_pss_self,
)

# ---------------------------------------------------------------------------
# MemoryPhase
# ---------------------------------------------------------------------------


class TestMemoryPhase:
    def test_values(self):
        assert MemoryPhase.STARTUP == "startup"
        assert MemoryPhase.POST_CONFIG == "post_config"
        assert MemoryPhase.SHUTDOWN == "shutdown"

    def test_case_insensitive_lookup(self):
        assert MemoryPhase("STARTUP") == MemoryPhase.STARTUP
        assert MemoryPhase("Shutdown") == MemoryPhase.SHUTDOWN
        assert MemoryPhase("Post_Config") == MemoryPhase.POST_CONFIG


# ---------------------------------------------------------------------------
# MemoryReading
# ---------------------------------------------------------------------------


class TestMemoryReading:
    def test_defaults_all_none(self):
        reading = MemoryReading()
        assert reading.pss is None
        assert reading.rss is None
        assert reading.uss is None
        assert reading.shared is None

    def test_with_values(self):
        reading = MemoryReading(pss=100, rss=200, uss=50, shared=150)
        assert reading.pss == 100
        assert reading.rss == 200
        assert reading.uss == 50
        assert reading.shared == 150

    def test_partial_values(self):
        reading = MemoryReading(pss=100)
        assert reading.pss == 100
        assert reading.rss is None


# ---------------------------------------------------------------------------
# MemorySnapshot
# ---------------------------------------------------------------------------


class TestMemorySnapshot:
    def test_creation(self):
        snap = MemorySnapshot(pid=1234, label="worker_0", group="worker")
        assert snap.pid == 1234
        assert snap.label == "worker_0"
        assert snap.group == "worker"

    def test_set_and_get_reading(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        reading = MemoryReading(pss=100)
        snap.set_reading(MemoryPhase.STARTUP, reading)
        assert snap.get_reading(MemoryPhase.STARTUP) is reading

    def test_get_reading_missing_returns_none(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        assert snap.get_reading(MemoryPhase.STARTUP) is None

    def test_startup_property(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        assert snap.startup is None
        reading = MemoryReading(pss=10)
        snap.set_reading(MemoryPhase.STARTUP, reading)
        assert snap.startup is reading

    def test_post_config_property(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        assert snap.post_config is None
        reading = MemoryReading(pss=20)
        snap.set_reading(MemoryPhase.POST_CONFIG, reading)
        assert snap.post_config is reading

    def test_shutdown_property(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        assert snap.shutdown is None
        reading = MemoryReading(pss=30)
        snap.set_reading(MemoryPhase.SHUTDOWN, reading)
        assert snap.shutdown is reading

    def test_set_reading_overwrites(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        snap.set_reading(MemoryPhase.STARTUP, MemoryReading(pss=10))
        snap.set_reading(MemoryPhase.STARTUP, MemoryReading(pss=20))
        assert snap.startup.pss == 20

    def test_multiple_phases(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        snap.set_reading(MemoryPhase.STARTUP, MemoryReading(pss=10))
        snap.set_reading(MemoryPhase.POST_CONFIG, MemoryReading(pss=50))
        snap.set_reading(MemoryPhase.SHUTDOWN, MemoryReading(pss=100))
        assert snap.startup.pss == 10
        assert snap.post_config.pss == 50
        assert snap.shutdown.pss == 100

    def test_capture_reads_own_process(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        reading = snap.capture(MemoryPhase.STARTUP)
        assert reading is not None
        assert isinstance(reading, MemoryReading)
        assert reading.pss is not None
        assert reading.pss > 0
        assert reading.rss is not None
        assert snap.startup is reading

    def test_capture_returns_none_on_failure(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        with patch("aiperf.common.memory_tracker.read_memory_self", return_value=None):
            reading = snap.capture(MemoryPhase.STARTUP)
        assert reading is None
        assert snap.startup is None

    def test_capture_overwrites_existing(self):
        snap = MemorySnapshot(pid=1, label="a", group="g")
        snap.set_reading(MemoryPhase.STARTUP, MemoryReading(pss=1))
        reading = snap.capture(MemoryPhase.STARTUP)
        assert reading is not None
        assert snap.startup is reading
        assert snap.startup.pss != 1


# ---------------------------------------------------------------------------
# MemoryTracker
# ---------------------------------------------------------------------------


class TestMemoryTracker:
    @pytest.fixture
    def tracker(self):
        return MemoryTracker()

    def test_empty_tracker(self, tracker: MemoryTracker):
        assert tracker.snapshots == {}

    def test_record_creates_snapshot(self, tracker: MemoryTracker):
        tracker.record(
            label="svc_0",
            group="worker",
            pid=100,
            phase=MemoryPhase.STARTUP,
            reading=MemoryReading(pss=1000),
        )
        assert "svc_0" in tracker.snapshots
        snap = tracker.snapshots["svc_0"]
        assert snap.pid == 100
        assert snap.label == "svc_0"
        assert snap.group == "worker"
        assert snap.startup.pss == 1000

    def test_record_same_label_adds_phase(self, tracker: MemoryTracker):
        tracker.record(
            "svc_0", "worker", 100, MemoryPhase.STARTUP, MemoryReading(pss=1000)
        )
        tracker.record(
            "svc_0", "worker", 100, MemoryPhase.SHUTDOWN, MemoryReading(pss=2000)
        )
        snap = tracker.snapshots["svc_0"]
        assert snap.startup.pss == 1000
        assert snap.shutdown.pss == 2000

    def test_record_overwrites_phase(self, tracker: MemoryTracker):
        tracker.record(
            "svc_0", "worker", 100, MemoryPhase.STARTUP, MemoryReading(pss=1000)
        )
        tracker.record(
            "svc_0", "worker", 100, MemoryPhase.STARTUP, MemoryReading(pss=9999)
        )
        assert tracker.snapshots["svc_0"].startup.pss == 9999

    def test_record_multiple_labels(self, tracker: MemoryTracker):
        tracker.record("a", "g1", 1, MemoryPhase.STARTUP, MemoryReading(pss=10))
        tracker.record("b", "g2", 2, MemoryPhase.STARTUP, MemoryReading(pss=20))
        assert len(tracker.snapshots) == 2
        assert tracker.snapshots["a"].startup.pss == 10
        assert tracker.snapshots["b"].startup.pss == 20

    def test_clear(self, tracker: MemoryTracker):
        tracker.record(
            "svc_0", "worker", 100, MemoryPhase.STARTUP, MemoryReading(pss=1000)
        )
        tracker.clear()
        assert tracker.snapshots == {}

    def test_record_all_three_phases(self, tracker: MemoryTracker):
        tracker.record(
            "svc_0", "worker", 100, MemoryPhase.STARTUP, MemoryReading(pss=100)
        )
        tracker.record(
            "svc_0", "worker", 100, MemoryPhase.POST_CONFIG, MemoryReading(pss=500)
        )
        tracker.record(
            "svc_0", "worker", 100, MemoryPhase.SHUTDOWN, MemoryReading(pss=1000)
        )
        snap = tracker.snapshots["svc_0"]
        assert snap.startup.pss == 100
        assert snap.post_config.pss == 500
        assert snap.shutdown.pss == 1000

    def test_record_preserves_full_reading(self, tracker: MemoryTracker):
        reading = MemoryReading(pss=100, rss=200, uss=50, shared=150)
        tracker.record("svc_0", "worker", 100, MemoryPhase.STARTUP, reading)
        stored = tracker.snapshots["svc_0"].startup
        assert stored.pss == 100
        assert stored.rss == 200
        assert stored.uss == 50
        assert stored.shared == 150

    def test_capture_creates_snapshot_and_reads(self, tracker: MemoryTracker):
        reading = tracker.capture("ctrl", "controller", 99, MemoryPhase.SHUTDOWN)
        assert reading is not None
        assert "ctrl" in tracker.snapshots
        snap = tracker.snapshots["ctrl"]
        assert snap.pid == 99
        assert snap.label == "ctrl"
        assert snap.group == "controller"
        assert snap.shutdown is reading
        assert reading.pss is not None
        assert reading.pss > 0

    def test_capture_adds_phase_to_existing(self, tracker: MemoryTracker):
        tracker.record("svc", "g", 1, MemoryPhase.STARTUP, MemoryReading(pss=10))
        reading = tracker.capture("svc", "g", 1, MemoryPhase.SHUTDOWN)
        assert reading is not None
        assert tracker.snapshots["svc"].startup.pss == 10
        assert tracker.snapshots["svc"].shutdown is reading

    def test_capture_returns_none_on_failure(self, tracker: MemoryTracker):
        with patch("aiperf.common.memory_tracker.read_memory_self", return_value=None):
            reading = tracker.capture("svc", "g", 1, MemoryPhase.STARTUP)
        assert reading is None
        assert "svc" in tracker.snapshots
        assert tracker.snapshots["svc"].startup is None


# ---------------------------------------------------------------------------
# MemoryTracker.print_summary
# ---------------------------------------------------------------------------


class TestMemoryTrackerPrintSummary:
    @pytest.fixture
    def tracker(self):
        return MemoryTracker()

    def test_empty_tracker_prints_nothing(self, tracker: MemoryTracker):
        with patch("rich.console.Console.print") as mock_print:
            tracker.print_summary()
        mock_print.assert_not_called()

    def test_prints_table_with_data(self, tracker: MemoryTracker):
        tracker.record(
            "svc_a", "worker", 1, MemoryPhase.STARTUP, MemoryReading(pss=50_000_000)
        )
        tracker.record(
            "svc_a", "worker", 1, MemoryPhase.SHUTDOWN, MemoryReading(pss=100_000_000)
        )
        # Just verify it doesn't raise
        tracker.print_summary(title="Test Memory")

    def test_prints_with_post_config(self, tracker: MemoryTracker):
        tracker.record(
            "svc_a", "worker", 1, MemoryPhase.STARTUP, MemoryReading(pss=10_000_000)
        )
        tracker.record(
            "svc_a", "worker", 1, MemoryPhase.POST_CONFIG, MemoryReading(pss=50_000_000)
        )
        tracker.record(
            "svc_a", "worker", 1, MemoryPhase.SHUTDOWN, MemoryReading(pss=100_000_000)
        )
        # Just verify no exception
        tracker.print_summary()

    def test_prints_with_missing_phases(self, tracker: MemoryTracker):
        tracker.record(
            "svc_a", "worker", 1, MemoryPhase.SHUTDOWN, MemoryReading(pss=100_000_000)
        )
        # Only shutdown, no startup — should show N/A for delta
        tracker.print_summary()

    def test_prints_multiple_processes_sorted(self, tracker: MemoryTracker):
        tracker.record("z_svc", "g", 1, MemoryPhase.STARTUP, MemoryReading(pss=100))
        tracker.record("a_svc", "g", 2, MemoryPhase.STARTUP, MemoryReading(pss=200))
        # Sorted output — a_svc before z_svc. Just verify no exception.
        tracker.print_summary()

    def test_custom_title(self, tracker: MemoryTracker):
        tracker.record("svc", "g", 1, MemoryPhase.STARTUP, MemoryReading(pss=100))
        # Verify no exception with custom title
        tracker.print_summary(title="Custom Title")


# ---------------------------------------------------------------------------
# read_memory_self / read_pss_self
# ---------------------------------------------------------------------------


class TestReadMemorySelf:
    def test_returns_memory_reading(self):
        result = read_memory_self()
        assert result is not None
        assert isinstance(result, MemoryReading)
        assert result.pss is not None
        assert result.pss > 0
        assert result.rss is not None
        assert result.rss > 0

    def test_returns_none_on_no_such_process(self):
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(99999)):
            result = read_memory_self()
        assert result is None

    def test_returns_none_on_access_denied(self):
        with patch("psutil.Process", side_effect=psutil.AccessDenied(99999)):
            result = read_memory_self()
        assert result is None

    def test_returns_none_on_attribute_error(self):
        with patch("psutil.Process") as mock_proc:
            mock_proc.return_value.memory_full_info.side_effect = AttributeError
            result = read_memory_self()
        assert result is None


class TestReadPssSelf:
    def test_returns_int(self):
        result = read_pss_self()
        assert result is not None
        assert isinstance(result, int)
        assert result > 0

    def test_returns_none_on_no_such_process(self):
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(99999)):
            result = read_pss_self()
        assert result is None

    def test_returns_none_on_access_denied(self):
        with patch("psutil.Process", side_effect=psutil.AccessDenied(99999)):
            result = read_pss_self()
        assert result is None


# ---------------------------------------------------------------------------
# MemoryReportMessage serialization
# ---------------------------------------------------------------------------


class TestMemoryReportMessage:
    def test_phase_field_is_typed(self):
        from aiperf.common.messages import MemoryReportMessage

        msg = MemoryReportMessage(
            service_id="svc_0",
            service_type="worker",
            pid=1234,
            phase=MemoryPhase.STARTUP,
            pss_bytes=100,
        )
        assert isinstance(msg.phase, MemoryPhase)
        assert msg.phase == MemoryPhase.STARTUP

    @pytest.mark.parametrize(
        "phase",
        [
            param(MemoryPhase.STARTUP, id="startup"),
            param(MemoryPhase.POST_CONFIG, id="post_config"),
            param(MemoryPhase.SHUTDOWN, id="shutdown"),
        ],
    )  # fmt: skip
    def test_all_phases_serialize(self, phase: MemoryPhase):
        from aiperf.common.messages import MemoryReportMessage

        msg = MemoryReportMessage(
            service_id="svc_0",
            service_type="worker",
            pid=1234,
            phase=phase,
            pss_bytes=100,
        )
        assert msg.phase == phase

    def test_roundtrip_serialization(self):
        import orjson

        from aiperf.common.messages import MemoryReportMessage

        msg = MemoryReportMessage(
            service_id="svc_0",
            service_type="worker",
            pid=1234,
            phase=MemoryPhase.POST_CONFIG,
            pss_bytes=100,
            rss_bytes=200,
            uss_bytes=50,
            shared_bytes=150,
        )
        data = orjson.loads(orjson.dumps(msg.model_dump()))
        restored = MemoryReportMessage.from_json(data)
        assert restored.phase == MemoryPhase.POST_CONFIG
        assert restored.pss_bytes == 100
        assert restored.rss_bytes == 200
        assert restored.uss_bytes == 50
        assert restored.shared_bytes == 150

    def test_optional_fields_default_none(self):
        from aiperf.common.messages import MemoryReportMessage

        msg = MemoryReportMessage(
            service_id="svc_0",
            service_type="worker",
            pid=1234,
            phase=MemoryPhase.STARTUP,
            pss_bytes=100,
        )
        assert msg.rss_bytes is None
        assert msg.uss_bytes is None
        assert msg.shared_bytes is None


# We need psutil for the mock patches
import psutil  # noqa: E402
