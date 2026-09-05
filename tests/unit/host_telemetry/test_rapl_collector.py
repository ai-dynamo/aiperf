# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RAPL host telemetry collector.

These build a fake powercap tree on disk, so they need neither an Intel CPU nor
root, and they run on every platform the rest of the suite runs on.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.host_telemetry.rapl_collector import (
    RAPLDomain,
    RAPLTelemetryCollector,
    RAPLUnavailableError,
    discover_domains,
    rapl_is_available,
)

# A real Sapphire Rapids box reports 262143328850 for max_energy_range_uj on the
# package domain, which is where the wraparound tests get their magnitude.
MAX_RANGE = 262143328850


def make_domain(
    parent: Path, name: str, label: str, energy: int | None, max_range=MAX_RANGE
):
    """Create one fake powercap domain directory."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "name").write_text(label + "\n")
    if max_range is not None:
        (d / "max_energy_range_uj").write_text(f"{max_range}\n")
    if energy is not None:
        (d / "energy_uj").write_text(f"{energy}\n")
    return d


@pytest.fixture
def powercap(tmp_path):
    """A two-package tree with core and dram subdomains, all readable."""
    root = tmp_path / "powercap"
    root.mkdir()
    p0 = make_domain(root, "intel-rapl:0", "package-0", 1_000_000)
    make_domain(p0, "intel-rapl:0:0", "core", 400_000)
    make_domain(p0, "intel-rapl:0:1", "dram", 100_000)
    p1 = make_domain(root, "intel-rapl:1", "package-1", 2_000_000)
    make_domain(p1, "intel-rapl:1:0", "core", 800_000)
    return root


class TestDiscovery:
    def test_finds_packages_and_subdomains(self, powercap):
        domains = discover_domains(powercap)
        assert [d.domain_id for d in domains] == [
            "intel-rapl:0",
            "intel-rapl:0:0",
            "intel-rapl:0:1",
            "intel-rapl:1",
            "intel-rapl:1:0",
        ]

    def test_parents_come_before_their_children(self, powercap):
        domains = discover_domains(powercap)
        ids = [d.domain_id for d in domains]
        for d in domains:
            if d.parent_id is not None:
                assert ids.index(d.parent_id) < ids.index(d.domain_id)

    def test_subdomains_record_their_parent(self, powercap):
        by_id = {d.domain_id: d for d in discover_domains(powercap)}
        assert by_id["intel-rapl:0:0"].parent_id == "intel-rapl:0"
        assert by_id["intel-rapl:0:1"].parent_id == "intel-rapl:0"
        assert by_id["intel-rapl:1:0"].parent_id == "intel-rapl:1"
        assert by_id["intel-rapl:0"].parent_id is None

    def test_indices_are_unique_and_dense(self, powercap):
        domains = discover_domains(powercap)
        assert [d.index for d in domains] == list(range(len(domains)))

    def test_reads_the_platform_label(self, powercap):
        by_id = {d.domain_id: d for d in discover_domains(powercap)}
        assert by_id["intel-rapl:0"].name == "package-0"
        assert by_id["intel-rapl:0:1"].name == "dram"

    def test_missing_root_yields_nothing(self, tmp_path):
        assert discover_domains(tmp_path / "absent") == []

    def test_ignores_unrelated_directories(self, tmp_path):
        root = tmp_path / "powercap"
        root.mkdir()
        make_domain(root, "intel-rapl:0", "package-0", 1)
        (root / "dtpm").mkdir()
        (root / "intel-rapl-mmio:0").mkdir()
        assert [d.domain_id for d in discover_domains(root)] == ["intel-rapl:0"]


class TestWraparound:
    def test_normal_increase_passes_through(self, powercap):
        d = RAPLDomain(powercap / "intel-rapl:0", 0)
        assert d.read_energy_uj() == pytest.approx(1_000_000)
        (d.path / "energy_uj").write_text("1500000\n")
        assert d.read_energy_uj() == pytest.approx(1_500_000)

    def test_wrap_is_corrected_using_the_declared_range(self, powercap):
        d = RAPLDomain(powercap / "intel-rapl:0", 0)
        (d.path / "energy_uj").write_text(f"{MAX_RANGE - 1000}\n")
        first = d.read_energy_uj()
        # Counter wraps to a small value.
        (d.path / "energy_uj").write_text("2000\n")
        second = d.read_energy_uj()

        assert second > first, "a wrap must not make the running total go backwards"
        assert second - first == pytest.approx(3000)

    def test_repeated_wraps_accumulate(self, powercap):
        d = RAPLDomain(powercap / "intel-rapl:0", 0)
        (d.path / "energy_uj").write_text("10\n")
        d.read_energy_uj()
        total_before = None
        for _ in range(3):
            (d.path / "energy_uj").write_text(f"{MAX_RANGE - 10}\n")
            d.read_energy_uj()
            (d.path / "energy_uj").write_text("10\n")
            total = d.read_energy_uj()
            if total_before is not None:
                assert total > total_before
            total_before = total
        # Three wraps, so the offset is three full ranges.
        assert d.read_energy_uj() == pytest.approx(10 + 3 * MAX_RANGE)

    def test_wrap_without_a_declared_range_still_never_goes_backwards(self, tmp_path):
        root = tmp_path / "powercap"
        root.mkdir()
        make_domain(root, "intel-rapl:0", "package-0", 900, max_range=None)
        d = RAPLDomain(root / "intel-rapl:0", 0)
        first = d.read_energy_uj()
        (d.path / "energy_uj").write_text("100\n")
        second = d.read_energy_uj()
        assert second >= first

    def test_more_than_one_wrap_between_reads_is_not_recoverable(self, tmp_path):
        # Found by feeding the reader real measured energy from a Raspberry Pi 5
        # PMIC against a deliberately small range. A backwards step is the only
        # evidence of a wrap, so two wraps between reads look exactly like one
        # and the total is short by a range. This pins the bound rather than
        # claiming it does not exist; recovering it needs a timestamped counter,
        # which powercap does not give.
        root = tmp_path / "powercap"
        root.mkdir()
        make_domain(root, "intel-rapl:0", "package-0", 800, max_range=1000)
        d = RAPLDomain(root / "intel-rapl:0", 0)
        assert d.read_energy_uj() == 800

        # True energy advances by 2400 to 3200, crossing the 1000 range twice
        # and landing at raw 200. That is one backwards step, so one range is
        # added and the other is lost.
        (d.path / "energy_uj").write_text("200\n")
        assert d.read_energy_uj() == 1200  # true cumulative energy is 3200

    def test_unreadable_counter_returns_none(self, tmp_path):
        root = tmp_path / "powercap"
        root.mkdir()
        make_domain(root, "intel-rapl:0", "package-0", None)
        d = RAPLDomain(root / "intel-rapl:0", 0)
        assert d.read_energy_uj() is None
        assert d.is_readable() is False

    def test_garbage_counter_returns_none(self, tmp_path):
        root = tmp_path / "powercap"
        root.mkdir()
        p = make_domain(root, "intel-rapl:0", "package-0", 1)
        (p / "energy_uj").write_text("not-a-number\n")
        assert RAPLDomain(p, 0).read_energy_uj() is None


class TestValidateEnvironment:
    def test_rejects_non_linux(self, powercap):
        with (
            patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", False),
            pytest.raises(RAPLUnavailableError, match="Linux"),
        ):
            RAPLTelemetryCollector.validate_environment(powercap)

    def test_rejects_missing_powercap(self, tmp_path):
        with (
            patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True),
            pytest.raises(RAPLUnavailableError, match="does not exist"),
        ):
            RAPLTelemetryCollector.validate_environment(tmp_path / "absent")

    def test_rejects_empty_powercap(self, tmp_path):
        root = tmp_path / "powercap"
        root.mkdir()
        with (
            patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True),
            pytest.raises(RAPLUnavailableError, match="no intel-rapl domains"),
        ):
            RAPLTelemetryCollector.validate_environment(root)

    def test_unreadable_counters_name_the_permission_cause(self, tmp_path):
        """The common real-world failure: domains exist, energy_uj is root-only."""
        root = tmp_path / "powercap"
        root.mkdir()
        make_domain(root, "intel-rapl:0", "package-0", None)
        with (
            patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True),
            pytest.raises(RAPLUnavailableError, match="not running as root"),
        ):
            RAPLTelemetryCollector.validate_environment(root)

    def test_accepts_a_good_tree(self, powercap):
        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            RAPLTelemetryCollector.validate_environment(powercap)

    def test_availability_helper_matches(self, powercap, tmp_path):
        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            assert rapl_is_available(powercap) is True
            assert rapl_is_available(tmp_path / "absent") is False


class TestCollector:
    def test_identity(self, powercap):
        c = RAPLTelemetryCollector(powercap)
        assert c.id == "rapl"
        assert c.endpoint_url == "rapl://localhost"

    @pytest.mark.asyncio
    async def test_initialize_keeps_only_readable_domains(self, tmp_path):
        root = tmp_path / "powercap"
        root.mkdir()
        make_domain(root, "intel-rapl:0", "package-0", 500)
        make_domain(root, "intel-rapl:1", "package-1", None)  # root-only
        c = RAPLTelemetryCollector(root)
        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            await c.initialize()
        assert [d.domain_id for d in c.domains] == ["intel-rapl:0"]

    @pytest.mark.asyncio
    async def test_collect_emits_one_record_per_domain(self, powercap):
        c = RAPLTelemetryCollector(powercap)
        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            await c.initialize()
        records = c.collect()

        assert len(records) == 5
        assert {r.domain.domain_id for r in records} == {
            "intel-rapl:0",
            "intel-rapl:0:0",
            "intel-rapl:0:1",
            "intel-rapl:1",
            "intel-rapl:1:0",
        }
        assert all(r.telemetry_source_url == "rapl://localhost" for r in records)
        assert all(r.timestamp_ns > 0 for r in records)

    @pytest.mark.asyncio
    async def test_collect_shares_one_timestamp(self, powercap):
        """All domains in a sample must carry the same instant, or deltas skew."""
        c = RAPLTelemetryCollector(powercap)
        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            await c.initialize()
        records = c.collect()
        assert len({r.timestamp_ns for r in records}) == 1

    @pytest.mark.asyncio
    async def test_reports_energy_not_derived_power(self, powercap):
        c = RAPLTelemetryCollector(powercap)
        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            await c.initialize()
        record = next(r for r in c.collect() if r.domain.domain_id == "intel-rapl:0")
        assert record.telemetry_data.energy_consumption_uj == pytest.approx(1_000_000)
        assert record.telemetry_data.energy_range_uj == pytest.approx(MAX_RANGE)
        assert record.telemetry_data.power_usage_w is None

    @pytest.mark.asyncio
    async def test_domain_going_unreadable_is_skipped_not_zeroed(self, powercap):
        c = RAPLTelemetryCollector(powercap)
        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            await c.initialize()
        assert len(c.collect()) == 5

        (powercap / "intel-rapl:1" / "energy_uj").unlink()
        records = c.collect()

        assert len(records) == 4
        assert "intel-rapl:1" not in {r.domain.domain_id for r in records}
        # The point: it disappears rather than reporting an idle package.
        assert all(r.telemetry_data.energy_consumption_uj > 0 for r in records)

    @pytest.mark.asyncio
    async def test_is_url_reachable(self, powercap, tmp_path):
        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            assert await RAPLTelemetryCollector(powercap).is_url_reachable() is True
            assert (
                await RAPLTelemetryCollector(tmp_path / "absent").is_url_reachable()
                is False
            )


class TestProtocolConformance:
    def test_collector_satisfies_the_host_protocol_surface(self, powercap):
        """The full protocol surface, not a subset.

        The earlier version of this test listed five names and missed exactly
        the three lifecycle methods the collector did not implement, so it
        passed against a collector that did not satisfy the protocol. The
        protocol is runtime_checkable, so an instance check catches a missing
        method without hand-maintaining the list.
        """
        from aiperf.host_telemetry.protocols import HostTelemetryCollectorProtocol

        collector = RAPLTelemetryCollector(powercap)
        assert isinstance(collector, HostTelemetryCollectorProtocol)
        for name in (
            "id",
            "endpoint_url",
            "initialize",
            "start",
            "stop",
            "is_url_reachable",
            "collect_and_process_metrics",
            "validate_environment",
        ):
            assert hasattr(collector, name), name

    @pytest.mark.asyncio
    async def test_collect_and_process_dispatches_via_callback(self, powercap):
        """Records reach the callback with the collector id, errors do not raise."""

        received = []

        async def record_callback(records, collector_id):
            received.append((records, collector_id))

        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            collector = RAPLTelemetryCollector(
                powercap, record_callback=record_callback
            )
            await collector.initialize()
            await collector.collect_and_process_metrics()

        assert len(received) == 1
        records, collector_id = received[0]
        assert collector_id == "rapl"
        assert all(r.telemetry_data.energy_consumption_uj >= 0 for r in records)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_runs_the_background_loop_and_stop_halts_it(self, powercap):
        """initialize -> start -> a record arrives via the loop -> stop.

        The conformance isinstance check only proves the lifecycle methods
        exist; this is the test that proves they work, which is exactly the
        gap that let an unstartable collector pass the previous suite.
        """
        received = asyncio.Event()
        deliveries = []

        async def record_callback(records, collector_id):
            deliveries.append((records, collector_id))
            received.set()

        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            c = RAPLTelemetryCollector(
                powercap, collection_interval=0.02, record_callback=record_callback
            )
            await c.initialize()
            await c.start()
            try:
                await asyncio.wait_for(received.wait(), timeout=5.0)
            finally:
                await c.stop()

        records, collector_id = deliveries[0]
        assert collector_id == "rapl"
        assert records and all(
            r.telemetry_data.energy_consumption_uj >= 0 for r in records
        )


class TestResetVersusWrap:
    def test_reset_far_from_range_is_absorbed_not_credited_as_wrap(self, powercap):
        """A counter reset at 10% of range must not inject a full range."""
        d = discover_domains(powercap)[0]
        assert d.max_energy_uj == 262143328850.0
        (d.path / "energy_uj").write_text("26214332885")   # 10% of range
        first = d.read_energy_uj()
        (d.path / "energy_uj").write_text("1000")           # backwards, far from ceiling
        second = d.read_energy_uj()
        assert second == first                              # absorbed, nondecreasing
        (d.path / "energy_uj").write_text("2000")
        assert d.read_energy_uj() == first + 1000           # counting resumes

    def test_backwards_step_near_range_ceiling_is_still_a_wrap(self, powercap):
        d = discover_domains(powercap)[0]
        near_top = int(d.max_energy_uj * 0.9)
        (d.path / "energy_uj").write_text(str(near_top))
        d.read_energy_uj()
        (d.path / "energy_uj").write_text("1000")
        assert d.read_energy_uj() == 1000 + d.max_energy_uj


class TestFailureVisibility:
    @pytest.mark.asyncio
    async def test_all_domains_unreadable_dispatches_an_error(self, powercap):
        """Total telemetry loss must not look like a healthy idle run."""
        errors = []

        async def error_callback(details, collector_id):
            errors.append((details, collector_id))

        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            c = RAPLTelemetryCollector(powercap, error_callback=error_callback)
            await c.initialize()
            for d in c.domains:
                (d.path / "energy_uj").unlink()
            await c.collect_and_process_metrics()

        assert len(errors) == 1
        assert "no sample" in str(errors[0][0])

    @pytest.mark.asyncio
    async def test_one_invalid_domain_does_not_poison_the_others(self, powercap):
        """A negative reading fails model validation; the rest still deliver."""
        received = []

        async def record_callback(records, collector_id):
            received.append(records)

        with patch("aiperf.host_telemetry.rapl_collector.IS_LINUX", True):
            c = RAPLTelemetryCollector(powercap, record_callback=record_callback)
            await c.initialize()
            assert len(c.domains) >= 2
            (c.domains[0].path / "energy_uj").write_text("-5")
            await c.collect_and_process_metrics()

        assert len(received) == 1
        ids = [r.domain.domain_id for r in received[0]]
        assert c.domains[0].domain_id not in ids
        assert len(ids) == len(c.domains) - 1


class TestDomainOrdering:
    def test_double_digit_domains_sort_numerically(self, tmp_path):
        for i in (0, 1, 2, 10, 11):
            d = tmp_path / f"intel-rapl:{i}"
            d.mkdir()
            (d / "name").write_text(f"package-{i}")
            (d / "energy_uj").write_text("1")
            (d / "max_energy_range_uj").write_text("262143328850")
        order = [(d.index, d.domain_id) for d in discover_domains(tmp_path)]
        assert order == [
            (0, "intel-rapl:0"), (1, "intel-rapl:1"), (2, "intel-rapl:2"),
            (3, "intel-rapl:10"), (4, "intel-rapl:11"),
        ]
