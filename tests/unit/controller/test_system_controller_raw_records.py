# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`aiperf.controller.system_controller_raw_records`.

Focuses on:
- _should_wait_for_raw_records gating on ExportLevel.RAW
- _wait_for_raw_record_uploads happy path (all files arrive), early return
  for zero WGMs, partial-data and zero-data timeout warnings
- _shutdown_record_processors_and_wait_for_flush dispatches SHUTDOWN to each
  WGM, waits for stable file snapshots, and warns on timeout
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType, ExportLevel
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType

# ============================================================
# Helpers
# ============================================================


def _fake_monotonic_sequence(values: list[float]) -> Any:
    """Build a callable that returns a clock value per call, then sticks at the last."""

    it: Iterator[float] = iter(values)
    last = [values[-1]]

    def _now() -> float:
        try:
            v = next(it)
            last[0] = v
            return v
        except StopIteration:
            return last[0]

    return _now


def _make_wgm(service_id: str) -> MagicMock:
    """Stand-in ServiceRunInfo with the .service_id attribute the helper reads."""
    info = MagicMock()
    info.service_id = service_id
    return info


# ============================================================
# _should_wait_for_raw_records
# ============================================================


class TestShouldWaitForRawRecords:
    """Verify gating depends solely on cfg.output.export_level == RAW.

    `export_level` is a derived property of `ArtifactsConfig`; toggle it via
    the underlying `raw` / `records` fields rather than direct assignment.
    """

    def test_raw_level_returns_true(self, system_controller: SystemController) -> None:
        system_controller.run.cfg.output.raw = True
        assert system_controller.run.cfg.output.export_level == ExportLevel.RAW
        assert system_controller._should_wait_for_raw_records() is True

    def test_non_raw_level_returns_false(
        self, system_controller: SystemController
    ) -> None:
        # Default configuration has raw=False → RECORDS or SUMMARY level.
        system_controller.run.cfg.output.raw = False
        assert system_controller.run.cfg.output.export_level != ExportLevel.RAW
        assert system_controller._should_wait_for_raw_records() is False


# ============================================================
# _wait_for_raw_record_uploads
# ============================================================


class TestWaitForRawRecordUploads:
    """Verify polling semantics for raw-record API uploads."""

    async def test_zero_wgms_returns_immediately(
        self,
        system_controller: SystemController,
        tmp_path: Path,
    ) -> None:
        system_controller.run.cfg.output.dir = tmp_path

        with patch(
            "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
            return_value=[],
        ):
            await system_controller._wait_for_raw_record_uploads()
        # No assertion beyond "did not raise / hang"; this exercises the
        # early-exit branch.

    async def test_returns_when_all_files_present(
        self,
        system_controller: SystemController,
        tmp_path: Path,
    ) -> None:
        system_controller.run.cfg.output.dir = tmp_path
        raw_dir = tmp_path / "raw_records"
        raw_dir.mkdir()
        (raw_dir / "raw_records_a.jsonl").write_text("{}\n")
        (raw_dir / "raw_records_b.jsonl").write_text("{}\n")

        with patch(
            "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
            return_value=[_make_wgm("a"), _make_wgm("b")],
        ):
            await system_controller._wait_for_raw_record_uploads()

    async def test_timeout_with_partial_data_warns_partial(
        self,
        system_controller: SystemController,
        tmp_path: Path,
    ) -> None:
        system_controller.run.cfg.output.dir = tmp_path
        raw_dir = tmp_path / "raw_records"
        raw_dir.mkdir()
        (raw_dir / "raw_records_a.jsonl").write_text("{}\n")
        # Only 1 of 2 files; loop will hit deadline.

        # Sequence: deadline_setup, first while-check (0), final-check (deadline+1)
        clock = _fake_monotonic_sequence([0.0, 0.0, 1000.0])

        with (
            patch(
                "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
                return_value=[_make_wgm("a"), _make_wgm("b")],
            ),
            patch(
                "aiperf.controller.system_controller_raw_records.time.monotonic",
                side_effect=clock,
            ),
            patch.object(system_controller, "warning") as mock_warn,
        ):
            await system_controller._wait_for_raw_record_uploads()

        mock_warn.assert_called_once()
        assert "1/2" in mock_warn.call_args[0][0]

    async def test_timeout_with_zero_data_warns_missing(
        self,
        system_controller: SystemController,
        tmp_path: Path,
    ) -> None:
        system_controller.run.cfg.output.dir = tmp_path
        # raw_dir does not exist at all.

        clock = _fake_monotonic_sequence([0.0, 0.0, 1000.0])

        with (
            patch(
                "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
                return_value=[_make_wgm("a")],
            ),
            patch(
                "aiperf.controller.system_controller_raw_records.time.monotonic",
                side_effect=clock,
            ),
            patch.object(system_controller, "warning") as mock_warn,
        ):
            await system_controller._wait_for_raw_record_uploads()

        mock_warn.assert_called_once()
        assert "missing" in mock_warn.call_args[0][0].lower()


# ============================================================
# _shutdown_record_processors_and_wait_for_flush
# ============================================================


class TestShutdownRecordProcessorsAndWaitForFlush:
    """Verify shutdown commands are sent and flush stability is awaited."""

    async def test_no_wgms_returns_immediately(
        self, system_controller: SystemController
    ) -> None:
        system_controller.control_router = MagicMock()
        system_controller.control_router.send_to = AsyncMock()

        with patch(
            "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
            return_value=[],
        ):
            await system_controller._shutdown_record_processors_and_wait_for_flush()

        system_controller.control_router.send_to.assert_not_called()

    async def test_sends_shutdown_to_each_wgm(
        self,
        system_controller: SystemController,
        tmp_path: Path,
    ) -> None:
        system_controller.run.cfg.output.dir = tmp_path
        raw_dir = tmp_path / "raw_records"
        raw_dir.mkdir()
        (raw_dir / "raw_records_a.jsonl").write_text("{}\n")
        (raw_dir / "raw_records_b.jsonl").write_text("{}\n")

        sent_to: list[tuple[str, Command]] = []

        async def _capture(identity: str, struct: Command) -> None:
            sent_to.append((identity, struct))

        system_controller.control_router = MagicMock()
        system_controller.control_router.send_to = AsyncMock(side_effect=_capture)

        with patch(
            "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
            return_value=[_make_wgm("a"), _make_wgm("b")],
        ):
            await system_controller._shutdown_record_processors_and_wait_for_flush()

        assert {ident for ident, _ in sent_to} == {"a", "b"}
        for _, cmd in sent_to:
            assert isinstance(cmd, Command)
            assert cmd.cmd == CommandType.SHUTDOWN
            # Each command should have a 32-char uuid hex cid.
            assert len(cmd.cid) == 32

    async def test_send_to_exception_swallowed_per_wgm(
        self,
        system_controller: SystemController,
        tmp_path: Path,
    ) -> None:
        system_controller.run.cfg.output.dir = tmp_path
        raw_dir = tmp_path / "raw_records"
        raw_dir.mkdir()
        (raw_dir / "raw_records_a.jsonl").write_text("{}\n")
        (raw_dir / "raw_records_b.jsonl").write_text("{}\n")

        async def _send(identity: str, _struct: Command) -> None:
            if identity == "a":
                raise RuntimeError("zmq closed")

        system_controller.control_router = MagicMock()
        system_controller.control_router.send_to = AsyncMock(side_effect=_send)

        with patch(
            "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
            return_value=[_make_wgm("a"), _make_wgm("b")],
        ):
            # Should not raise — exception per-WGM is best-effort.
            await system_controller._shutdown_record_processors_and_wait_for_flush()

    async def test_returns_when_snapshot_stable_across_polls(
        self,
        system_controller: SystemController,
        tmp_path: Path,
    ) -> None:
        """Two identical (count, total_size) snapshots in a row → return."""
        system_controller.run.cfg.output.dir = tmp_path
        raw_dir = tmp_path / "raw_records"
        raw_dir.mkdir()
        (raw_dir / "raw_records_a.jsonl").write_text("hello")
        (raw_dir / "raw_records_b.jsonl").write_text("world")

        system_controller.control_router = MagicMock()
        system_controller.control_router.send_to = AsyncMock()

        with patch(
            "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
            return_value=[_make_wgm("a"), _make_wgm("b")],
        ):
            await system_controller._shutdown_record_processors_and_wait_for_flush()
        # If the helper had not returned on a stable snapshot it would block on
        # asyncio.sleep until the deadline; the auto-sleep fixture and tight
        # monotonic budget would cause a real test timeout.

    async def test_timeout_emits_warning(
        self,
        system_controller: SystemController,
        tmp_path: Path,
    ) -> None:
        """No files ever arrive → deadline expires → controller.warning() called."""
        system_controller.run.cfg.output.dir = tmp_path

        system_controller.control_router = MagicMock()
        system_controller.control_router.send_to = AsyncMock()

        # Sequence: deadline_setup, first while-check pass, second pass past deadline
        clock = _fake_monotonic_sequence([0.0, 0.0, 1000.0])

        with (
            patch(
                "aiperf.controller.system_controller_raw_records.ServiceRegistry.get_services",
                return_value=[_make_wgm("a")],
            ),
            patch(
                "aiperf.controller.system_controller_raw_records.time.monotonic",
                side_effect=clock,
            ),
            patch.object(system_controller, "warning") as mock_warn,
        ):
            await system_controller._shutdown_record_processors_and_wait_for_flush()

        mock_warn.assert_called_once()
        assert "Timed out" in mock_warn.call_args[0][0]


# ============================================================
# Sanity: ServiceRegistry isolation between tests
# ============================================================


class TestRegistryIsolation:
    """Ensure singletons-reset autofixture really does clear ServiceRegistry."""

    def test_registry_starts_empty_for_each_test(self) -> None:
        # If singletons reset is broken, this would fail randomly.
        assert ServiceRegistry.get_services(ServiceType.WORKER_GROUP_MANAGER) == []
