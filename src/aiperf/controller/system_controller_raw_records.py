# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw-record flush / upload helpers for the SystemController."""

from __future__ import annotations

import asyncio
import time
import uuid

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.common.environment import Environment
from aiperf.common.service_registry import ServiceRegistry
from aiperf.config.defaults import OutputDefaults
from aiperf.plugin.enums import ServiceType


class SystemControllerRawRecordsMixin:
    """Raw-record flushing / upload wait helpers for :class:`SystemController`.

    These methods coordinate with WorkerGroupManager / RecordProcessor shutdown
    flushes so exporter threads never read a truncated ``raw_records_*.jsonl``
    file.
    """

    def _should_wait_for_raw_records(self) -> bool:
        """Check if we need to wait for raw record uploads from worker pods."""
        from aiperf.common.enums import ExportLevel

        return self.run.cfg.output.export_level == ExportLevel.RAW

    async def _wait_for_raw_record_uploads(self) -> None:
        """Wait for worker pods to upload raw record files to the API.

        Polls the raw_records subdirectory until we have at least one file
        per worker group manager, or the timeout expires.
        """
        raw_records_dir = (
            self.run.cfg.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        timeout = Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
        poll_interval = 1.0
        deadline = time.monotonic() + timeout

        wgm_count = len(ServiceRegistry.get_services(ServiceType.WORKER_GROUP_MANAGER))
        if wgm_count == 0:
            self.debug("No worker group managers registered, skipping raw record wait")
            return

        self.info(f"Waiting for raw record uploads from {wgm_count} worker group(s)...")

        while time.monotonic() < deadline:
            if raw_records_dir.exists():
                files = list(raw_records_dir.glob("raw_records_*.jsonl"))
                if len(files) >= wgm_count:
                    self.info(
                        f"Received {len(files)} raw record file(s) from "
                        f"{wgm_count} group(s), proceeding with export"
                    )
                    return
                if files:
                    self.debug(
                        f"Have {len(files)}/{wgm_count} raw record file(s), "
                        "waiting for remaining pods..."
                    )
            await asyncio.sleep(poll_interval)

        # Check what we got before warning
        actual = 0
        if raw_records_dir.exists():
            actual = len(list(raw_records_dir.glob("raw_records_*.jsonl")))
        if actual > 0:
            self.warning(
                f"Timed out after {timeout}s: received {actual}/{wgm_count} "
                "raw record file(s). Proceeding with partial data."
            )
        else:
            self.warning(
                f"Timed out waiting for raw record uploads after {timeout}s. "
                "Raw records may be missing from export."
            )

    async def _shutdown_record_processors_and_wait_for_flush(self) -> None:
        """Shut down WorkerGroupManager(s) and poll for flushed raw record files.

        In local multiprocessing mode, RPs are WGM subprocesses — outside the
        controller's service_manager — so stop_service(RECORD_PROCESSOR) on
        the controller is a no-op. Instead, send SHUTDOWN to each WGM over the
        control router; each WGM cascades shutdown to its child RPs, whose
        @on_stop hooks (BufferedJSONLWriterMixin._close_file) flush the
        raw_records_*.jsonl files before the aggregator reads them.
        """
        wgm_services = ServiceRegistry.get_services(ServiceType.WORKER_GROUP_MANAGER)
        if not wgm_services:
            return

        for svc in wgm_services:
            try:
                await self.control_router.send_to(
                    svc.service_id,
                    Command(cid=uuid.uuid4().hex, cmd=CommandType.SHUTDOWN),
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - shutdown is best-effort per WGM
                self.debug(f"Failed to send shutdown to {svc.service_id}: {e}")

        raw_records_dir = (
            self.run.cfg.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        expected = len(wgm_services)
        deadline = time.monotonic() + Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
        stable_snapshots: list[tuple[int, int]] = []
        # Wait until at least `expected` files exist, each with size > 0, and
        # the (count, total_size) tuple is unchanged across two consecutive
        # samples ~300ms apart. Mere existence is insufficient: RPs flush in
        # batches, so the first write of a partial batch can lead the
        # aggregator to read a truncated file.
        while time.monotonic() < deadline:
            if raw_records_dir.exists():
                files = list(raw_records_dir.glob("raw_records_*.jsonl"))
                if len(files) >= expected and all(f.stat().st_size > 0 for f in files):
                    snapshot = (len(files), sum(f.stat().st_size for f in files))
                    stable_snapshots.append(snapshot)
                    if (
                        len(stable_snapshots) >= 2
                        and stable_snapshots[-1] == stable_snapshots[-2]
                    ):
                        self.debug(
                            f"Raw record files stable at {raw_records_dir} "
                            f"(files={snapshot[0]}, bytes={snapshot[1]})"
                        )
                        return
                else:
                    stable_snapshots.clear()
            await asyncio.sleep(0.3)

        self.warning(
            f"Timed out waiting for record processors to flush raw records to "
            f"{raw_records_dir}; export may be incomplete."
        )
