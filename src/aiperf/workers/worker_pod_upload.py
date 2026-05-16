# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw-record upload helpers for the WorkerGroupManager.

Extracted from ``worker_pod_manager`` to keep that module within the
ergonomics file-size limit. These helpers run during shutdown after sibling
record-processor containers have flushed their raw JSONL files to the shared
results volume.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import aiohttp

from aiperf.common.enums import ExportLevel
from aiperf.common.environment import Environment
from aiperf.config.defaults import OutputDefaults
from aiperf.transports.aiohttp_client import create_tcp_connector

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class _UploadLogger(Protocol):
    """Structural protocol matching logging methods used on BaseComponentService."""

    def info(self, msg: str) -> None: ...
    def debug(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...


async def wait_for_raw_record_files(
    run: BenchmarkRun,
    record_processors_per_pod: int,
    logger: _UploadLogger,
) -> None:
    """Wait for sibling record-processor containers to flush raw files locally."""
    cfg = run.cfg
    if cfg.output.export_level != ExportLevel.RAW:
        return

    raw_records_dir = cfg.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
    expected_files = max(1, record_processors_per_pod)
    deadline = (
        asyncio.get_running_loop().time()
        + Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
    )
    last_snapshot: tuple[tuple[str, int], ...] | None = None
    stable_reads = 0

    while asyncio.get_running_loop().time() < deadline:
        files = (
            sorted(raw_records_dir.glob("raw_records_*.jsonl"))
            if raw_records_dir.exists()
            else []
        )
        snapshot = tuple((path.name, path.stat().st_size) for path in files)
        if len(files) >= expected_files and snapshot == last_snapshot:
            stable_reads += 1
            if stable_reads >= 2:
                return
        else:
            stable_reads = 0
        last_snapshot = snapshot
        await asyncio.sleep(0.5)

    actual_files = len(last_snapshot or ())
    logger.warning(
        "Timed out waiting for raw record files to stabilize before upload: "
        f"expected at least {expected_files}, found {actual_files}"
    )


async def upload_raw_records(run: BenchmarkRun, logger: _UploadLogger) -> None:
    """Upload raw record files to the controller API for aggregation."""
    cfg = run.cfg
    if cfg.output.export_level != ExportLevel.RAW:
        return

    raw_records_dir = cfg.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
    if not raw_records_dir.exists():
        logger.debug("No raw_records directory found, skipping upload")
        return

    raw_files = list(raw_records_dir.glob("raw_records_*.jsonl"))
    if not raw_files:
        logger.debug("No raw record files found, skipping upload")
        return

    upload_base_url = _get_upload_base_url(run)
    if not upload_base_url:
        logger.warning("Cannot determine controller API URL for raw record upload")
        return

    logger.info(f"Uploading {len(raw_files)} raw record file(s) to controller API")
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(connector=connector) as session:
        for file_path in raw_files:
            await _upload_file(session, upload_base_url, file_path, logger)


def _get_upload_base_url(run: BenchmarkRun) -> str | None:
    """Derive the results upload URL from the dataset API URL."""
    base_url = run.cfg.runtime.dataset_api_base_url
    if not base_url:
        return None
    # dataset_api_base_url is http://{host}:{port}/api/dataset
    # We need http://{host}:{port}/api/results/upload
    api_base = base_url.rsplit("/api/dataset", 1)[0]
    return f"{api_base}/api/results/upload"


async def _upload_file(
    session: aiohttp.ClientSession,
    upload_base_url: str,
    file_path: Path,
    logger: _UploadLogger,
) -> None:
    """Upload a single raw record file to the controller API."""
    url = f"{upload_base_url}/{file_path.name}"
    try:
        file_size = file_path.stat().st_size
        file_bytes = await asyncio.to_thread(file_path.read_bytes)
        data = aiohttp.FormData()
        data.add_field(
            "file",
            file_bytes,
            filename=file_path.name,
            content_type="application/x-ndjson",
        )
        async with session.post(
            url, data=data, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status == 201:
                logger.info(
                    f"Uploaded raw record file: {file_path.name} ({file_size:,} bytes)"
                )
            else:
                body = await resp.text()
                logger.warning(
                    f"Failed to upload {file_path.name}: HTTP {resp.status} - {body}"
                )
    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.warning(f"Error uploading {file_path.name}: {e!r}")
