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
import orjson

from aiperf.common.enums import ExportLevel
from aiperf.config.defaults import OutputDefaults
from aiperf.plugin.enums import ServiceRunType
from aiperf.transports.aiohttp_client import create_tcp_connector

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class _UploadLogger(Protocol):
    """Structural protocol matching logging methods used on BaseComponentService."""

    def info(self, msg: str) -> None: ...
    def debug(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...


async def upload_raw_records(run: BenchmarkRun, logger: _UploadLogger) -> None:
    """Upload every materialized final RAW file or raise on any failure.

    Exact processor finalization and shutdown precede this function. An empty
    directory is therefore valid for an idle pod; file counts are not used as
    a proxy for processor completion.
    """
    cfg = run.cfg
    if cfg.artifacts.export_level != ExportLevel.RAW:
        return

    raw_records_dir = cfg.artifacts.dir / OutputDefaults.RAW_RECORDS_FOLDER
    if not raw_records_dir.exists():
        logger.debug("No raw_records directory found, skipping upload")
        return

    raw_files = sorted(raw_records_dir.glob("raw_records_*.jsonl"))
    if not raw_files:
        logger.debug("No raw record files found, skipping upload")
        return

    upload_base_url = _get_upload_base_url(run)
    if not upload_base_url:
        if cfg.runtime.service_run_type == ServiceRunType.KUBERNETES:
            raise RuntimeError(
                "Cannot determine controller API URL for Kubernetes RAW record upload"
            )
        logger.debug("No controller API URL configured; RAW files remain local")
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
            response_body = await resp.read()
            if resp.status != 201:
                raise RuntimeError(
                    f"HTTP {resp.status}: {response_body.decode(errors='replace')}"
                )
            payload = orjson.loads(response_body)
            uploaded_size = int(payload["size"])
            if uploaded_size != file_size:
                raise RuntimeError(
                    f"controller reported {uploaded_size} bytes, expected {file_size}"
                )
            logger.info(
                f"Uploaded raw record file: {file_path.name} ({file_size:,} bytes)"
            )
    except asyncio.CancelledError:
        raise
    except (TimeoutError, aiohttp.ClientError, OSError) as e:
        raise RuntimeError(f"Error uploading {file_path.name}: {e!r}") from e
    except (KeyError, TypeError, ValueError, orjson.JSONDecodeError) as e:
        raise RuntimeError(
            f"Invalid upload acknowledgement for {file_path.name}: {e!r}"
        ) from e
