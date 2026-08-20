# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RAW shard uploads must receive a complete controller acknowledgement."""

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from aiperf.common.enums import ExportLevel
from aiperf.plugin.enums import ServiceRunType
from aiperf.workers.worker_pod_upload import _upload_file, upload_raw_records


class _Response:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def read(self) -> bytes:
        return self._payload


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def post(self, *args, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_upload_file_rejects_http_failure(tmp_path) -> None:
    raw_file = tmp_path / "raw_records_0.jsonl"
    raw_file.write_bytes(b"{}\n")

    with (
        patch(
            "aiperf.workers.worker_pod_upload.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
        ),
        pytest.raises(RuntimeError, match="HTTP 500"),
    ):
        await _upload_file(
            _Session(_Response(500, b"failed")),
            "http://controller/api/results/upload",
            raw_file,
            MagicMock(),
        )


@pytest.mark.asyncio
async def test_upload_file_rejects_size_mismatch(tmp_path) -> None:
    raw_file = tmp_path / "raw_records_0.jsonl"
    raw_file.write_bytes(b"{}\n")
    acknowledgement = orjson.dumps({"filename": raw_file.name, "size": "1"})

    with (
        patch(
            "aiperf.workers.worker_pod_upload.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
        ),
        pytest.raises(RuntimeError, match="reported 1 bytes"),
    ):
        await _upload_file(
            _Session(_Response(201, acknowledgement)),
            "http://controller/api/results/upload",
            raw_file,
            MagicMock(),
        )


@pytest.mark.asyncio
async def test_kubernetes_upload_requires_controller_url(tmp_path) -> None:
    raw_dir = tmp_path / "raw_records"
    raw_dir.mkdir()
    (raw_dir / "raw_records_0.jsonl").write_bytes(b"{}\n")
    run = MagicMock()
    run.cfg.artifacts.export_level = ExportLevel.RAW
    run.cfg.artifacts.dir = tmp_path
    run.cfg.runtime.service_run_type = ServiceRunType.KUBERNETES
    run.cfg.runtime.dataset_api_base_url = None

    with pytest.raises(RuntimeError, match="Cannot determine controller API URL"):
        await upload_raw_records(run, MagicMock())


@pytest.mark.asyncio
async def test_no_materialized_files_is_valid_after_exact_finalize(tmp_path) -> None:
    run = MagicMock()
    run.cfg.artifacts.export_level = ExportLevel.RAW
    run.cfg.artifacts.dir = tmp_path

    await upload_raw_records(run, MagicMock())
