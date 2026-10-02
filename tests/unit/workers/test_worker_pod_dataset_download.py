# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``download_dataset`` retry budget and backoff cap."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.workers import worker_pod_dataset_download
from aiperf.workers.worker_pod_dataset_download import download_dataset


@pytest.fixture
def run() -> MagicMock:
    run = MagicMock()
    run.benchmark_id = "bench-1"
    run.cfg.runtime.dataset_api_base_url = "http://controller/api/dataset"
    return run


@pytest.fixture
def patched_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(
        worker_pod_dataset_download.aiohttp,
        "ClientSession",
        MagicMock(return_value=session_context),
    )
    monkeypatch.setattr(
        worker_pod_dataset_download, "create_tcp_connector", MagicMock()
    )


def _capture_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(worker_pod_dataset_download.asyncio, "sleep", fake_sleep)
    return delays


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_retries",
    [
        param(0, id="no_retries"),
        param(2, id="two_retries"),
        param(5, id="five_retries"),
    ],
)  # fmt: skip
async def test_download_dataset_exhausted_honours_configured_max_retries(
    run: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_session: None,
    max_retries: int,
) -> None:
    monkeypatch.setattr(
        worker_pod_dataset_download.Environment.DATASET, "MMAP_BASE_PATH", tmp_path
    )
    monkeypatch.setattr(
        worker_pod_dataset_download.Environment.DATASET,
        "DOWNLOAD_MAX_RETRIES",
        max_retries,
    )
    delays = _capture_sleeps(monkeypatch)
    download_file = AsyncMock(side_effect=RuntimeError("controller not listening"))

    with pytest.raises(RuntimeError, match=f"after {max_retries + 1} attempts"):
        await download_dataset(run, MagicMock(), download_file=download_file)

    assert len(delays) == max_retries
    # Each attempt fans out to both the data and index endpoints.
    assert download_file.await_count == 2 * (max_retries + 1)


@pytest.mark.asyncio
async def test_download_dataset_backoff_is_capped_at_max_backoff(
    run: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_session: None,
) -> None:
    monkeypatch.setattr(
        worker_pod_dataset_download.Environment.DATASET, "MMAP_BASE_PATH", tmp_path
    )
    monkeypatch.setattr(
        worker_pod_dataset_download.Environment.DATASET, "DOWNLOAD_MAX_RETRIES", 6
    )
    monkeypatch.setattr(
        worker_pod_dataset_download.Environment.DATASET, "DOWNLOAD_RETRY_DELAY", 1.0
    )
    monkeypatch.setattr(
        worker_pod_dataset_download.Environment.DATASET,
        "DOWNLOAD_MAX_BACKOFF_SECONDS",
        4.0,
    )
    delays = _capture_sleeps(monkeypatch)
    download_file = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="after 7 attempts"):
        await download_dataset(run, MagicMock(), download_file=download_file)

    assert delays == [1.0, 2.0, 4.0, 4.0, 4.0, 4.0]


@pytest.mark.asyncio
async def test_download_dataset_recovers_within_configured_budget(
    run: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_session: None,
) -> None:
    monkeypatch.setattr(
        worker_pod_dataset_download.Environment.DATASET, "MMAP_BASE_PATH", tmp_path
    )
    monkeypatch.setattr(
        worker_pod_dataset_download.Environment.DATASET, "DOWNLOAD_MAX_RETRIES", 4
    )
    delays = _capture_sleeps(monkeypatch)
    failures = {"count": 0}

    async def download_file(_session, url: str, dest_path: Path) -> None:
        if url.endswith("/index") and failures["count"] < 2:
            failures["count"] += 1
            raise RuntimeError("index not ready")
        dest_path.write_bytes(b"x" * (1024 if url.endswith("/data") else 256))

    data_path, index_path = await download_dataset(
        run, MagicMock(), download_file=download_file
    )

    assert len(delays) == 2
    assert data_path.stat().st_size == 1024
    assert index_path.stat().st_size == 256
