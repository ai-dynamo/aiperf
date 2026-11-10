# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The event-bus proxy container never holds presigned dataset URLs."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import pytest

from aiperf.cli_commands.proxy import proxy
from aiperf.common.endpoint_credentials import AIPERF_INJECTED_DATASET_URLS
from aiperf.config import BenchmarkConfig, BenchmarkRun

_RAW_DATASET_URL = "https://bucket.s3.amazonaws.com/t.jsonl?X-Amz-Signature=topsecret"


def _write_run(path: Path) -> Path:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://server/v1"], "type": "chat"},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {"name": "profiling", "type": "concurrency", "duration": 1, "concurrency": 1}
            ],
        }
    )  # fmt: skip
    run = BenchmarkRun(benchmark_id="job", cfg=cfg, artifact_dir=path.parent)
    path.write_bytes(orjson.dumps(run.model_dump(mode="json", exclude_none=True)))
    return path


def test_proxy_drops_injected_dataset_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        AIPERF_INJECTED_DATASET_URLS, orjson.dumps([_RAW_DATASET_URL]).decode()
    )
    with (
        patch("aiperf.cli_commands.proxy._run_proxy", new=MagicMock()) as run_proxy,
        patch("aiperf.cli_commands.proxy.asyncio.run") as asyncio_run,
    ):
        proxy(benchmark_run_file=_write_run(tmp_path / "run.json"))

    assert os.getenv(AIPERF_INJECTED_DATASET_URLS) is None
    asyncio_run.assert_called_once_with(run_proxy.return_value)


def test_proxy_drops_injected_dataset_urls_even_when_startup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        AIPERF_INJECTED_DATASET_URLS, orjson.dumps([_RAW_DATASET_URL]).decode()
    )
    with pytest.raises(SystemExit):
        proxy(benchmark_run_file=_write_run(tmp_path / "run.json"), kind="bogus")

    assert os.getenv(AIPERF_INJECTED_DATASET_URLS) is None
