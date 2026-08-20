# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for job_spec_file.save_job_spec_file."""

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest

from aiperf.common.redact import REDACTED_VALUE
from aiperf.operator.job_spec_file import save_job_spec_file


@pytest.mark.asyncio
async def test_save_job_spec_file_writes_indented_json(tmp_path: Path) -> None:
    with patch("aiperf.operator.job_spec_file.OperatorEnvironment") as env:
        env.RESULTS.DIR = tmp_path
        spec = {"benchmark": {"models": {"items": [{"name": "m"}]}}}
        await save_job_spec_file("ns", "job-1", spec, epoch="100")

    out = tmp_path / "ns" / "job-1" / "100" / "job_spec.json"
    assert out.exists()
    assert orjson.loads(out.read_bytes()) == spec


@pytest.mark.asyncio
async def test_save_job_spec_file_redacts_public_endpoint_credentials(
    tmp_path: Path,
) -> None:
    spec = {
        "benchmark": {
            "endpoint": {
                "apiKey": "public-secret",
                "urls": ["https://user:pass@host/v1?token=query-secret&model=m"],
                "headers": {
                    "Authorization": "Bearer header-secret",
                    "X-Trace-ID": "trace-1",
                },
            }
        }
    }
    with patch("aiperf.operator.job_spec_file.OperatorEnvironment") as env:
        env.RESULTS.DIR = tmp_path
        await save_job_spec_file("ns", "job-1", spec, epoch="100")

    saved = orjson.loads(
        (tmp_path / "ns" / "job-1" / "100" / "job_spec.json").read_bytes()
    )
    endpoint = saved["benchmark"]["endpoint"]
    assert endpoint["apiKey"] == REDACTED_VALUE
    assert endpoint["headers"] == {
        "Authorization": REDACTED_VALUE,
        "X-Trace-ID": "trace-1",
    }
    assert endpoint["urls"] == [
        f"https://{REDACTED_VALUE}@host/v1?token={REDACTED_VALUE}&model=m"
    ]
    assert spec["benchmark"]["endpoint"]["apiKey"] == "public-secret"
