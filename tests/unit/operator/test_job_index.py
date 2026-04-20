# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the job index module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest

from aiperf.operator.job_index import (
    INDEX_FILENAME,
    _job_key,
    _read_index,
    _write_index,
    get_index,
    get_job_spec,
    index_job_completed,
    index_job_created,
    index_job_failed,
    save_job_spec_file,
)


@pytest.fixture()
def results_dir(tmp_path: Path):
    """Patch RESULTS.DIR to use a temporary directory."""
    with patch("aiperf.operator.job_index.OperatorEnvironment") as mock_env:
        mock_env.RESULTS.DIR = tmp_path
        yield tmp_path


class TestJobKey:
    def test_basic_key(self):
        assert _job_key("default", "my-job") == "default/my-job"

    def test_preserves_case(self):
        assert _job_key("Ns", "Job-ID") == "Ns/Job-ID"


class TestReadWriteIndex:
    def test_read_empty(self, results_dir: Path):
        assert _read_index() == {}

    def test_write_and_read(self, results_dir: Path):
        data = {"default/job-1": {"namespace": "default", "job_id": "job-1"}}
        _write_index(data)
        assert _read_index() == data

    def test_corrupt_file_returns_empty(self, results_dir: Path):
        path = results_dir / INDEX_FILENAME
        path.write_text("not valid json{{{")
        assert _read_index() == {}

    def test_atomic_write(self, results_dir: Path):
        _write_index({"a": 1})
        # No .tmp file should remain
        assert not (results_dir / f"{INDEX_FILENAME}.tmp").exists()
        assert (results_dir / INDEX_FILENAME).exists()


class TestIndexJobCreated:
    @pytest.mark.asyncio()
    async def test_creates_entry(self, results_dir: Path):
        spec = {
            "benchmark": {
                "endpoint": {"urls": ["http://server:8000/v1"], "type": "chat"},
                "models": {"items": [{"name": "llama-70b"}]},
            },
            "image": "aiperf:local",
        }
        await index_job_created("default", "bench-1", spec)

        index = _read_index()
        assert "default/bench-1" in index
        entry = index["default/bench-1"]
        assert entry["model"] == "llama-70b"
        assert entry["endpoint"] == "http://server:8000/v1"
        assert entry["phase"] == "Pending"
        assert entry["spec"] == spec
        assert entry["start_time"] is not None

    @pytest.mark.asyncio()
    async def test_handles_flat_model_names(self, results_dir: Path):
        spec = {
            "benchmark": {
                "models": {"modelNames": ["gpt-4"]},
                "endpoint": {"url": "http://api:8080"},
            }
        }
        await index_job_created("ns", "j1", spec)
        entry = _read_index()["ns/j1"]
        assert entry["model"] == "gpt-4"

    @pytest.mark.asyncio()
    async def test_multiple_jobs(self, results_dir: Path):
        await index_job_created("ns", "j1", {"benchmark": {}})
        await index_job_created("ns", "j2", {"benchmark": {}})
        index = _read_index()
        assert len(index) == 2


class TestIndexJobCompleted:
    @pytest.mark.asyncio()
    async def test_updates_existing_entry(self, results_dir: Path):
        await index_job_created("ns", "j1", {"benchmark": {}})
        await index_job_completed(
            "ns",
            "j1",
            phase="Completed",
            metrics={
                "request_throughput": {"avg": 1234.5},
                "request_latency": {"p99": 456.7},
            },
            downloaded_files=["a.json", "b.csv"],
        )
        entry = _read_index()["ns/j1"]
        assert entry["phase"] == "Completed"
        assert entry["throughput_rps"] == 1234.5
        assert entry["latency_p99_ms"] == 456.7
        assert entry["file_count"] == 2
        assert entry["end_time"] is not None

    @pytest.mark.asyncio()
    async def test_creates_entry_if_missing(self, results_dir: Path):
        await index_job_completed("ns", "orphan", phase="Completed")
        assert "ns/orphan" in _read_index()


class TestIndexJobFailed:
    @pytest.mark.asyncio()
    async def test_records_error(self, results_dir: Path):
        await index_job_created("ns", "j1", {"benchmark": {}})
        await index_job_failed("ns", "j1", "OOMKilled")
        entry = _read_index()["ns/j1"]
        assert entry["phase"] == "Failed"
        assert entry["error"] == "OOMKilled"


class TestGetIndex:
    @pytest.mark.asyncio()
    async def test_returns_full_index(self, results_dir: Path):
        await index_job_created("ns", "j1", {"benchmark": {}})
        index = await get_index()
        assert "ns/j1" in index


class TestGetJobSpec:
    @pytest.mark.asyncio()
    async def test_returns_spec(self, results_dir: Path):
        spec = {"benchmark": {"models": {"items": [{"name": "test"}]}}}
        await index_job_created("ns", "j1", spec)
        result = await get_job_spec("ns", "j1")
        assert result == spec

    @pytest.mark.asyncio()
    async def test_returns_none_for_missing(self, results_dir: Path):
        result = await get_job_spec("ns", "nonexistent")
        assert result is None


class TestSaveJobSpecFile:
    @pytest.mark.asyncio
    async def test_saves_json_file(self, results_dir: Path):
        spec = {"image": "aiperf:local", "benchmark": {"models": {}}}
        await save_job_spec_file("ns", "j1", spec)

        path = results_dir / "ns" / "j1" / "job_spec.json"
        assert path.exists()
        saved = orjson.loads(path.read_bytes())
        assert saved == spec

    @pytest.mark.asyncio
    async def test_creates_directories(self, results_dir: Path):
        await save_job_spec_file("deep/ns", "my-job", {"test": True})
        assert (results_dir / "deep/ns" / "my-job" / "job_spec.json").exists()
