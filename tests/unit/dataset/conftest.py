# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for dataset manager testing.
"""

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest

import aiperf.endpoints  # noqa: F401  # Import to register endpoints
import aiperf.transports  # noqa: F401  # Import to register transports
from aiperf.common.models import Conversation
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli


def make_dynamo_record(ts: int, sid: str, input_tokens: int, hashes: list[int]) -> dict:
    """Build one ``dynamo.request.trace.v1`` request_end record."""
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": {"session_id": sid},
        "request": {
            "request_id": f"r{ts}",
            "model": "m",
            "input_tokens": input_tokens,
            "output_tokens": 8,
            "cached_tokens": 0,
            "replay": {
                "trace_block_size": 16,
                "input_length": input_tokens,
                "input_sequence_hashes": hashes,
            },
        },
    }


def write_dynamo_trace(path: Path, records: list[dict]) -> Path:
    """Write ``records`` as a newline-delimited dynamo trace file and return the path."""
    path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return path


def write_shared_dynamo_trace(path: Path) -> Path:
    """Write the canonical 3-record dynamo fixture: two ``s1`` turns sharing a hash prefix plus a standalone ``s2`` session."""
    return write_dynamo_trace(
        path,
        [
            make_dynamo_record(1000, "s1", 32, [111, 222]),
            make_dynamo_record(2000, "s1", 64, [111, 222, 333, 444]),
            make_dynamo_record(3000, "s2", 48, [555, 666, 777]),
        ],
    )


def assert_store_dirs_identical(dir_a: Path, dir_b: Path, why: str = "") -> None:
    """Assert two unified-store directories hold the same file set with byte-identical contents."""
    files_a = sorted(p.name for p in dir_a.iterdir())
    files_b = sorted(p.name for p in dir_b.iterdir())
    assert files_a == files_b and files_a, (
        f"unified store file sets differ: {files_a} vs {files_b}"
    )
    for name in files_a:
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes(), (
            f"unified store file {name!r} differs{f' -- {why}' if why else ''}"
        )


@pytest.fixture
def cli_config(tmp_path: Path) -> CLIConfig:
    """Create a CLIConfig for testing."""
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        url="http://localhost:8000",
        artifact_directory=tmp_path,
    )


@pytest.fixture
def benchmark_run(cli_config: CLIConfig):
    """Build a v2 BenchmarkRun from the dataset-scoped cli_config fixture."""
    return make_run_from_cli(cli_config)


@pytest.fixture
def empty_dataset_manager(benchmark_run) -> DatasetManager:
    """Create a DatasetManager instance with empty dataset."""
    manager = DatasetManager(
        run=benchmark_run,
        service_id="test_dataset_manager",
    )
    manager.dataset = {}
    return manager


@pytest.fixture
def populated_dataset_manager(
    benchmark_run,
    sample_conversations: dict[str, Conversation],
) -> DatasetManager:
    """Create a DatasetManager instance with sample data."""
    manager = DatasetManager(
        run=benchmark_run,
        service_id="test_dataset_manager",
    )
    manager.dataset = sample_conversations
    return manager


@pytest.fixture
def capture_file_writes():
    """Provide a fixture to capture file write operations for testing purposes."""

    class FileWriteCapture:
        def __init__(self):
            self.written_content = ""

        def write_bytes(self, data: bytes):
            self.written_content = data.decode("utf-8")

    capture = FileWriteCapture()

    class _FakeAsyncFile:
        async def write(self, data):
            if isinstance(data, (bytes, bytearray)):
                capture.write_bytes(bytes(data))
            else:
                capture.written_content = data

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_aiofiles_open(*args, **kwargs):
        return _FakeAsyncFile()

    with patch("aiofiles.open", fake_aiofiles_open):
        yield capture


@pytest.fixture
def conversation_ids() -> list[str]:
    """Standard list of conversation IDs for sampler testing."""
    return ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
