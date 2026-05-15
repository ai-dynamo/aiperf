# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for dataset manager testing.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import aiperf.endpoints  # noqa: F401  # Import to register endpoints
import aiperf.transports  # noqa: F401  # Import to register transports
from aiperf.common.models import Conversation
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli


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
