# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for dataset manager testing.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import aiperf.endpoints  # noqa: F401  # Import to register endpoints
import aiperf.transports  # noqa: F401  # Import to register transports
from aiperf.common.models import Conversation
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import EndpointType
from tests.harness.fake_communication import FakeCommunication  # noqa: F401


@pytest.fixture(autouse=True)
def _fast_corpus():
    """Replace expensive Shakespeare corpus tokenization with a cheap stub.

    PromptGenerator._initialize_corpus reads and tokenizes the entire Shakespeare
    text file (~486 chunks via ThreadPoolExecutor) on every construction. This
    takes ~150ms per call and is the dominant cost in dataset tests.
    """
    with patch(
        "aiperf.dataset.generator.prompt.PromptGenerator._initialize_corpus",
        lambda self: (
            setattr(self, "_tokenized_corpus", list(range(1000))),
            setattr(self, "_corpus_size", 1000),
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _skip_gc_in_tests():
    """Skip gc.collect() calls in dataset manager tests.

    DatasetManager._configure_dataset_client_and_free_memory calls gc.collect()
    twice to clean up circular references. In the test suite this is extremely
    expensive (~300ms) because it collects garbage from the entire test process,
    not just the current test's objects.
    """
    with patch("aiperf.dataset.dataset_manager.gc.collect"):
        yield


@pytest.fixture(autouse=True)
def _mock_control_client():
    """Mock the ZMQ DEALER control client for all dataset tests."""
    with patch(
        "aiperf.zmq.streaming_dealer_client.ZMQStreamingDealerClient",
        return_value=AsyncMock(),
    ):
        yield


@pytest.fixture
def dataset_config(tmp_path: Path) -> AIPerfConfig:
    """Create an AIPerfConfig for dataset testing."""
    return AIPerfConfig(
        models=["test-model"],
        endpoint={
            "urls": ["http://localhost:8000/v1/chat/completions"],
            "type": EndpointType.CHAT,
            "streaming": False,
        },
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
        artifacts={"dir": str(tmp_path)},
    )


@pytest.fixture
def dataset_run(dataset_config: AIPerfConfig) -> BenchmarkRun:
    """Create a BenchmarkRun wrapping the dataset config."""
    return BenchmarkRun(
        benchmark_id="test",
        cfg=dataset_config,
        artifact_dir=Path("/tmp/test"),
    )


@pytest.fixture
def empty_dataset_manager(
    dataset_run: BenchmarkRun,
) -> DatasetManager:
    """Create a DatasetManager instance with empty dataset."""
    manager = DatasetManager(
        run=dataset_run,
        service_id="test_dataset_manager",
    )
    manager.dataset = {}
    return manager


@pytest.fixture
def populated_dataset_manager(
    dataset_run: BenchmarkRun,
    sample_conversations: dict[str, Conversation],
) -> DatasetManager:
    """Create a DatasetManager instance with sample data."""
    manager = DatasetManager(
        run=dataset_run,
        service_id="test_dataset_manager",
    )
    manager.dataset = sample_conversations
    return manager


@pytest.fixture
def conversation_ids() -> list[str]:
    """Standard list of conversation IDs for sampler testing."""
    return ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


@pytest.fixture
def capture_file_writes():
    """Provide a fixture to capture file write operations for testing purposes."""

    class FileWriteCapture:
        def __init__(self):
            self.written_content = ""

        def write_bytes(self, data: bytes):
            self.written_content = data.decode("utf-8")

    capture = FileWriteCapture()

    def mock_write_bytes(self, data):
        capture.write_bytes(data)

    with patch("pathlib.Path.write_bytes", mock_write_bytes):
        yield capture
