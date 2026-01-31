# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for visualization tests.

This file contains fixtures that are automatically discovered by pytest
and made available to test functions in the same directory and subdirectories.
"""

import logging
from pathlib import Path
from typing import Any

import pytest

from aiperf.plot.core.mode_detector import ModeDetector

logging.getLogger("choreographer").setLevel(logging.WARNING)
logging.getLogger("kaleido").setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def cleanup_logging_handlers():
    """Clean up logging handlers after each test to prevent ResourceWarnings.

    This fixture runs after each test and closes all handlers on the root logger,
    ensuring file handles are properly released.
    """
    yield
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)


# Path constants for static fixture data (shared across tests for speed)
FIXTURES_DIR = Path(__file__).parent / "fixtures"
QWEN_CONCURRENCY1_DIR = FIXTURES_DIR / "qwen_concurrency1"
QWEN_CONCURRENCY2_DIR = FIXTURES_DIR / "qwen_concurrency2"
QWEN_CONCURRENCY4_DIR = QWEN_CONCURRENCY2_DIR / "qwen_concurrency4"


@pytest.fixture
def single_run_dir() -> Path:
    """
    Return path to parent directory containing only a single run directory.

    Uses pre-existing fixture with real data.

    Returns:
        Path to parent directory containing one run.
    """
    return QWEN_CONCURRENCY1_DIR


@pytest.fixture
def multiple_run_dirs() -> list[Path]:
    """
    Return paths to real qwen fixture directories for multi-run testing.

    Uses pre-existing minimal fixture files for speed.
    All tests share these directories (read-only).

    Returns:
        List of 2 qwen run directory paths.
    """
    return [QWEN_CONCURRENCY1_DIR, QWEN_CONCURRENCY2_DIR]


@pytest.fixture
def parent_dir_with_runs(multiple_run_dirs: list[Path]) -> Path:
    """
    Get parent directory containing multiple run subdirectories.

    Args:
        multiple_run_dirs: List of run directories.

    Returns:
        Path to parent directory.
    """
    return multiple_run_dirs[0].parent


@pytest.fixture
def nested_run_dirs() -> Path:
    """
    Return path to nested run directories (run containing another run).

    Uses pre-existing fixture with real data.

    Returns:
        Path to parent directory containing nested runs.
    """
    return QWEN_CONCURRENCY2_DIR


@pytest.fixture
def mode_detector() -> ModeDetector:
    """
    Create a ModeDetector instance for testing.

    Returns:
        ModeDetector instance.
    """
    return ModeDetector()


@pytest.fixture
def sample_jsonl_data() -> list[dict[str, Any]]:
    """
    Generate sample JSONL data for testing.

    Returns:
        List of dictionaries representing JSONL records.
    """
    return [
        {
            "metadata": {
                "session_num": 0,
                "x_request_id": "req-1",
                "request_start_ns": 1000000000000,
                "request_ack_ns": 1000000100000,
                "request_end_ns": 1000001000000,
                "benchmark_phase": "profiling",
                "was_cancelled": False,
                "worker_id": "0",
                "record_processor_id": "0",
            },
            "metrics": {
                "time_to_first_token": {"value": 45.5, "unit": "ms"},
                "inter_token_latency": {"value": 18.2, "unit": "ms"},
                "request_latency": {"value": 900.0, "unit": "ms"},
                "output_sequence_length": {"value": 100, "unit": "tokens"},
                "input_sequence_length": {"value": 50, "unit": "tokens"},
            },
            "error": None,
        },
    ]


@pytest.fixture
def sample_aggregated_data() -> dict[str, Any]:
    """
    Generate sample aggregated JSON data for testing.

    Returns:
        Dictionary with aggregated data structure.
    """
    return {
        "input_config": {
            "endpoint": {
                "model_names": ["test-model"],
                "type": "chat",
                "streaming": True,
            },
            "loadgen": {
                "concurrency": 4,
                "request_count": 100,
            },
        },
        "was_cancelled": False,
        "error_summary": [],
    }


@pytest.fixture(autouse=True)
def mock_kaleido_write_image(request, monkeypatch):
    """Mock Plotly's write_image to avoid slow Kaleido rendering in unit tests only.

    This fixture automatically patches fig.write_image() and fig.to_image()
    to avoid the expensive Kaleido/Chrome rendering process during unit tests.
    Integration tests are skipped to allow real PNG generation.
    """

    def mock_write_image(self, *args, **kwargs):
        """Mock write_image that creates an empty file."""
        # Extract the file path from args or kwargs
        if args:
            path = Path(args[0])
        else:
            path = Path(kwargs.get("file", kwargs.get("path", "/tmp/mock.png")))

        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write a minimal PNG header to make it a valid file
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    def mock_to_image(self, *args, **kwargs):
        """Mock to_image that returns minimal PNG bytes."""
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    # Patch the methods on the Figure class
    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", mock_write_image)
    monkeypatch.setattr("plotly.graph_objects.Figure.to_image", mock_to_image)
