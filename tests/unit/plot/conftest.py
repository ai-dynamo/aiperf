# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
