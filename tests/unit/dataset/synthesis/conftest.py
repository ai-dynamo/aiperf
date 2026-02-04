# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for synthesis tests."""

import tempfile
from pathlib import Path

import orjson
import pytest


@pytest.fixture
def sample_trace_data() -> list[dict]:
    """Provide sample trace data for testing.

    Note: input_length must be >= len(hash_ids) * block_size (512) for consistency.
    """
    return [
        {"input_length": 1024, "output_length": 20, "timestamp": 0, "hash_ids": [1, 2]},
        {
            "input_length": 1536,
            "output_length": 30,
            "timestamp": 1000,
            "hash_ids": [1, 2, 3],
        },
        {
            "input_length": 1024,
            "output_length": 25,
            "timestamp": 2000,
            "hash_ids": [1, 2],
        },
        {
            "input_length": 1536,
            "output_length": 40,
            "timestamp": 3000,
            "hash_ids": [2, 3, 4],
        },
        {
            "input_length": 1024,
            "output_length": 22,
            "timestamp": 4000,
            "hash_ids": [1, 3],
        },
    ]


@pytest.fixture
def trace_file_simple(sample_trace_data: list[dict]) -> Path:
    """Create a temporary trace JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for trace in sample_trace_data:
            f.write(orjson.dumps(trace).decode() + "\n")
        filename = f.name

    yield Path(filename)

    # Cleanup
    Path(filename).unlink(missing_ok=True)


@pytest.fixture
def sample_trace_without_hashes() -> list[dict]:
    """Provide sample trace data without hash IDs."""
    return [
        {"input_length": 100, "output_length": 20},
        {"input_length": 150, "output_length": 30},
        {"input_length": 120, "output_length": 25},
    ]


@pytest.fixture
def large_sample_trace_data() -> list[dict]:
    """Provide larger sample trace data for distribution testing.

    Note: input_length must be >= len(hash_ids) * block_size (512) for consistency.
    """
    traces = []
    # hash_id_pools with lengths 2, 3, 3, 2, 3 -> need 1024, 1536, 1536, 1024, 1536
    hash_id_pools = [[1, 2], [1, 2, 3], [2, 3, 4], [1, 3], [3, 4, 5]]
    input_length_pools = [1024, 1536, 1536, 1024, 1536]

    for i in range(100):
        pool_idx = i % len(hash_id_pools)
        traces.append(
            {
                "input_length": input_length_pools[pool_idx] + (i % 50) * 10,
                "output_length": 20 + (i % 30) * 2,
                "timestamp": i * 1000,
                "hash_ids": hash_id_pools[pool_idx],
            }
        )

    return traces
