# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for integration tests."""

from pathlib import Path

import orjson


def create_rankings_dataset(tmp_path: Path, num_entries: int) -> Path:
    """Create a rankings dataset for testing.

    Args:
        tmp_path: Temporary directory path
        num_entries: Number of entries to create in the dataset

    Returns:
        Path to the created dataset file
    """
    dataset_path = tmp_path / "rankings.jsonl"
    with open(dataset_path, "w") as f:
        for i in range(num_entries):
            entry = {
                "texts": [
                    {"name": "query", "contents": [f"What is AI topic {i}?"]},
                    {"name": "passages", "contents": [f"AI passage {i}"]},
                ]
            }
            f.write(orjson.dumps(entry).decode("utf-8") + "\n")
    return dataset_path
