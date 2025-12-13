#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quick test script to verify orjson migration works."""

import tempfile
from pathlib import Path

import orjson


def test_orjson_basic():
    """Test basic orjson functionality."""
    print("Testing basic orjson functionality...")

    # Test dumps/loads
    data = {"test": "value", "number": 42, "nested": {"key": "val"}}
    encoded = orjson.dumps(data)
    print(f"  Encoded (bytes): {type(encoded)} = {encoded}")

    decoded = orjson.loads(encoded)
    print(f"  Decoded: {decoded}")
    assert decoded == data

    # Test file writing/reading
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".json") as f:
        f.write(orjson.dumps(data))
        temp_path = Path(f.name)

    print(f"  Created temp file: {temp_path}")

    # Read back with binary mode
    with open(temp_path, "rb") as f:
        loaded = orjson.loads(f.read())

    print(f"  Loaded from file: {loaded}")
    assert loaded == data

    temp_path.unlink()
    print("  ✓ Basic orjson tests passed!")


def test_jsonl_encoding():
    """Test JSONL line-by-line encoding with orjson."""
    print("\nTesting JSONL encoding...")

    records = [
        {"id": 1, "value": "test1"},
        {"id": 2, "value": "test2"},
    ]

    # Write JSONL
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        for record in records:
            f.write(orjson.dumps(record).decode("utf-8") + "\n")
        temp_path = Path(f.name)

    print(f"  Created temp JSONL file: {temp_path}")

    # Read JSONL back
    loaded_records = []
    with open(temp_path) as f:
        for line in f:
            if line.strip():
                loaded_records.append(orjson.loads(line.encode("utf-8")))

    print(f"  Loaded {len(loaded_records)} records from JSONL")
    assert loaded_records == records

    temp_path.unlink()
    print("  ✓ JSONL encoding tests passed!")


if __name__ == "__main__":
    test_orjson_basic()
    test_jsonl_encoding()
    print("\n✅ All orjson migration tests passed!")
