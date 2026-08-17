# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared JSONL-fixture writers and digest helpers for the dynamo adapter tests."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any

import orjson


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write ``records`` as newline-terminated JSONL and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    return path


def write_jsonl_gz(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write ``records`` as gzipped newline-terminated JSONL and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    return path


def blake_digest(text: str) -> str:
    """Stable 16-byte blake2b hex digest used to pin oracle content."""
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
