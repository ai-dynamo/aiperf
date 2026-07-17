# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for graph component-integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.environment import Environment


@pytest.fixture
def mmap_base_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect MMAP_BASE_PATH to tmp_path so stores land in a known dir."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
    return tmp_path
