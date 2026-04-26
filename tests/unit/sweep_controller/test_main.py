# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.sweep_controller.main import (
    AGGREGATE_READY_MARKER,
    aggregate_marker_exists,
    write_aggregate_marker,
)


def test_aggregate_marker_lifecycle(tmp_path: Path):
    base = tmp_path / "results"
    base.mkdir()
    assert aggregate_marker_exists(base) is False
    write_aggregate_marker(base)
    assert aggregate_marker_exists(base) is True
    assert (base / AGGREGATE_READY_MARKER).exists()


def test_aggregate_marker_atomic_rename(tmp_path: Path):
    """Marker is written via .tmp + rename; partial writes don't appear ready."""
    base = tmp_path
    write_aggregate_marker(base)
    assert (base / AGGREGATE_READY_MARKER).exists()
    # No leftover .tmp
    assert not (base / (AGGREGATE_READY_MARKER + ".tmp")).exists()
