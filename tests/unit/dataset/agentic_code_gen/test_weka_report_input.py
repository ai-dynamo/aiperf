# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the weka -> ParsedTurn light reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.dataset.agentic_code_gen.reporting.weka_input import load_weka_as_parsed

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "weka_traces"


def test_single_file_parent_normals_become_one_session() -> None:
    parsed = load_weka_as_parsed(FIXTURES / "simple.json")

    assert list(parsed.keys()) == ["trace_simple"]
    turns = parsed["trace_simple"]
    assert len(turns) == 2

    assert turns[0].session_id == "trace_simple"
    assert turns[0].input_length == 200
    assert turns[0].output_length == 30
    assert turns[0].hash_ids == [1, 2, 3]
    assert turns[0].delay_ms == 0.0
    assert turns[0].group_id is None
    assert turns[0].is_restart is False

    assert turns[1].input_length == 250
    assert turns[1].output_length == 40
    assert turns[1].hash_ids == [1, 2, 3, 4]
    # delay = (5.0 - 0.0) * 1000.0
    assert turns[1].delay_ms == pytest.approx(5000.0)


def test_directory_yields_one_session_per_trace() -> None:
    parsed = load_weka_as_parsed(
        Path(__file__).resolve().parents[3] / "fixtures" / "weka_traces_small"
    )
    # 10 trace files in this fixture dir.
    assert len(parsed) == 10
    # Insertion order must match sorted(glob("*.json")) — pin against the
    # explicit fixture so a regression that drops the sort or returns the
    # wrong subset is caught.
    expected_ids = [f"trace_{i:02d}_n{i}" for i in range(1, 11)]
    assert list(parsed.keys()) == expected_ids


def test_duplicate_trace_id_raises(tmp_path: Path) -> None:
    """Two files with the same trace.id in one dir is an error."""
    blob = (FIXTURES / "simple.json").read_bytes()
    (tmp_path / "a.json").write_bytes(blob)
    (tmp_path / "b.json").write_bytes(blob)

    with pytest.raises(ValueError, match="Duplicate trace id 'trace_simple'"):
        load_weka_as_parsed(tmp_path)
