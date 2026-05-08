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
