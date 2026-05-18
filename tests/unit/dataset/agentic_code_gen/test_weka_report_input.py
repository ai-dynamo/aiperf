# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the weka -> ParsedTurn light reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.dataset.agentic_code_gen.reporting.weka_input import (
    infer_weka_block_size,
    load_weka_as_parsed,
)

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


def test_subagent_becomes_separate_session() -> None:
    parsed = load_weka_as_parsed(FIXTURES / "one_subagent.json")

    # 1 parent + 1 subagent
    assert set(parsed.keys()) == {"trace_sa", "trace_sa::sa:agent_001"}

    parent = parsed["trace_sa"]
    # parent has two normals (the subagent entry between them is skipped)
    assert len(parent) == 2
    # delay between the two normals: (6.0 - 0.0) * 1000
    assert parent[0].delay_ms == 0.0
    assert parent[1].delay_ms == pytest.approx(6000.0)

    sub = parsed["trace_sa::sa:agent_001"]
    assert len(sub) == 1
    assert sub[0].input_length == 100
    assert sub[0].output_length == 50
    assert sub[0].hash_ids == [10, 11]
    assert sub[0].delay_ms == 0.0  # first turn of a session


def test_no_subagents_flag_omits_subagent_sessions() -> None:
    parsed = load_weka_as_parsed(
        FIXTURES / "one_subagent.json", include_subagents=False
    )
    assert set(parsed.keys()) == {"trace_sa"}


def test_max_context_length_drops_oversized_traces() -> None:
    # simple.json has peak input_length=250; cap below that drops it.
    parsed = load_weka_as_parsed(FIXTURES / "simple.json", max_context_length=100)
    assert parsed == {}

    # Cap above the peak keeps it.
    parsed = load_weka_as_parsed(FIXTURES / "simple.json", max_context_length=1000)
    assert "trace_simple" in parsed


def test_max_context_length_drops_subagents_with_parent() -> None:
    # one_subagent.json parent peak input_length=400; cap=100 drops parent
    # and its subagent.
    parsed = load_weka_as_parsed(FIXTURES / "one_subagent.json", max_context_length=100)
    assert parsed == {}


def test_parsed_to_sim_sessions_shape() -> None:
    from aiperf.dataset.agentic_code_gen.reporting.weka_input import (
        parsed_to_sim_sessions,
    )

    parsed = load_weka_as_parsed(FIXTURES / "simple.json")
    sim = parsed_to_sim_sessions(parsed)

    assert len(sim) == 1
    s = sim[0]
    assert s["session_id"] == "trace_simple"
    assert s["group_id"] == 0
    assert s["is_restart"] is False
    assert len(s["turns"]) == 2

    t0, t1 = s["turns"]
    assert t0["input_length"] == 200
    assert t0["output_length"] == 30
    assert t0["delay_ms"] == 0.0
    assert t0["hash_ids"] == [1, 2, 3]
    # cumulative_input_length = running sum of input + output prior to and
    # including the current input. Matches load_simulation_sessions's rule:
    # cumulative += input_length (before append), then cumulative += output_length.
    assert t0["cumulative_input_length"] == 200

    assert t1["input_length"] == 20
    assert t1["delay_ms"] == pytest.approx(5000.0)
    assert t1["cumulative_input_length"] == 250


def test_infer_weka_block_size_from_trace_files() -> None:
    assert infer_weka_block_size(FIXTURES / "simple.json") == 64
