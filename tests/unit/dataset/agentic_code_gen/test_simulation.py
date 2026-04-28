# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for simulation dashboard JSONL loading and rendering."""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.dataset.agentic_code_gen.reporting.simulation import (
    load_sessions,
    render_simulation,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    jsonl = path / "data.jsonl"
    with jsonl.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")
    return jsonl


class TestLoadSessions:
    def test_groups_turns_and_preserves_hash_ids(self, tmp_path: Path) -> None:
        rows = [
            {
                "session_id": "s1",
                "input_length": 100,
                "output_length": 20,
                "timestamp": 0.0,
                "group_id": 7,
                "hash_ids": [10, 11, 12],
            },
            {
                "session_id": "s1",
                "input_length": 50,
                "output_length": 10,
                "delay": 5000,
                "hash_ids": [13, 14],
            },
            {
                "session_id": "s2",
                "input_length": 80,
                "output_length": 8,
                "timestamp": 0.0,
                "group_id": 2,
                "is_restart": True,
            },
        ]

        sessions = load_sessions(_write_jsonl(tmp_path, rows))

        assert [s["session_id"] for s in sessions] == ["s1", "s2"]
        assert sessions[0]["group_id"] == 7
        assert sessions[0]["turns"][0]["hash_ids"] == [10, 11, 12]
        assert sessions[0]["turns"][1]["hash_ids"] == [13, 14]
        assert sessions[0]["turns"][0]["cumulative_input_length"] == 100
        assert sessions[0]["turns"][1]["cumulative_input_length"] == 170
        assert sessions[1]["turns"][0]["hash_ids"] == []
        assert sessions[1]["is_restart"] is True


class TestRenderSimulation:
    def test_writes_html_with_inlined_data(self, tmp_path: Path) -> None:
        sessions = [
            {
                "session_id": "s1",
                "group_id": 0,
                "is_restart": False,
                "turns": [
                    {
                        "input_length": 100,
                        "output_length": 20,
                        "delay_ms": 0,
                        "hash_ids": [1, 2],
                        "cumulative_input_length": 100,
                    }
                ],
            }
        ]
        output = tmp_path / "simulation.html"

        render_simulation(sessions, output, block_size=64, l1_tokens=128, l1_5_tokens=0)

        html = output.read_text()
        assert "Simulation Dashboard" in html
        assert "const SESSIONS =" in html
        assert '"session_id":"s1"' in html
        assert 'value="64"' in html

    def test_escapes_inline_json_script_end_tags(self, tmp_path: Path) -> None:
        sessions = [
            {
                "session_id": "</script><div>",
                "group_id": 0,
                "is_restart": False,
                "turns": [],
            }
        ]
        output = tmp_path / "simulation.html"

        render_simulation(sessions, output)

        html = output.read_text()
        assert "<\\/script>" in html
        assert "</script><div>" not in html
