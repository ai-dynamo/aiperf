# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for `aiperf report weka-trace`."""

from __future__ import annotations

from pathlib import Path

import orjson

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "weka_traces_small"


def test_report_weka_trace_writes_three_html_files(tmp_path: Path) -> None:
    from aiperf.cli_commands.report import report_weka_trace

    report_weka_trace(
        path=FIXTURES_DIR,
        output=tmp_path,
    )

    run_dirs = list(tmp_path.glob("weka-report_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    for name in ("report.html", "cache_explorer.html", "simulation.html"):
        path = run_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"{name} is empty"

    cache_json = run_dir / "cache_structure.json"
    assert cache_json.exists() and cache_json.stat().st_size > 0
    assert orjson.loads(cache_json.read_bytes())["block_size"] == 64
