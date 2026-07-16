# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.watch_render_rich."""

from __future__ import annotations

from datetime import datetime

from rich.console import Console

from aiperf.kubernetes.watch_models import WatchSnapshot, WorkersSnapshot
from aiperf.kubernetes.watch_render_rich import RichRenderer


def _snapshot(**overrides) -> WatchSnapshot:
    """Build a WatchSnapshot with sensible test defaults."""
    base = {
        "timestamp": datetime(2026, 1, 1, 12, 0, 0),
        "job_id": "latency-sweep",
        "namespace": "aiperf-bench",
        "phase": "Succeeded",
        "target_kind": "AIPerfSweep",
        "elapsed_seconds": 5.0,
    }
    base.update(overrides)
    return WatchSnapshot(**base)


def test_sweep_snapshot_renders_run_progress_instead_of_job_workers() -> None:
    renderer = RichRenderer()
    console = Console(record=True, force_terminal=False, width=120)
    renderer._console = console

    renderer.render(
        _snapshot(
            sweep_runs_completed=3,
            sweep_runs_failed=1,
            sweep_runs_cancelled=1,
            sweep_runs_total=5,
            workers=WorkersSnapshot(ready=0, total=0),
        )
    )

    output = console.export_text()
    assert "Sweep runs" in output
    assert "5/5 done" in output
    assert "3 succeeded" in output
    assert "1 failed" in output
    assert "1 cancelled" in output
    assert "Workers: 0/0" not in output
    assert "Waiting for progress data" not in output
    assert "No metrics available yet" not in output
