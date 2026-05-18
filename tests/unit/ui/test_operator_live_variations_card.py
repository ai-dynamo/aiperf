# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for live variations card helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

LIVE_VARIATIONS_HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "live-variations-helpers.js"
)


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def test_archived_trials_contribute_final_metrics() -> None:
    script = f"""
        import {{ trialContributesMetrics }} from {LIVE_VARIATIONS_HELPERS_PATH.as_uri()!r};
        console.log(JSON.stringify({{
          archived: trialContributesMetrics('Archived'),
          completed: trialContributesMetrics('Completed'),
          running: trialContributesMetrics('Running'),
        }}));
    """

    assert json.loads(_run_node(script)) == {
        "archived": True,
        "completed": True,
        "running": False,
    }
