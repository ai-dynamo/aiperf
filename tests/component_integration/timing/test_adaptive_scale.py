# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for adaptive scale timing mode."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.component_integration.timing.conftest import defaults
from tests.harness.utils import AIPerfCLI


def _load_jsonl(path: Path) -> list[dict]:
    return [
        orjson.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.component_integration
def test_adaptive_scale_profile_writes_controller_artifacts(cli: AIPerfCLI) -> None:
    """Exercise CLI -> config -> TimingConfig -> AdaptiveScaleStrategy wiring."""
    result = cli.run_sync(
        f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --concurrency 4 \
            --benchmark-duration 2.5 \
            --adaptive-scale \
            --adaptive-sustain-duration 1.0 \
            --adaptive-assessment-period 1.0 \
            --adaptive-scale-sla request_latency:p95:le:10000 \
            --osl 8 \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
        """,
        timeout=30.0,
    )

    assert result.exit_code == 0
    assert result.request_count > 0

    event_path = result.artifacts_dir / "adaptive_scale_events.jsonl"
    summary_path = result.artifacts_dir / "adaptive_scale_summary.json"
    assert event_path.exists()
    assert summary_path.exists()

    events = _load_jsonl(event_path)
    event_names = {event["event"] for event in events}
    assert "adaptive_phase_started" in event_names
    assert "adaptive_window" in event_names
    assert event_names & {
        "adaptive_complete",
        "adaptive_incomplete",
        "adaptive_failed",
        "boundary_discovered",
    }

    decisions = [event for event in events if event["event"] == "adaptive_decision"]
    assert decisions
    assert any(event["concurrency_after"] > 1 for event in decisions)

    summary = orjson.loads(summary_path.read_bytes())
    assert summary["control_variable"] == "concurrency"
