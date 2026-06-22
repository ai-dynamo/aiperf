# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for adaptive scale timing mode."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from aiperf_mock_server.config import MockServerConfig

from aiperf.timing.strategies.adaptive_scale import AdaptiveScaleStrategy
from tests.component_integration.timing.conftest import defaults
from tests.harness.fake_transport import FakeTransport
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
    assert result.json.was_cancelled is False

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


@pytest.mark.component_integration
def test_adaptive_scale_profile_discovers_sla_boundary(
    cli: AIPerfCLI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise SLA-margin ramp-up and boundary sustain under load."""

    def deterministic_sla_values(
        self: AdaptiveScaleStrategy, stats: object
    ) -> dict[str, float]:
        del stats
        key = self._sla_key(self._primary_sla)
        value = 20.0 if self._current_concurrency <= 11 else 120.0
        return {key: value}

    def deterministic_passes_sla(
        self: AdaptiveScaleStrategy, observed: dict[str, float]
    ) -> bool:
        return observed[self._sla_key(self._primary_sla)] <= 80.0

    monkeypatch.setattr(AdaptiveScaleStrategy, "_sla_values", deterministic_sla_values)
    monkeypatch.setattr(AdaptiveScaleStrategy, "_passes_sla", deterministic_passes_sla)

    config_path = tmp_path / "adaptive_scale_boundary.yaml"
    config_path.write_text(
        f"""
schemaVersion: "2.0"

benchmark:
  model: {defaults.model}
  endpoint:
    url: http://localhost:8000
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 1000
    prompts:
      isl: 550
      osl: 8
  phases:
    - name: profiling
      type: concurrency
      concurrency: 50
      duration: 10.0
      adaptive_scale:
        enabled: true
        control_variable: concurrency
        min_concurrency: 1
        assessment_period: 1.0
        min_completed_requests: 1
        sustain_duration: 1.0
        strategy:
          type: ramp_until_fail
          step_policy: sla_margin
          base_step: 10
          max_step_multiplier: 1
      sla:
        request_latency:
          p95:
            le: 80
""".lstrip(),
        encoding="utf-8",
    )

    original_config = FakeTransport._DEFAULT_CONFIG
    FakeTransport._DEFAULT_CONFIG = MockServerConfig(
        ttft=1.0,
        itl=0.0,
    )
    try:
        result = cli.run_sync(
            f"""
            aiperf profile
                --config {config_path}
                --extra-inputs ignore_eos:true
                --ui {defaults.ui}
            """,
            timeout=45.0,
        )
    finally:
        FakeTransport._DEFAULT_CONFIG = original_config

    assert result.exit_code == 0
    assert result.request_count > 0
    assert result.json.was_cancelled is False

    events = _load_jsonl(result.artifacts_dir / "adaptive_scale_events.jsonl")
    boundary_events = [
        event for event in events if event["event"] == "boundary_discovered"
    ]
    assert boundary_events

    discover_windows = [
        event
        for event in events
        if event["event"] == "adaptive_window" and event["phase"] == "discover"
    ]
    discover_values = [event["active_concurrency"] for event in discover_windows]
    assert len(discover_values) >= 2
    assert discover_values[0] == 1

    discover_decisions = [
        event
        for event in events
        if event["event"] == "adaptive_decision" and event["phase"] == "discover"
    ]
    scaled_values = [event["concurrency_after"] for event in discover_decisions]
    step_sizes = [event["step_size"] for event in discover_decisions]
    assert scaled_values
    assert scaled_values[0] > discover_values[0]
    assert step_sizes[0] >= 10
    assert scaled_values == sorted(scaled_values)

    boundary = boundary_events[-1]
    assert boundary["last_passing_value"] == boundary["boundary_concurrency"]
    assert boundary["first_failing_value"] > boundary["boundary_concurrency"]
    assert boundary["sla_value"] > boundary["sla_bound"]

    summary = orjson.loads(
        (result.artifacts_dir / "adaptive_scale_summary.json").read_bytes()
    )
    assert summary["boundary_concurrency"] == boundary["boundary_concurrency"]
    assert summary["first_failing_value"] == boundary["first_failing_value"]
    assert summary["control_value"] <= boundary["boundary_concurrency"]
    assert summary["completed_reason"] == "sustain_duration_completed"
    assert summary["sustain_windows"] > 0
