# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess contract tests for adaptive scale."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


def _load_jsonl(path: Path) -> list[dict]:
    return [
        orjson.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adaptive_scale_subprocess_contract_with_deterministic_saturation(
    cli: AIPerfCLI,
    mock_server_factory: Callable[..., object],
    tmp_path: Path,
) -> None:
    """Verify the packaged CLI path discovers a boundary under controlled load.

    This is intentionally a deterministic saturation contract test, not a
    production realism benchmark. The mock server adds a quadratic TTFT penalty
    as active in-flight requests rise, giving adaptive scale a stable boundary
    to discover without monkeypatching strategy internals across the subprocess
    boundary.
    """
    async with mock_server_factory(
        ttft=15.0,
        itl=0.0,
        ttft_concurrency_quad_ms=3.0,
        ttft_jitter_cv=0.0,
        itl_jitter_cv=0.0,
        workers=1,
    ) as server:
        assert isinstance(server, AIPerfMockServer)
        config_path = tmp_path / "adaptive_scale_subprocess.yaml"
        config_path.write_text(
            f"""
schemaVersion: "2.0"

benchmark:
  model: {defaults.model}
  endpoint:
    url: {server.url}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 1000
    prompts:
      isl: 32
      osl: 8
  phases:
    - name: profiling
      type: concurrency
      concurrency: 8
      duration: 8.0
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
          base_step: 3
          max_step_multiplier: 1
      sla:
        request_latency:
          p95:
            le: 100
""".lstrip(),
            encoding="utf-8",
        )

        result = await cli.run(
            f"""
            aiperf profile \
                --config {config_path} \
                --extra-inputs ignore_eos:true \
                --workers-max {defaults.workers_max} \
                --tokenizer builtin \
                --ui none
            """,
            timeout=defaults.timeout,
        )

    assert result.exit_code == 0
    assert result.request_count > 0
    assert result.json is not None
    assert result.json.was_cancelled is False

    event_path = result.artifacts_dir / "adaptive_scale_events.jsonl"
    summary_path = result.artifacts_dir / "adaptive_scale_summary.json"
    assert event_path.exists()
    assert summary_path.exists()

    events = _load_jsonl(event_path)
    assert events
    event_names = {event["event"] for event in events}
    assert "adaptive_phase_started" in event_names
    assert "adaptive_window" in event_names
    assert "adaptive_decision" in event_names
    assert "boundary_discovered" in event_names
    assert "sustain_started" in event_names
    assert "adaptive_complete" in event_names

    discover_windows = [
        event
        for event in events
        if event["event"] == "adaptive_window" and event["phase"] == "discover"
    ]
    assert len(discover_windows) >= 2
    assert discover_windows[0]["active_concurrency"] == 1
    assert discover_windows[0]["schema_version"] == 1
    assert discover_windows[0]["timestamp_ns"] == discover_windows[0]["timestamp"]
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
        discover_windows[0]["timestamp_utc"],
    )

    discover_decisions = [
        event
        for event in events
        if event["event"] == "adaptive_decision" and event["phase"] == "discover"
    ]
    assert discover_decisions
    assert discover_decisions[0]["concurrency_after"] > 1
    assert all(event["step_size"] >= 1 for event in discover_decisions)

    boundary_events = [
        event for event in events if event["event"] == "boundary_discovered"
    ]
    boundary = boundary_events[-1]
    assert boundary["control_variable"] == "concurrency"
    assert boundary["last_passing_value"] == boundary["boundary_concurrency"]
    assert boundary["first_failing_value"] > boundary["boundary_concurrency"]
    assert boundary["sla_metric"] == "request_latency"
    assert boundary["sla_stat"] == "p95"
    assert boundary["sla_op"] == "le"
    assert boundary["sla_bound"] == 100
    assert boundary["sla_value"] > boundary["sla_bound"]

    sustain_windows = [
        event
        for event in events
        if event["event"] == "adaptive_window" and event["phase"] == "sustain"
    ]
    assert sustain_windows
    assert all(
        event["active_concurrency"] <= boundary["boundary_concurrency"]
        for event in sustain_windows
    )

    summary = orjson.loads(summary_path.read_bytes())
    assert summary["schema_version"] == 1
    assert summary["status"] == "completed"
    assert summary["control_variable"] == "concurrency"
    assert summary["sla"] == {
        "metric": "request_latency",
        "stat": "p95",
        "op": "le",
        "bound": 100,
    }
    assert summary["boundary_concurrency"] == boundary["boundary_concurrency"]
    assert summary["first_failing_value"] == boundary["first_failing_value"]
    assert summary["control_value"] <= boundary["boundary_concurrency"]
    assert summary["completed_reason"] == "sustain_duration_completed"
    assert summary["result"]["last_passing_value"] == summary["last_passing_value"]
    assert summary["result"]["first_failing_value"] == boundary["first_failing_value"]
    assert summary["result"]["boundary_value"] == boundary["boundary_concurrency"]
    assert summary["totals"]["sent"] >= summary["totals"]["completed"]
    assert summary["totals"]["cancelled"] is None
    assert summary["sustain_windows"] > 0
    assert summary["strategy_type"] == "ramp_until_fail"
    assert summary["step_policy"] == "sla_margin"
