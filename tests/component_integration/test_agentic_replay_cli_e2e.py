# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI-surface end-to-end tests for the ``agentic_replay`` timing mode."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI, AIPerfResults

pytestmark = pytest.mark.component_integration


def _write_weka_fixture(target_dir: Path, *, num_traces: int = 6) -> Path:
    """Write a minimal hash_id-valid weka trace fixture into ``target_dir``."""
    block_size = 16
    min_turns = 4
    target_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_traces):
        n = min_turns + i
        requests = []
        for k in range(n):
            hash_ids = list(range(1, k + 2))
            in_tokens = (k + 1) * block_size + 8
            requests.append(
                {
                    "t": k * 2.0,
                    "type": "n",
                    "model": "claude-opus-4-5-20251101",
                    "in": in_tokens,
                    "out": 8,
                    "hash_ids": hash_ids,
                    "input_types": ["text"],
                    "output_types": ["text"],
                    "stop": "end_turn",
                    "api_time": 0.05,
                    "think_time": 0.5 if k % 2 else 0.0,
                }
            )
        trace = {
            "id": f"trace_{n:02d}_n{n}",
            "models": ["claude-opus-4-5-20251101"],
            "block_size": block_size,
            "hash_id_scope": "local",
            "requests": requests,
        }
        (target_dir / f"trace_{n:02d}_n{n}.json").write_text(json.dumps(trace))
    return target_dir


@pytest.fixture
def weka_small_dir(tmp_path: Path) -> Path:
    """A 6-trace block-size-valid weka fixture written into tmp_path."""
    return _write_weka_fixture(tmp_path / "weka_small", num_traces=6)


def _build_command(weka_dir: Path, *, scenario: bool, unsafe_override: bool) -> str:
    """Build the full ``aiperf profile`` command line for the agentic_replay run."""
    cmd = f"""
        aiperf profile
            --model claude-haiku-4-5-20251001
            --model claude-opus-4-5-20251101
            --endpoint-type chat
            --streaming
            --custom-dataset-type weka_trace
            --input-file {weka_dir}
            --no-fixed-schedule
            --benchmark-duration 30
            --concurrency 4
            --random-seed 42
            --tokenizer {defaults.tokenizer}
            --extra-inputs ignore_eos:true
            --workers-max {defaults.workers_max}
            --ui {defaults.ui}
    """
    if scenario:
        cmd += " --scenario inferencex-agentx-mvp"
    if unsafe_override:
        cmd += " --unsafe-override"
    return cmd


def _assert_metric_present(
    result: AIPerfResults, metric_name: str, *, require_percentiles: bool = True
) -> None:
    """Assert a JSON-export metric is present and numerically populated."""
    assert result.json is not None, "JSON export must exist"
    metric = getattr(result.json, metric_name, None)
    assert metric is not None, f"metric {metric_name!r} missing from JSON export"
    assert metric.avg is not None and isinstance(metric.avg, int | float), (
        f"metric {metric_name!r} avg must be numeric"
    )
    if require_percentiles:
        for pct in ("p50", "p75", "p90", "p99"):
            value = getattr(metric, pct, None)
            assert value is not None and isinstance(value, int | float), (
                f"metric {metric_name!r} {pct} must be numeric (got {value!r})"
            )


@pytest.mark.component_integration
def test_agentic_replay_cli_scenario_unsafe_override_runs_to_completion(
    cli: AIPerfCLI,
    caplog: pytest.LogCaptureFixture,
    weka_small_dir: Path,
) -> None:
    """Spec section 8.2 #2 at the CLI surface."""
    caplog.set_level(logging.INFO, logger="aiperf.common.scenario.validator")

    cmd = _build_command(weka_small_dir, scenario=True, unsafe_override=True)
    result = cli.run_sync(cmd, timeout=defaults.timeout)

    assert result.exit_code == 0, (
        f"CLI run failed; stderr=\n{result.stderr}\n\nlog=\n{result.log}"
    )

    log_text = caplog.text
    assert "setting profiling-phase timing_mode" in log_text, (
        "scenario resolver must log the per-phase timing_mode auto-set under "
        "--scenario (covers the ConfigResolver -> apply_scenario chain)"
    )
    assert "auto-set --trace-idle-gap-cap-seconds=10.0" in log_text, (
        "resolver must auto-set the per-trace idle-gap cap when unset "
        "(the AgentX scenario locks trace_idle_gap_cap_seconds=10.0)"
    )

    assert result.json is not None, "JSON export must exist"
    assert result.request_count > 0, (
        "request_count must be > 0; warmup barrier did not release into "
        "PROFILING (likely a TrajectorySource construction or strategy bug)"
    )
    _assert_metric_present(result, "time_to_first_token")
    _assert_metric_present(result, "inter_token_latency")
    _assert_metric_present(result, "request_latency")
    _assert_metric_present(result, "output_sequence_length", require_percentiles=False)
    assert result.json.output_sequence_length is not None
    assert (result.json.output_sequence_length.avg or 0) >= 1, (
        "OSL avg should be >= 1 token under the weka small fixture"
    )

    extra = result.json.model_extra or {}
    metadata = extra.get("metadata", {}) if isinstance(extra, dict) else {}
    scenario_name = (
        metadata.get("scenario")
        or extra.get("scenario")
        or getattr(result.json, "scenario", None)
    )
    submission_valid = (
        metadata.get("submission_valid")
        if "submission_valid" in metadata
        else extra.get("submission_valid")
    )
    invalid_reasons = (
        metadata.get("submission_invalid_reasons")
        or extra.get("submission_invalid_reasons")
        or []
    )

    assert scenario_name == "inferencex-agentx-mvp", (
        f"scenario stamp missing or wrong: {scenario_name!r} "
        f"(metadata keys: {list(metadata.keys())}, extra keys: {list(extra.keys())})"
    )
    assert submission_valid is False, (
        "duration<900s under --unsafe-override must stamp submission_valid=False; "
        f"got {submission_valid!r}"
    )
    assert "unsafe_override" in invalid_reasons or any(
        "unsafe" in str(r).lower() or "duration" in str(r).lower()
        for r in invalid_reasons
    ), (
        f"submission_invalid_reasons must reference the override or "
        f"duration violation; got {invalid_reasons!r}"
    )


@pytest.mark.component_integration
def test_agentic_replay_cli_cache_warmup_runs_to_completion(
    cli: AIPerfCLI,
    weka_small_dir: Path,
) -> None:
    """E2E smoke for ``--agentic-cache-warmup-duration``."""
    cmd = _build_command(weka_small_dir, scenario=True, unsafe_override=True)
    cmd += " --agentic-cache-warmup-duration 2"
    result = cli.run_sync(cmd, timeout=defaults.timeout)

    assert result.exit_code == 0, (
        "cache-pressure warmup run must exit 0 (no deadlock in the warmup "
        f"substage, drain, or profiling handoff); stderr=\n{result.stderr}\n\n"
        f"log=\n{result.log}"
    )
    assert result.json is not None, "JSON export must exist"
    assert result.request_count > 0, (
        "request_count must be > 0; the warmup -> profiling handoff did not "
        "release PROFILING credits (accelerated-warmup barrier/handoff bug)"
    )


@pytest.mark.component_integration
def test_agentic_replay_cli_scenario_without_override_raises_lock_error(
    cli: AIPerfCLI, weka_small_dir: Path
) -> None:
    """Spec section 8.2 corollary: scenario lock errors block CLI startup."""
    cmd = _build_command(weka_small_dir, scenario=True, unsafe_override=False)
    result = cli.run_sync(cmd, timeout=defaults.timeout, assert_success=False)

    assert result.exit_code != 0, (
        "scenario lock without --unsafe-override must fail the run; "
        f"stderr=\n{result.stderr}\n\nlog=\n{result.log}"
    )
    combined = (result.stderr or "") + "\n" + (result.log or "")
    assert (
        "inferencex-agentx-mvp" in combined
        or "benchmark-duration" in combined
        or "ScenarioLockError" in combined
        or "scenario" in combined.lower()
    ), (
        "lock-error output must reference the scenario or violated flag; "
        f"got:\n{combined}"
    )
