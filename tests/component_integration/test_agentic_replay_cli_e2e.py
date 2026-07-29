# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-CLI-surface end-to-end tests driving ``aiperf profile --scenario inferencex-agentx-mvp`` through the genuine TrajectorySource + AgenticReplayStrategy engine and inspecting the JSON export and captured logs."""

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
    """Write a block-size-consistent, hash_id-valid multi-turn weka trace fixture into ``target_dir`` that the real synthesize path accepts."""
    # MIN_TURNS >= 2 is required: a single-turn trace samples at next_turn_index==0
    # with warmup_turn_index None, so it occupies a warmup lane but dispatches no
    # warmup credit, the barrier never releases, and the run hangs. Turn k uses
    # hash_ids=[1..k+1] with in=(k+1)*block_size+8 (partial final block).
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
    """Build the full ``aiperf profile`` command line for the agentic_replay run (duration below the 900s floor so it completes inside the test timeout)."""
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
    """Assert a JSON-export metric is present and numerically populated (avg plus percentile band)."""
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
    """Spec 8.2 #2 at the CLI surface: ``--scenario --unsafe-override`` exits 0, fires the resolver auto-sets, populates metrics, and stamps ``submission_valid=False``."""
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
    assert "auto-set --system-idle-gap-cap-seconds=10.0" in log_text, (
        "resolver must auto-set the global system-idle cap when unset without "
        "changing per-trace or per-turn timing"
    )

    assert result.json is not None, "JSON export must exist"
    assert result.request_count > 0, (
        "request_count must be > 0; warmup barrier did not release into "
        "PROFILING (likely a TrajectorySource construction or strategy bug)"
    )
    _assert_metric_present(result, "time_to_first_token")
    _assert_metric_present(result, "inter_token_latency")
    _assert_metric_present(result, "request_latency")
    # OSL (not ISL) is the load-bearing dataset-path proof: ISL is not recorded
    # client-side on the weka delta-replay path unless --use-server-token-count.
    _assert_metric_present(result, "output_sequence_length", require_percentiles=False)
    assert result.json.output_sequence_length is not None
    assert (result.json.output_sequence_length.avg or 0) >= 1, (
        "OSL avg should be >= 1 token under the weka small fixture"
    )

    # submission_* / scenario fields surface as extras (extra="allow"); the
    # exporter may land them at top level or under metadata, so check both.
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
@pytest.mark.parametrize(
    "warmup_option",
    [
        "--agentic-cache-warmup-duration 2",
        "--warmup-requests-per-lane 2",
    ],
)
def test_agentic_replay_cli_cache_warmup_runs_to_completion(
    cli: AIPerfCLI,
    weka_small_dir: Path,
    warmup_option: str,
) -> None:
    """Both cache-pressure modes complete their warmup, drain, and handoff."""
    cmd = _build_command(weka_small_dir, scenario=True, unsafe_override=True)
    cmd += f" {warmup_option}"
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
    """Spec 8.2 corollary: ``--scenario`` without ``--unsafe-override`` raises ``ScenarioLockError`` on the duration violation and exits non-zero before any dispatch."""
    cmd = _build_command(weka_small_dir, scenario=True, unsafe_override=False)
    result = cli.run_sync(cmd, timeout=defaults.timeout, assert_success=False)

    assert result.exit_code != 0, (
        "scenario lock without --unsafe-override must fail the run; "
        f"stderr=\n{result.stderr}\n\nlog=\n{result.log}"
    )
    # The error message must mention the scenario name or the violated flag so
    # users can act on it. Look across stderr+log because the error can route
    # to either depending on the failure mode.
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
