# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI-surface end-to-end tests for the ``agentic_replay`` timing mode.

Complements ``test_agentic_replay_e2e.py``, which stops at the
strategy/exporter boundary and constructs ``TrajectorySource`` /
``AgenticReplayStrategy`` directly from Python. This file drives the *full*
``aiperf profile --scenario inferencex-agentx-mvp --unsafe-override`` flow
through cyclopts via the in-process ``app(args)`` runner used by every other
component-integration test, then inspects the JSON export and captured logs.

The CLI surface this exercises that the strategy-boundary tests do *not*:

* cyclopts parsing of ``--scenario`` and ``--unsafe-override`` (both real
  ``CLIParameter`` flags hung off ``UserConfig``).
* ``UserConfig.model_post_init`` -> ``_run_scenario_validator`` ->
  ``validate_scenario`` firing during config construction (not from the
  manual ``MagicMock`` stubbing path in ``test_agentic_replay_e2e.py``).
* ``validate_scenario`` writing through to the read-only ``timing_mode``
  property: the validator falls back to ``user_config._timing_mode`` when the
  setter raises ``AttributeError``. ``test_agentic_replay_e2e.py`` mocks
  both attributes so this path is never exercised; the CLI test uses a real
  ``UserConfig`` where the property *is* read-only.
* The validator's auto-set behaviors mutating real config (``random_seed``,
  ``--inter-turn-delay-cap-seconds``, ``--use-think-time-only``,
  ``extra_inputs.ignore_eos``).
* ``PhaseOrchestrator`` (at ``timing/phase_orchestrator.py:120``)
  detecting ``timing_mode == AGENTIC_REPLAY`` on its phase configs and
  constructing a ``TrajectorySource`` instead of the default
  ``ConversationSource``.
* ``cli_runner._run_multi_benchmark`` stamping the validator-outcome carrier
  keys onto ``AggregateResult.metadata`` and the JSON exporter consuming
  them into ``submission_valid`` / ``submission_invalid_reasons`` /
  ``scenario`` fields.
* The full export pipeline producing the JSON file the user actually sees
  under ``artifacts/<run-id>/profile_export_aiperf.json``.

Note on fixtures:
    The shipped ``tests/fixtures/weka_traces_small/`` was designed for the
    strategy-boundary path in ``test_agentic_replay_e2e.py``, which
    monkeypatches ``synthesize_prompts_from_hash_ids`` to a no-op. Many of
    its turns satisfy ``len(hash_ids) * block_size > in[k]``, which the
    real ``PromptGenerator`` rejects with ``ConfigurationError``. The CLI
    path cannot easily monkeypatch loader internals, so this file builds
    its own block-size-consistent mini fixture in ``tmp_path`` via
    ``_write_weka_fixture``.
"""

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


# =============================================================================
# Fixture: per-test mini weka trace dataset (built fresh in tmp_path).
# =============================================================================


def _write_weka_fixture(target_dir: Path, *, num_traces: int = 6) -> Path:
    """Write a minimal hash_id-valid weka trace fixture into ``target_dir``.

    The shipped ``tests/fixtures/weka_traces_small/`` was designed for the
    strategy-boundary tests in ``test_agentic_replay_e2e.py``, which
    monkeypatch the loader's ``synthesize_prompts_from_hash_ids`` and
    never actually reconstruct prompts. Many of its turns satisfy
    ``len(hash_ids) * block_size > in[k]`` with a final-block size <=0,
    which is rejected by the real
    ``PromptGenerator.synthesize_prompts_from_hash_ids`` path that the CLI
    surface exercises. This helper writes a smaller (default 6-trace),
    block-size-consistent fixture instead so the full CLI pipeline can run
    end-to-end on tier 1 hardware without depending on tokenizer arithmetic.

    Per-trace shape:
    - trace_NN_nN.json with N in [1, num_traces]
    - block_size = 16 (FakeTokenizer encodes ~4 chars/token; we keep blocks
      small so synthetic prompts stay short and fixture write time stays
      negligible)
    - turn k uses hash_ids=[1..k+1] with in = (k+1) * block_size + 8 -- a
      partial final block of 8 tokens. Always satisfies
      ``(k+1)*16 < in <= (k+2)*16``.
    - api_time = 0.05s, think_time alternates 0/0.5s to exercise both paths.
    """
    block_size = 16
    target_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, num_traces + 1):
        requests = []
        for k in range(n):
            hash_ids = list(range(1, k + 2))
            in_tokens = (k + 1) * block_size + 8
            requests.append(
                {
                    "t": k * 1.0,
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
    """Build the full ``aiperf profile`` command line for the agentic_replay run.

    Uses ``--custom-dataset-type weka_trace`` because this is the explicit
    plugin name registered for the loader (see ``plugins.yaml``);
    ``--input-file`` alone does not auto-detect weka trace directories.

    Notes on values:
    - ``--benchmark-duration 30`` is intentionally below the
      ``min_benchmark_duration_seconds=900`` floor in
      ``inferencex-agentx-mvp`` so the run completes inside the test timeout
      while the scenario's ``--unsafe-override`` path is exercised.
    - ``--no-fixed-schedule`` suppresses the weka loader's default
      auto-activation of fixed-schedule mode, leaving timing under the
      AGENTIC_REPLAY strategy (which is what ``--scenario`` selects via
      the validator's ``user_config._timing_mode = AGENTIC_REPLAY`` write).
    - ``--concurrency 4`` and the small fixture's 10 traces keep the
      trajectory pool at 4 (min(concurrency, len(pool))) and force the
      recycle queue to spin up (10 - 4 = 6 entries).
    - ``--ui simple`` matches every other component-integration test;
      ``--ui dashboard`` would race with the in-process runner.
    - ``--tokenizer`` is overridden to ``defaults.tokenizer`` because the
      ``mock_tokenizer_from_pretrained`` autouse fixture intercepts
      ``Tokenizer.from_pretrained`` regardless of name; using the test
      default keeps logs/snapshots stable across the suite.
    """
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
    """Assert a JSON-export metric is present and numerically populated.

    Centralised because every metric assertion needs the same shape check
    (avg + percentile band) and inlining repeats noise.
    """
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


# =============================================================================
# Test 1: --scenario inferencex-agentx-mvp --unsafe-override drives the full
#         CLI surface to a successful exit and produces JSON with
#         submission_valid=False (duration below the 900s floor).
# =============================================================================


@pytest.mark.component_integration
def test_agentic_replay_cli_scenario_unsafe_override_runs_to_completion(
    cli: AIPerfCLI,
    caplog: pytest.LogCaptureFixture,
    weka_small_dir: Path,
) -> None:
    """Spec section 8.2 #2 at the CLI surface.

    Drives ``aiperf profile --scenario inferencex-agentx-mvp
    --unsafe-override`` against the small synthetic weka fixture through
    cyclopts + the in-process app runner, then verifies:

    1. Process exits 0 (no AIPerfMultiError, no ScenarioLockError, no
       crash from the read-only timing_mode property write inside the
       validator).
    2. The validator's auto-set hooks fired -- ``setting timing_mode=`` and
       ``auto-set --inter-turn-delay-cap-seconds=`` both surface in the
       captured log records (covers the ``model_post_init`` ->
       ``_run_scenario_validator`` chain).
    3. Streaming + non-streaming metrics (TTFT, TPOT, request_latency,
       ISL, OSL) are present and numerically populated -- proves the
       PhaseOrchestrator built a working TrajectorySource and dispatched
       through the credit pipeline to records-manager.
    4. ``request_count > 0`` -- proves the warmup barrier released and
       PROFILING dispatched real credits.
    5. The JSON export carries ``scenario: 'inferencex-agentx-mvp'`` and
       ``submission_valid: false`` with ``unsafe_override`` listed in
       ``submission_invalid_reasons`` (the duration-below-floor violation
       converted to a warning under unsafe-override).
    """
    caplog.set_level(logging.INFO, logger="aiperf.common.scenario.validator")

    cmd = _build_command(weka_small_dir, scenario=True, unsafe_override=True)
    result = cli.run_sync(cmd, timeout=defaults.timeout)

    assert result.exit_code == 0, (
        f"CLI run failed; stderr=\n{result.stderr}\n\nlog=\n{result.log}"
    )

    log_text = caplog.text
    assert "setting timing_mode" in log_text, (
        "validator must log timing_mode auto-set under --scenario "
        "(covers the read-only-property setter path against real UserConfig)"
    )
    assert "auto-set --trace-idle-gap-cap-seconds=60.0" in log_text, (
        "validator must auto-set the per-trace idle-gap cap when unset "
        "(the AgentX scenario locks trace_idle_gap_cap_seconds, not the "
        "inter-turn delay cap, since 932b4bc)"
    )

    assert result.json is not None, "JSON export must exist"
    assert result.request_count > 0, (
        "request_count must be > 0; warmup barrier did not release into "
        "PROFILING (likely a TrajectorySource construction or strategy bug)"
    )
    _assert_metric_present(result, "time_to_first_token")
    _assert_metric_present(result, "inter_token_latency")
    _assert_metric_present(result, "request_latency")
    # ISL/OSL come straight from records; the small fixture's traces have
    # in[k] in the low-hundreds of tokens, so a non-zero P50 confirms the
    # tokenizer + dataset path executed.
    _assert_metric_present(result, "input_sequence_length", require_percentiles=False)
    assert result.json.input_sequence_length is not None
    assert (result.json.input_sequence_length.p50 or 0) >= 1, (
        "ISL P50 should be >= 1 token under the weka small fixture"
    )

    # ---- (5) submission carrier keys + scenario stamp ----
    # JsonExportData has model_config = ConfigDict(extra="allow"), so the
    # exporter-stamped submission_* / scenario fields surface as raw extras.
    extra = result.json.model_extra or {}
    metadata = extra.get("metadata", {}) if isinstance(extra, dict) else {}
    # cli_runner stamps these onto AggregateResult.metadata, which the JSON
    # exporter folds into the top-level ``metadata`` block. Look in both
    # places to stay robust to where the exporter lands them.
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


# =============================================================================
# Test 2: --scenario without --unsafe-override fails fast on the duration
#         violation, proving the lock-error path is also wired through the
#         CLI surface (not just the strategy boundary).
# =============================================================================


@pytest.mark.component_integration
def test_agentic_replay_cli_scenario_without_override_raises_lock_error(
    cli: AIPerfCLI, weka_small_dir: Path
) -> None:
    """Spec section 8.2 corollary: scenario lock errors block CLI startup.

    Without ``--unsafe-override``, the validator's duration-below-floor
    violation must raise ``ScenarioLockError`` at startup, surfaced as a
    non-zero exit from cyclopts before any PhaseOrchestrator construction.

    Pinning this path catches regressions where:
    - ``model_post_init`` skips ``_run_scenario_validator`` entirely
      (e.g. someone marks the validator ``mode='before'`` and the
      pre-validation copy bypasses it).
    - ``validate_scenario`` swallows the lock error.
    - cyclopts re-raises but exits 0 for some reason.
    """
    cmd = _build_command(weka_small_dir, scenario=True, unsafe_override=False)
    result = cli.run_sync(cmd, timeout=defaults.timeout, assert_success=False)

    assert result.exit_code != 0, (
        "scenario lock without --unsafe-override must fail the run; "
        f"stderr=\n{result.stderr}\n\nlog=\n{result.log}"
    )
    # The error message must mention the scenario name or the violated flag
    # so users can act on it. Look across stderr+log because cyclopts can
    # route the error to either depending on the failure mode.
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
