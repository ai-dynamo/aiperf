# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the graph-IR scenario invariant-lock validator (apply_scenario).

These build REAL ``BenchmarkRun`` objects (NOT MagicMock) against a real weka
graph-workload fixture: MagicMock auto-creates any attribute path and would
hide real-config drift (e.g. a renamed ``endpoint.cache_bust`` field or a
non-graph dataset slipping past ``resolve_graph_workload``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import ScenarioLockError, apply_scenario
from aiperf.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun

_WEKA_FIXTURE = (
    Path(__file__).parents[2] / "unit/graph/fixtures/weka_min.json"
).resolve()
_SCENARIO = "inferencex-agentx-mvp"
_MODEL = "claude-opus-4-5-20251101"


def _make_graph_run(
    *,
    scenario: str | None = _SCENARIO,
    unsafe_override: bool = False,
    dataset_synthesis: dict | None = None,
    trajectory_start_min_ratio: float | None = None,
    trajectory_start_max_ratio: float | None = None,
    **endpoint,
) -> BenchmarkRun:
    """Build a real graph-workload ``BenchmarkRun`` wrapping the weka fixture.

    The profiling phase uses ``sessions`` (not ``duration``) so the duration
    auto-fill path is exercised. ``endpoint`` kwargs are passed through so a
    test can set an explicit ``streaming`` / ``cache_bust`` to trigger a
    violation. ``dataset_synthesis`` merges a ``synthesis`` block onto the weka
    dataset (e.g. to set ``idle_gap_cap_seconds``). The trajectory kwargs mark
    the top-level window fields user-explicit (only when passed).
    """
    dataset: dict = {"name": "profiling", "type": "file", "path": str(_WEKA_FIXTURE)}
    if dataset_synthesis is not None:
        dataset["synthesis"] = dataset_synthesis
    top_level: dict = {}
    if trajectory_start_min_ratio is not None:
        top_level["trajectory_start_min_ratio"] = trajectory_start_min_ratio
    if trajectory_start_max_ratio is not None:
        top_level["trajectory_start_max_ratio"] = trajectory_start_max_ratio
    cfg = BenchmarkConfig(
        models=[_MODEL],
        endpoint={
            "urls": ["http://localhost:8000/v1/chat/completions"],
            **endpoint,
        },
        datasets=[dataset],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "sessions": 5,
            }
        ],
        scenario=scenario,
        unsafe_override=unsafe_override,
        **top_level,
    )
    return BenchmarkRun(benchmark_id="test-run", cfg=cfg, artifact_dir=Path("/tmp/x"))


def _make_synthetic_run() -> BenchmarkRun:
    """Build a real NON-graph (synthetic) run with the scenario set."""
    cfg = BenchmarkConfig(
        models=["m"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[
            {
                "name": "profiling",
                "type": "synthetic",
                "entries": 5,
                "prompts": {"isl": 32},
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "sessions": 5,
            }
        ],
        scenario=_SCENARIO,
        unsafe_override=True,
    )
    return BenchmarkRun(benchmark_id="test-run", cfg=cfg, artifact_dir=Path("/tmp/x"))


def _flags(run: BenchmarkRun) -> list[str]:
    """Apply the scenario (override active) and return the violation flags."""
    outcome = apply_scenario(run)
    return [v.flag for v in outcome.violations]


# Scenario application is config-native (apply-or-lock on run.cfg) and
# performs no process-global writes, so nothing can leak across tests.


# ---------------------------------------------------------------------------
# No-scenario short-circuit
# ---------------------------------------------------------------------------


def test_no_scenario_is_noop_outcome() -> None:
    run = _make_graph_run(scenario=None)
    outcome = apply_scenario(run)
    assert outcome.scenario_name is None
    assert outcome.submission_valid is None
    assert run.resolved.scenario_outcome is outcome


# ---------------------------------------------------------------------------
# Per-_apply_* auto-fills (unset -> filled, no violation)
# ---------------------------------------------------------------------------


def test_streaming_autoenabled_when_unset() -> None:
    run = _make_graph_run()
    apply_scenario(run)
    assert run.cfg.endpoint.streaming is True


def test_ignore_eos_injected_when_absent() -> None:
    run = _make_graph_run()
    apply_scenario(run)
    assert run.cfg.endpoint.extra["ignore_eos"] is True


def test_cache_bust_autofilled_when_default() -> None:
    run = _make_graph_run()
    apply_scenario(run)
    assert run.cfg.endpoint.cache_bust == CacheBustTarget.FIRST_TURN_PREFIX


def test_duration_autofilled_when_unset() -> None:
    run = _make_graph_run()
    apply_scenario(run)
    assert run.cfg.get_profiling_phases()[0].duration == 1800.0


def test_timing_mode_passes_for_graph_workload() -> None:
    run = _make_graph_run()
    outcome = apply_scenario(run)
    assert "timing_mode" in outcome.applied_locks


# ---------------------------------------------------------------------------
# Per-_apply_* violations (explicit conflict)
# ---------------------------------------------------------------------------


def test_streaming_explicit_false_violates() -> None:
    run = _make_graph_run(unsafe_override=True, streaming=False)
    assert "--streaming" in _flags(run)


def test_cache_bust_explicit_conflict_violates() -> None:
    run = _make_graph_run(unsafe_override=True, cache_bust="none")
    assert "--cache-bust" in _flags(run)


def test_non_graph_workload_violates_timing_mode_and_loader() -> None:
    run = _make_synthetic_run()
    flags = _flags(run)
    assert "--input-file (timing_mode)" in flags
    assert "--input-file (loader)" in flags


# ---------------------------------------------------------------------------
# Config apply-or-lock (explicit flag values conflicting with the spec)
# ---------------------------------------------------------------------------


def test_trajectory_start_min_ratio_flag_override_violates() -> None:
    # Under apply-or-lock semantics an override is "explicit" only when the
    # config field is user-set; unset would simply be auto-applied.
    run = _make_graph_run(
        unsafe_override=True,
        trajectory_start_min_ratio=0.10,
        trajectory_start_max_ratio=1.0,
    )
    assert "--trajectory-start-min-ratio" in _flags(run)


def test_trace_idle_gap_cap_config_override_violates() -> None:
    # Scenario locks the cap to 10.0; a run setting --synthesis-idle-gap-cap=30
    # via config must violate.
    run = _make_graph_run(
        unsafe_override=True, dataset_synthesis={"idle_gap_cap_seconds": 30.0}
    )
    assert "--synthesis-idle-gap-cap" in _flags(run)


def test_trace_idle_gap_cap_explicit_match_passes() -> None:
    # An explicit --synthesis-idle-gap-cap equal to the spec (10.0) locks clean.
    run = _make_graph_run(dataset_synthesis={"idle_gap_cap_seconds": 10.0})
    outcome = apply_scenario(run)
    assert "trace_idle_gap_cap" in outcome.applied_locks
    assert outcome.violations == []


def test_env_defaults_satisfy_scenario_no_violation() -> None:
    # Sanity: with a bare run the t* window (0.0/1.0) and idle-gap cap (10.0)
    # are auto-applied, not violated.
    run = _make_graph_run()
    outcome = apply_scenario(run)
    assert "trajectory_start_ratios" in outcome.applied_locks
    assert "trace_idle_gap_cap" in outcome.applied_locks
    # The cap landed on the live config (unset -> scenario auto-applies 10.0),
    # overriding the bare 60s adapter default.
    dataset = run.cfg.get_default_dataset()
    assert dataset.synthesis.idle_gap_cap_seconds == 10.0


# ---------------------------------------------------------------------------
# End-to-end: all auto-fills + submission_valid; conflict raises; override
# ---------------------------------------------------------------------------


def test_end_to_end_all_autofills_submission_valid() -> None:
    run = _make_graph_run()
    outcome = apply_scenario(run)
    assert outcome.submission_valid is True
    assert outcome.scenario_name == _SCENARIO
    assert outcome.violations == []
    # Every auto-fill landed on the live config.
    assert run.cfg.endpoint.cache_bust == CacheBustTarget.FIRST_TURN_PREFIX
    assert run.cfg.endpoint.streaming is True
    assert run.cfg.endpoint.extra["ignore_eos"] is True
    assert run.cfg.get_profiling_phases()[0].duration == 1800.0
    assert run.resolved.scenario_outcome is outcome


def test_end_to_end_explicit_conflict_raises_lock_error() -> None:
    run = _make_graph_run(streaming=False)
    with pytest.raises(ScenarioLockError):
        apply_scenario(run)


def test_end_to_end_unsafe_override_downgrades_and_stamps_invalid() -> None:
    run = _make_graph_run(unsafe_override=True, streaming=False)
    outcome = apply_scenario(run)
    assert outcome.submission_valid is False
    assert outcome.submission_invalid_reasons == ["unsafe_override"]
    assert any(v.flag == "--streaming" for v in outcome.violations)
    assert run.resolved.scenario_outcome is outcome


# ---------------------------------------------------------------------------
# C1 regression: model_fields_set is stale after the multi-run subprocess
# round-trip (model_dump(exclude_none=True) -> model_validate re-marks every
# non-None field as set). Resolving in the PARENT before the dump must bake the
# auto-fills so the in-subprocess re-resolution is a clean no-op (no spurious
# ScenarioLockError on streaming / cache_bust).
# ---------------------------------------------------------------------------


def _roundtrip(run: BenchmarkRun) -> BenchmarkRun:
    """Mirror the orchestrator->subprocess hop (local_executor + subprocess_runner)."""
    import orjson

    data = run.model_dump(mode="json", exclude_none=True)
    return BenchmarkRun.model_validate(orjson.loads(orjson.dumps(data)))


def test_roundtrip_without_parent_resolve_marks_defaults_as_set() -> None:
    """Documents the C1 trap: the round-trip makes endpoint.model_fields_set
    spuriously contain streaming + cache_bust even when the user set neither."""
    run = _make_graph_run()
    assert run.cfg.endpoint.model_fields_set == {"urls"}
    run2 = _roundtrip(run)
    assert "streaming" in run2.cfg.endpoint.model_fields_set
    assert "cache_bust" in run2.cfg.endpoint.model_fields_set


def test_parent_resolve_then_roundtrip_then_resolve_no_spurious_violation() -> None:
    """C1 FIX: resolve in the parent (faithful model_fields_set), dump+reload,
    then re-resolve in the 'subprocess' -- must NOT raise and stay valid."""
    from aiperf.orchestrator.local_executor import _resolve_scenario_in_parent

    parent = _make_graph_run()
    _resolve_scenario_in_parent(parent)
    assert parent.resolved.scenario_outcome.submission_valid is True

    child = _roundtrip(parent)
    # The parent baked the auto-fills; the child sees satisfying values.
    assert child.cfg.endpoint.streaming is True
    assert child.cfg.endpoint.cache_bust == CacheBustTarget.FIRST_TURN_PREFIX
    assert child.cfg.get_profiling_phases()[0].duration == 1800.0

    # Subprocess re-resolution is a clean no-op (no spurious ScenarioLockError).
    outcome = apply_scenario(child)
    assert outcome.submission_valid is True
    assert outcome.violations == []


def test_roundtrip_without_parent_resolve_raises_spurious_lock_error() -> None:
    """Proves the bug exists absent the parent-resolve: a default scenario run
    that round-trips and is THEN resolved (as the subprocess does) raises."""
    run = _make_graph_run()
    child = _roundtrip(run)
    with pytest.raises(ScenarioLockError):
        apply_scenario(child)


def test_parent_resolve_noop_when_no_scenario() -> None:
    from aiperf.orchestrator.local_executor import _resolve_scenario_in_parent

    run = _make_graph_run(scenario=None)
    _resolve_scenario_in_parent(run)
    assert run.resolved.scenario_outcome is None
    # No auto-fill applied when no scenario.
    assert run.cfg.endpoint.cache_bust == CacheBustTarget.NONE


# ---------------------------------------------------------------------------
# Gap I1: concurrency-sweep rejection (a swept concurrency multiplies the
# locked config into N diverging runs; AgentX rejects a list-shaped concurrency)
# ---------------------------------------------------------------------------


def _make_swept_graph_run(values: dict) -> BenchmarkRun:
    """A graph run whose SweepVariation records a swept dotted-path key."""
    from aiperf.config.sweep import SweepVariation

    run = _make_graph_run(unsafe_override=True)
    run.variation = SweepVariation(index=0, label="swept", values=values)
    return run


def test_concurrency_sweep_under_scenario_violates() -> None:
    run = _make_swept_graph_run({"phases.profiling.concurrency": 20})
    assert "--concurrency" in _flags(run)


def test_prefill_concurrency_sweep_under_scenario_violates() -> None:
    run = _make_swept_graph_run({"phases.profiling.prefill_concurrency": 4})
    assert "--concurrency" in _flags(run)


def test_non_concurrency_sweep_key_does_not_violate() -> None:
    # A request-rate sweep key is not a concurrency sweep -> no concurrency flag.
    run = _make_swept_graph_run({"phases.profiling.request_rate": 5.0})
    assert "--concurrency" not in _flags(run)


def test_single_run_no_variation_does_not_violate_concurrency() -> None:
    run = _make_graph_run()
    outcome = apply_scenario(run)
    assert all(v.flag != "--concurrency" for v in outcome.violations)


# ---------------------------------------------------------------------------
# Gap I3: require_loader full entry list + canonical weka HF-repo pin
# ---------------------------------------------------------------------------


def _make_hf_graph_run(hf_id: str, *, unsafe_override: bool = True) -> BenchmarkRun:
    """A graph run whose --input-file is a HuggingFace dataset id string."""
    cfg = BenchmarkConfig(
        models=[_MODEL],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[{"name": "profiling", "type": "file", "path": hf_id}],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "sessions": 5,
            }
        ],
        scenario=_SCENARIO,
        unsafe_override=unsafe_override,
    )
    return BenchmarkRun(benchmark_id="test-run", cfg=cfg, artifact_dir=Path("/tmp/x"))


def test_canonical_weka_hf_repo_passes_loader_and_pin() -> None:
    run = _make_hf_graph_run("semianalysisai/cc-traces-weka-062126")
    outcome = apply_scenario(run)
    assert "require_loader" in outcome.applied_locks
    assert all(
        v.flag not in ("--input-file (loader)", "--input-file (hf-repo)")
        for v in outcome.violations
    )


def test_canonical_weka_hf_repo_256k_variant_passes() -> None:
    run = _make_hf_graph_run("semianalysisai/cc-traces-weka-062126-256k")
    outcome = apply_scenario(run)
    assert "require_loader" in outcome.applied_locks


def test_local_weka_file_workload_passes_loader() -> None:
    # The local-file fixture is a weka graph workload -> loader OK, no HF pin.
    run = _make_graph_run()
    outcome = apply_scenario(run)
    assert "require_loader" in outcome.applied_locks


def test_foreign_hf_repo_with_weka_marker_violates_pin() -> None:
    # Carries the "weka" marker (so it sniffs as a graph workload) but is NOT
    # under the canonical SemiAnalysis prefix -> submission_invalid.
    run = _make_hf_graph_run("someorg/my-weka-traces")
    flags = _flags(run)
    assert "--input-file (hf-repo)" in flags
    assert "require_loader" not in apply_scenario(run).applied_locks


def test_non_weka_hf_repo_violates_loader() -> None:
    # No "weka" marker -> not a graph workload at all -> loader violation.
    run = _make_hf_graph_run("meta-llama/Llama-3")
    flags = _flags(run)
    assert "--input-file (loader)" in flags
