# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for the scenario validator -> aggregate exporter wire.

Pins the full chain:

    UserConfig -> validate_scenario(cfg) -> ValidationOutcome
                                          -> AggregateResult.metadata carrier keys
                                          -> AggregateConfidenceJsonExporter.export()
                                          -> final JSON metadata fields

The validator both *returns* a ValidationOutcome and *mutates* the user_config
in place (auto-injecting ignore_eos, random_seed, inter_turn_delay_cap, and the
storage backing timing_mode at default). The cli_runner wire then stamps the
outcome onto AggregateResult.metadata via underscore-prefixed carrier keys
(``_scenario_name``, ``_validator_submission_valid``,
``_validator_submission_invalid_reasons``, ``_total_responses``,
``_context_overflow_count``). The JSON exporter pops those keys and emits the
final ``scenario`` / ``submission_valid`` / ``submission_invalid_reasons``
fields under ``metadata``.

Closely mirrors:
- tests/component_integration/test_submission_valid_adversarial.py
  (the _make_aggregate / _export_and_load helpers + carrier-key contract)
- tests/component_integration/test_agentic_replay_e2e.py
  (the _make_aggregate_with_carriers factory)
- tests/unit/common/scenario/test_scenario_validator_adversarial.py
  (the _user_config MagicMock helper)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import (
    ScenarioLockError,
    validate_scenario,
)
from aiperf.exporters.aggregate import (
    AggregateConfidenceJsonExporter,
    AggregateExporterConfig,
)
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.plugin.enums import TimingMode
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.component_integration


# ---------------------------------------------------------------------------
# Helpers (inlined from sister test files; kept local to avoid coupling).
# ---------------------------------------------------------------------------


def _user_config(
    *,
    scenario: str | None = "inferencex-agentx-mvp",
    timing_mode: TimingMode | str = TimingMode.AGENTIC_REPLAY,
    extra_inputs: dict | None = None,
    use_think_time_only: bool = True,
    ignore_trace_delays: bool = False,
    synthesis_max_isl: int | None = None,
    loader: str | None = "semianalysis_cc_traces_weka_with_subagents",
    benchmark_duration: float | None = 900.0,
    inter_turn_delay_cap_seconds: float | None = None,
    trace_idle_gap_cap_seconds: float | None = 60.0,
    random_seed: int | None = 42,
    unsafe_override: bool = False,
    cache_bust_target: CacheBustTarget | None = None,
) -> MagicMock:
    """Build a MagicMock UserConfig pre-shaped for the scenario validator.

    Mirrors the helper in tests/unit/common/scenario/test_scenario_validator_adversarial.py
    so the same defaults flow through both unit and integration suites.
    """
    cfg = MagicMock()
    cfg.scenario = scenario
    cfg.unsafe_override = unsafe_override
    cfg.timing_mode = timing_mode
    cfg.input.extra_inputs_parsed = extra_inputs if extra_inputs is not None else {}
    cfg.input.use_think_time_only = use_think_time_only
    cfg.input.ignore_trace_delays = ignore_trace_delays
    cfg.input.random_seed = random_seed
    cfg.input.synthesis.max_isl = synthesis_max_isl
    cfg.input.detected_loader = loader
    cfg.loadgen.benchmark_duration = benchmark_duration
    cfg.loadgen.inter_turn_delay_cap_seconds = inter_turn_delay_cap_seconds
    cfg.loadgen.trace_idle_gap_cap_seconds = trace_idle_gap_cap_seconds
    cfg.input._use_think_time_only_explicitly_set = False
    cfg.loadgen._inter_turn_delay_cap_explicitly_set = False
    cfg.loadgen._trace_idle_gap_cap_explicitly_set = False
    # Scenario lock requires cache_bust.target=FIRST_TURN_PREFIX. Default to it
    # so tests targeting OTHER invariants don't trip the cache-bust check.
    cfg.input.prompt.cache_bust.target = (
        cache_bust_target
        if cache_bust_target is not None
        else CacheBustTarget.FIRST_TURN_PREFIX
    )
    cfg.input.prompt.cache_bust._target_explicitly_set = False
    return cfg


def _make_aggregate(metadata: dict) -> AggregateResult:
    """Build a minimal AggregateResult carrying the given metadata.

    Identical shape to test_submission_valid_adversarial.py::_make_aggregate.
    """
    return AggregateResult(
        aggregation_type="confidence",
        num_runs=1,
        num_successful_runs=1,
        failed_runs=[],
        metrics={},
        metadata=metadata,
    )


def _aggregate_from_outcome(
    outcome,
    *,
    scenario_name: str,
    total_responses: int = 500,
    context_overflow_count: int = 0,
) -> AggregateResult:
    """Stamp a ValidationOutcome onto an AggregateResult via the cli_runner carrier-key contract."""
    return _make_aggregate(
        {
            "_scenario_name": scenario_name,
            "_validator_submission_valid": outcome.submission_valid,
            "_validator_submission_invalid_reasons": list(
                outcome.submission_invalid_reasons
            ),
            "_total_responses": total_responses,
            "_context_overflow_count": context_overflow_count,
        }
    )


async def _export_and_load(aggregate: AggregateResult, tmp_path: Path) -> dict:
    """Write the aggregate via the JSON exporter and return the parsed JSON."""
    config = AggregateExporterConfig(result=aggregate, output_dir=tmp_path)
    exporter = AggregateConfidenceJsonExporter(config)
    out_path = await exporter.export()
    with open(out_path) as f:
        return json.load(f)


def _make_dataset_metadata(turn_counts_by_id: dict[str, int]) -> MagicMock:
    """Build a MagicMock DatasetMetadata with the requested turn counts.

    Mirrors tests/unit/timing/test_trajectory_source.py::_make_dataset_metadata.
    Used by test_validator_auto_sets_random_seed_when_unset to confirm the
    auto-set seed produces deterministic trajectories across two sources.
    """
    md = MagicMock()
    convs = []
    for cid, n in turn_counts_by_id.items():
        c = MagicMock()
        c.conversation_id = cid
        c.turns = [MagicMock(has_forks=False) for _ in range(n)]
        convs.append(c)
    md.conversations = convs
    return md


class _SequentialSampler:
    """Deterministic sampler over a fixed conversation_id list (rooted only).

    Mirrors tests/component_integration/test_agentic_replay_e2e.py::_SequentialSampler.
    """

    def __init__(self, conversation_ids: list[str]) -> None:
        self._ids = list(conversation_ids)
        self._idx = 0

    def next_conversation_id(self) -> str:
        if self._idx >= len(self._ids):
            raise StopIteration
        cid = self._ids[self._idx]
        self._idx += 1
        return cid


# ---------------------------------------------------------------------------
# Test 1: clean scenario -> validator returns submission_valid=True ->
# aggregate JSON metadata.scenario + metadata.submission_valid == True;
# no submission_invalid_reasons key (sister test pinned: omitted, not [] empty).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_scenario_validator_to_exporter_yields_submission_valid_true(
    tmp_path: Path,
) -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": True})

    outcome = validate_scenario(cfg)

    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert outcome.submission_invalid_reasons == []

    aggregate = _aggregate_from_outcome(outcome, scenario_name="inferencex-agentx-mvp")
    data = await _export_and_load(aggregate, tmp_path)

    md = data["metadata"]
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is True
    # Pinned by test_submission_valid_adversarial.py: when no reasons exist
    # the field is omitted entirely (not emitted as []).
    assert "submission_invalid_reasons" not in md
    # Carrier keys are stripped.
    for key in (
        "_scenario_name",
        "_validator_submission_valid",
        "_validator_submission_invalid_reasons",
        "_total_responses",
        "_context_overflow_count",
    ):
        assert key not in md


# ---------------------------------------------------------------------------
# Test 2: --unsafe-override + violations -> submission_valid=False with
# unsafe_override reason flowing all the way to the JSON metadata.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsafe_override_with_violations_yields_submission_valid_false_with_reasons(
    tmp_path: Path,
) -> None:
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        benchmark_duration=10.0,  # < 900s floor -> violation
        synthesis_max_isl=128,  # forbid_input_truncation -> violation
        unsafe_override=True,
    )

    outcome = validate_scenario(cfg)

    # Both violations were collected and the override flipped submission_valid to False.
    assert outcome.submission_valid is False
    assert "unsafe_override" in outcome.submission_invalid_reasons
    flags = [v.flag for v in outcome.violations]
    assert "--benchmark-duration" in flags
    assert "--synthesis-max-isl" in flags

    aggregate = _aggregate_from_outcome(outcome, scenario_name="inferencex-agentx-mvp")
    data = await _export_and_load(aggregate, tmp_path)

    md = data["metadata"]
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is False
    assert "unsafe_override" in md["submission_invalid_reasons"]


# ---------------------------------------------------------------------------
# Test 3: random_seed=None -> validator auto-sets it; reusing that seed
# yields deterministic trajectories.
# ---------------------------------------------------------------------------


def test_validator_auto_sets_random_seed_when_unset() -> None:
    cfg = _user_config(extra_inputs={"ignore_eos": True}, random_seed=None)

    outcome = validate_scenario(cfg)

    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert cfg.input.random_seed is not None
    assert isinstance(cfg.input.random_seed, int)
    assert cfg.input.random_seed >= 0  # secrets.randbits returns non-negative

    # Capture the auto-set seed and use it to drive two independent
    # TrajectorySources -- the trajectories must match exactly.
    seed = cfg.input.random_seed
    md1 = _make_dataset_metadata({"a": 10, "b": 10, "c": 10, "d": 10})
    md2 = _make_dataset_metadata({"a": 10, "b": 10, "c": 10, "d": 10})

    s1 = TrajectorySource(
        dataset_metadata=md1,
        dataset_sampler=_SequentialSampler(["a", "b", "c", "d"]),
        concurrency=4,
        random_seed=seed,
    )
    s2 = TrajectorySource(
        dataset_metadata=md2,
        dataset_sampler=_SequentialSampler(["a", "b", "c", "d"]),
        concurrency=4,
        random_seed=seed,
    )

    k1 = [(t.conversation_id, t.start_turn_index) for t in s1.trajectories]
    k2 = [(t.conversation_id, t.start_turn_index) for t in s2.trajectories]
    assert k1 == k2
    # And the trajectories actually populated (not empty by accident).
    assert len(k1) == 4


# ---------------------------------------------------------------------------
# Test 4: extra_inputs missing ignore_eos -> validator auto-injects True.
# ---------------------------------------------------------------------------


def test_validator_auto_injects_ignore_eos_when_absent() -> None:
    cfg = _user_config(extra_inputs={})

    outcome = validate_scenario(cfg)

    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert cfg.input.extra_inputs_parsed["ignore_eos"] is True


# ---------------------------------------------------------------------------
# Test 5: trace_idle_gap_cap_seconds=None + not explicitly set ->
# validator auto-sets it to the spec's locked 60.0.
# ---------------------------------------------------------------------------


def test_validator_auto_sets_trace_idle_gap_cap_when_unset() -> None:
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        trace_idle_gap_cap_seconds=None,
    )
    cfg.loadgen._trace_idle_gap_cap_explicitly_set = False

    outcome = validate_scenario(cfg)

    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert cfg.loadgen.trace_idle_gap_cap_seconds == 60.0


# ---------------------------------------------------------------------------
# Test 6: violations + unsafe_override=False -> ScenarioLockError;
# no AggregateResult should be constructed in this path.
# ---------------------------------------------------------------------------


def test_scenario_lock_error_prevents_aggregate_construction() -> None:
    cfg = _user_config(
        extra_inputs={"ignore_eos": True},
        benchmark_duration=10.0,  # violation 1
        synthesis_max_isl=128,  # violation 2
        unsafe_override=False,  # default
    )

    with pytest.raises(ScenarioLockError) as exc:
        validate_scenario(cfg)

    # Pin the violation count -- production code halts before aggregation,
    # so the test path likewise constructs no AggregateResult below.
    assert len(exc.value.violations) == 2
    flags = [v.flag for v in exc.value.violations]
    assert "--benchmark-duration" in flags
    assert "--synthesis-max-isl" in flags
