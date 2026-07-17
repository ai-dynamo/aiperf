# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ScenarioResolver step and its place in the default chain."""

from __future__ import annotations

from pathlib import Path

from aiperf.common.enums import CacheBustTarget
from aiperf.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.resolution.resolvers import (
    DatasetResolver,
    ScenarioResolver,
    TimingResolver,
    build_default_resolver_chain,
)

_WEKA_FIXTURE = (
    Path(__file__).parents[2] / "unit/graph/fixtures/weka_min.json"
).resolve()


def _make_graph_run(*, scenario: str | None) -> BenchmarkRun:
    cfg = BenchmarkConfig(
        models=["claude-opus-4-5-20251101"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[{"name": "profiling", "type": "file", "path": str(_WEKA_FIXTURE)}],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "sessions": 5,
            }
        ],
        scenario=scenario,
    )
    return BenchmarkRun(benchmark_id="test-run", cfg=cfg, artifact_dir=Path("/tmp/x"))


def test_scenario_resolver_populates_outcome_when_scenario_set() -> None:
    run = _make_graph_run(scenario="inferencex-agentx-mvp")
    ScenarioResolver().resolve(run)
    outcome = run.resolved.scenario_outcome
    assert outcome is not None
    assert outcome.submission_valid is True
    assert outcome.scenario_name == "inferencex-agentx-mvp"
    # The resolver's auto-fills landed on the live config.
    assert run.cfg.endpoint.cache_bust == CacheBustTarget.FIRST_TURN_PREFIX


def test_scenario_resolver_noop_when_scenario_unset() -> None:
    run = _make_graph_run(scenario=None)
    ScenarioResolver().resolve(run)
    assert run.resolved.scenario_outcome.submission_valid is None
    # No auto-fill when no scenario.
    assert run.cfg.endpoint.cache_bust == CacheBustTarget.NONE


def test_scenario_resolver_runs_between_dataset_and_timing() -> None:
    chain = build_default_resolver_chain()._resolvers
    types = [type(r) for r in chain]
    assert ScenarioResolver in types
    assert types.index(DatasetResolver) < types.index(ScenarioResolver)
    assert types.index(ScenarioResolver) < types.index(TimingResolver)


def test_scenario_resolver_autofilled_duration_visible_to_timing() -> None:
    """ScenarioResolver runs before TimingResolver, so the auto-filled
    1800s duration is summed into total_expected_duration."""
    run = _make_graph_run(scenario="inferencex-agentx-mvp")
    ScenarioResolver().resolve(run)
    TimingResolver().resolve(run)
    assert run.resolved.total_expected_duration == 1800.0
