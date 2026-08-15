# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScenarioResolver validates a named scenario, auto-fills the config it implies, and sits between the dataset and timing steps."""

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

_SCENARIO_LOADER = "semianalysis_cc_traces_weka_with_subagents"
_SCENARIO = "inferencex-agentx-mvp"


def _make_graph_run(artifact_dir: Path, *, scenario: str | None) -> BenchmarkRun:
    """Graph BenchmarkRun over the scenario's public dataset loader, with the given scenario lock."""
    cfg = BenchmarkConfig(
        models=["claude-opus-4-5-20251101"],
        endpoint={
            "urls": ["http://localhost:8000/v1/chat/completions"],
            "streaming": True,
            "extra": {"ignore_eos": True},
        },
        datasets=[{"name": "profiling", "type": "public", "dataset": _SCENARIO_LOADER}],
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
    return BenchmarkRun(benchmark_id="test-run", cfg=cfg, artifact_dir=artifact_dir)


def test_scenario_resolver_populates_outcome_when_scenario_set(tmp_path: Path) -> None:
    """A set scenario yields a valid outcome and lands its cache-bust auto-fill on the live config."""
    run = _make_graph_run(tmp_path, scenario=_SCENARIO)
    ScenarioResolver().resolve(run)
    outcome = run.resolved.scenario_outcome
    assert outcome is not None
    assert outcome.submission_valid is True
    assert outcome.scenario_name == _SCENARIO
    assert run.cfg.get_cache_bust_target() == CacheBustTarget.FIRST_TURN_PREFIX


def test_scenario_resolver_noop_when_scenario_unset(tmp_path: Path) -> None:
    """With no scenario the resolver leaves validity undecided and auto-fills nothing."""
    run = _make_graph_run(tmp_path, scenario=None)
    ScenarioResolver().resolve(run)
    assert run.resolved.scenario_outcome.submission_valid is None
    assert run.cfg.get_cache_bust_target() == CacheBustTarget.NONE


def test_scenario_resolver_runs_between_dataset_and_timing() -> None:
    """The default chain orders ScenarioResolver after DatasetResolver and before TimingResolver."""
    types = [type(r) for r in build_default_resolver_chain()._resolvers]
    assert ScenarioResolver in types
    assert types.index(DatasetResolver) < types.index(ScenarioResolver)
    assert types.index(ScenarioResolver) < types.index(TimingResolver)


def test_scenario_resolver_autofilled_duration_visible_to_timing(
    tmp_path: Path,
) -> None:
    """Because scenario resolution precedes timing, the auto-filled 1800s duration is summed into the expected total."""
    run = _make_graph_run(tmp_path, scenario=_SCENARIO)
    ScenarioResolver().resolve(run)
    TimingResolver().resolve(run)
    assert run.resolved.total_expected_duration == 1800.0
