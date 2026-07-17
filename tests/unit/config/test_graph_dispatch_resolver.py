# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GraphDispatchResolver: derives allow_dataset_wrap + publishes sampling.

Runs the real ScenarioResolver (so the scenario's cache-bust auto-fill lands on
``endpoint.cache_bust``) followed by the new GraphDispatchResolver, against real
config objects and a real weka graph fixture.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aiperf.common.enums import CacheBustTarget
from aiperf.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.resolution.resolvers import (
    DatasetResolver,
    GraphDispatchResolver,
    ScenarioResolver,
    build_default_resolver_chain,
)

_WEKA_FIXTURE = (
    Path(__file__).parents[2] / "unit/graph/fixtures/weka_min.json"
).resolve()


def _graph_run(
    *,
    scenario: str | None = None,
    cache_bust_unset: bool = True,
    allow_wrap_unset: bool = True,
    cache_bust: str | None = None,
    allow_wrap: bool | None = None,
) -> BenchmarkRun:
    """Build a graph-workload BenchmarkRun from real config objects.

    ``cache_bust``/``allow_wrap`` (when not None) are explicit user settings;
    the ``*_unset`` flags document the default-path cases and are otherwise
    unused (the None sentinels drive the actual wiring).
    """
    endpoint: dict[str, object] = {
        "urls": ["http://localhost:8000/v1/chat/completions"]
    }
    if cache_bust is not None:
        endpoint["cache_bust"] = cache_bust

    dataset: dict[str, object] = {
        "name": "profiling",
        "type": "file",
        "path": str(_WEKA_FIXTURE),
    }
    if allow_wrap is not None:
        dataset["synthesis"] = {"allow_dataset_wrap": allow_wrap}

    cfg = BenchmarkConfig(
        models=["claude-opus-4-5-20251101"],
        endpoint=endpoint,
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
    )
    return BenchmarkRun(benchmark_id="test-run", cfg=cfg, artifact_dir=Path("/tmp/x"))


def apply_chain(run: BenchmarkRun) -> SimpleNamespace:
    """Run the real ScenarioResolver then GraphDispatchResolver over ``run``.

    Returns a view exposing the resolved endpoint (for cache_bust assertions)
    and the resolved wrap/sampling fields.
    """
    DatasetResolver().resolve(run)
    ScenarioResolver().resolve(run)
    GraphDispatchResolver().resolve(run)
    return SimpleNamespace(
        endpoint=run.cfg.endpoint,
        allow_dataset_wrap=run.resolved.allow_dataset_wrap,
        dataset_sampling_strategy=run.resolved.dataset_sampling_strategy,
    )


def test_wrap_default_true_when_scenario_forces_cache_bust():
    run = _graph_run(
        scenario="inferencex-agentx-mvp", cache_bust_unset=True, allow_wrap_unset=True
    )
    resolved = apply_chain(
        run
    )  # full default chain incl. ScenarioResolver then the new step
    assert resolved.endpoint.cache_bust == CacheBustTarget.FIRST_TURN_PREFIX
    assert resolved.allow_dataset_wrap is True


def test_wrap_default_false_when_cache_bust_none():
    run = _graph_run(scenario=None, cache_bust_unset=True, allow_wrap_unset=True)
    assert apply_chain(run).allow_dataset_wrap is False


def test_explicit_allow_wrap_false_wins_even_with_cache_bust():
    run = _graph_run(cache_bust="first_turn_prefix", allow_wrap=False)
    assert apply_chain(run).allow_dataset_wrap is False


def test_graph_dispatch_resolver_runs_after_scenario_resolver():
    chain = build_default_resolver_chain()._resolvers
    types = [type(r) for r in chain]
    assert GraphDispatchResolver in types
    assert types.index(ScenarioResolver) < types.index(GraphDispatchResolver)


def test_publishes_dataset_sampling_strategy_for_graph_workload():
    from aiperf.plugin.enums import DatasetSamplingStrategy

    run = _graph_run(scenario=None)
    resolved = apply_chain(run)
    assert resolved.dataset_sampling_strategy == DatasetSamplingStrategy.SEQUENTIAL
