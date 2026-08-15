# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GraphDispatchResolver derives ``allow_dataset_wrap`` from cache-bust and publishes the graph sampling strategy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import param

from aiperf.common.enums import CacheBustTarget
from aiperf.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.resolution.resolvers import (
    DatasetResolver,
    GraphDispatchResolver,
    ScenarioResolver,
    build_default_resolver_chain,
)
from tests.unit.config.conftest import GRAPH_TRACE_FIXTURE

_GRAPH_FIXTURE = GRAPH_TRACE_FIXTURE.resolve()


def _graph_run(
    artifact_dir: Path,
    *,
    scenario: str | None = None,
    cache_bust: str | None = None,
    allow_wrap: bool | None = None,
) -> BenchmarkRun:
    """Graph-workload BenchmarkRun from real config objects; a None argument means the user left that knob unset."""
    dataset: dict[str, object] = {
        "name": "profiling",
        "type": "file",
        "path": str(_GRAPH_FIXTURE),
    }
    if cache_bust is not None:
        dataset["cache_bust"] = {"target": cache_bust}
    if allow_wrap is not None:
        dataset["synthesis"] = {"allow_dataset_wrap": allow_wrap}

    cfg = BenchmarkConfig(
        models=["claude-opus-4-5-20251101"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
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
    return BenchmarkRun(benchmark_id="test-run", cfg=cfg, artifact_dir=artifact_dir)


def apply_chain(run: BenchmarkRun) -> SimpleNamespace:
    """Run the real dataset/scenario/graph-dispatch resolvers and return a view of the resolved dispatch fields."""
    DatasetResolver().resolve(run)
    ScenarioResolver().resolve(run)
    GraphDispatchResolver().resolve(run)
    return SimpleNamespace(
        cache_bust=run.cfg.get_cache_bust_target(),
        allow_dataset_wrap=run.resolved.allow_dataset_wrap,
        dataset_sampling_strategy=run.resolved.dataset_sampling_strategy,
    )


@pytest.mark.parametrize(
    ("cache_bust", "allow_wrap", "expected_wrap", "expected_target"),
    [
        param(
            "first_turn_prefix",
            None,
            True,
            CacheBustTarget.FIRST_TURN_PREFIX,
            id="cache-bust-set-wrap-unset-defaults-true",
        ),
        param(None, None, False, CacheBustTarget.NONE, id="no-cache-bust-wrap-false"),
        param(
            "first_turn_prefix",
            False,
            False,
            CacheBustTarget.FIRST_TURN_PREFIX,
            id="explicit-wrap-false-beats-cache-bust",
        ),
    ],
)  # fmt: skip
def test_allow_dataset_wrap_derivation(
    tmp_path: Path,
    cache_bust: str | None,
    allow_wrap: bool | None,
    expected_wrap: bool,
    expected_target: CacheBustTarget,
) -> None:
    """Wrap defaults True under cache-bust (recycling is the point of cache-bust), but an explicit False always wins."""
    resolved = apply_chain(
        _graph_run(tmp_path, cache_bust=cache_bust, allow_wrap=allow_wrap)
    )
    assert resolved.cache_bust == expected_target
    assert resolved.allow_dataset_wrap is expected_wrap


def test_graph_dispatch_resolver_runs_after_scenario_resolver() -> None:
    """The default chain orders GraphDispatchResolver after ScenarioResolver so scenario auto-fills are visible to it."""
    types = [type(r) for r in build_default_resolver_chain()._resolvers]
    assert GraphDispatchResolver in types
    assert types.index(ScenarioResolver) < types.index(GraphDispatchResolver)


def test_publishes_dataset_sampling_strategy_for_graph_workload(tmp_path: Path) -> None:
    """A graph workload resolves to sequential dataset sampling."""
    from aiperf.plugin.enums import DatasetSamplingStrategy

    resolved = apply_chain(_graph_run(tmp_path, scenario=None))
    assert resolved.dataset_sampling_strategy == DatasetSamplingStrategy.SEQUENTIAL


@pytest.mark.parametrize(
    "target",
    [
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
        CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
    ],
)
def test_rejects_warmup_isolation_for_auto_detected_graph_workload(
    tmp_path: Path, target: CacheBustTarget
) -> None:
    """Auto-detected graph runs reject targets the graph payload path cannot isolate."""
    run = _graph_run(tmp_path, cache_bust=str(target))
    DatasetResolver().resolve(run)
    ScenarioResolver().resolve(run)

    with pytest.raises(ValueError, match="not compatible with agent_graph"):
        GraphDispatchResolver().resolve(run)
