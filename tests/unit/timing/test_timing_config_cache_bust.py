# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TimingConfig.cache_bust_enabled`` is read after scenario defaults apply.

The agentx scenario auto-fills ``cache_bust.target=first_turn_prefix``, and that
auto-fill is what lets a c > distinct-traces run wrap without
``--allow-dataset-wrap``. Ordering therefore matters: ``ScenarioResolver`` runs
before ``TimingConfig.from_run`` reads the target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import apply_scenario
from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.timing.config import TimingConfig

_WEKA_LOADER = "semianalysis_cc_traces_weka_with_subagents"


def _build_run(*, scenario: str | None) -> BenchmarkRun:
    body: dict[str, Any] = {
        "models": ["my-model"],
        "endpoint": {
            "urls": ["http://localhost:8000/v1/chat/completions"],
            "type": "chat",
        },
        "datasets": [{"name": "main", "type": "public", "dataset": _WEKA_LOADER}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 512,
                "duration": 1800,
            }
        ],
    }
    if scenario is not None:
        body["scenario"] = scenario
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=BenchmarkConfig.model_validate(body),
        artifact_dir=Path("/tmp/aiperf-cache-bust-order-test"),
    )


def test_cache_bust_enabled_reflects_scenario_autofill() -> None:
    run = _build_run(scenario="inferencex-agentx-mvp")
    assert run.cfg.get_cache_bust_target() == CacheBustTarget.NONE
    assert TimingConfig.from_run(run).cache_bust_enabled is False

    apply_scenario(run)

    assert run.cfg.get_cache_bust_target() == CacheBustTarget.FIRST_TURN_PREFIX
    assert TimingConfig.from_run(run).cache_bust_enabled is True


def test_cache_bust_enabled_false_without_scenario() -> None:
    run = _build_run(scenario=None)
    assert TimingConfig.from_run(run).cache_bust_enabled is False
