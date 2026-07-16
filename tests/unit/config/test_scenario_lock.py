# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the inferencex-agentx-mvp scenario config-lock (adapted port).

Exercises ``aiperf.common.scenario.apply_scenario`` over a resolved
``BenchmarkRun`` carrying a weka graph dataset: auto-fills, submission validity,
and the explicit-conflict lock (with / without ``unsafe_override``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.scenario import ScenarioLockError, apply_scenario
from aiperf.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.resolution.resolvers import DatasetResolver


def _weka_config(
    *, endpoint_extra: dict | None = None, streaming=None
) -> BenchmarkConfig:
    """A BenchmarkConfig with a weka_trace graph dataset (path need not exist)."""
    endpoint: dict = {"urls": ["http://localhost:8000/v1/chat/completions"]}
    if streaming is not None:
        endpoint["streaming"] = streaming
    if endpoint_extra is not None:
        endpoint["extra"] = endpoint_extra
    return BenchmarkConfig(
        scenario="inferencex-agentx-mvp",
        models=["test-model"],
        endpoint=endpoint,
        datasets=[
            {
                "name": "profiling",
                "type": "file",
                "path": "/tmp/does-not-exist-weka-trace",
                "format": "weka_trace",
            }
        ],
        phases=[
            # No duration -> scenario auto-fills the 1800s default.
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 4,
                "requests": 100,
            }
        ],
    )


def _resolve(config: BenchmarkConfig, tmp_path: Path) -> BenchmarkRun:
    run = BenchmarkRun(
        benchmark_id="test-run", cfg=config, artifact_dir=tmp_path / "artifacts"
    )
    # DatasetResolver populates run.resolved.dataset_types, which the scenario's
    # weka-workload detection reads.
    DatasetResolver().resolve(run)
    return run


def test_scenario_autofills_and_validates(tmp_path):
    run = _resolve(_weka_config(), tmp_path)
    outcome = apply_scenario(run)

    # Auto-filled invariants.
    assert run.cfg.trajectory_start_min_ratio == 0.0
    assert run.cfg.trajectory_start_max_ratio == 1.0
    assert run.cfg.endpoint.streaming is True
    assert run.cfg.endpoint.extra["ignore_eos"] is True
    dataset = run.cfg.get_default_dataset()
    assert dataset.synthesis is not None
    assert dataset.synthesis.idle_gap_cap_seconds == 10.0
    # Duration auto-filled above the 900s floor.
    assert run.cfg.get_profiling_phases()[0].duration == 1800.0

    # Submission valid, outcome stored on run.resolved.
    assert outcome.submission_valid is True
    assert run.resolved.scenario_outcome is outcome
    assert outcome.scenario_name == "inferencex-agentx-mvp"
    assert not outcome.violations


def test_explicit_streaming_false_raises(tmp_path):
    run = _resolve(_weka_config(streaming=False), tmp_path)
    with pytest.raises(ScenarioLockError):
        apply_scenario(run)


def test_explicit_conflict_downgraded_under_unsafe_override(tmp_path):
    config = _weka_config(streaming=False)
    config.unsafe_override = True
    run = _resolve(config, tmp_path)

    outcome = apply_scenario(run)
    assert outcome.submission_valid is False
    assert "unsafe_override" in outcome.submission_invalid_reasons
    assert any(v.flag == "--streaming" for v in outcome.violations)


def test_non_weka_workload_raises(tmp_path):
    config = BenchmarkConfig(
        scenario="inferencex-agentx-mvp",
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[
            {
                "name": "profiling",
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32},
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 4,
                "requests": 100,
            }
        ],
    )
    run = _resolve(config, tmp_path)
    with pytest.raises(ScenarioLockError):
        apply_scenario(run)


def test_no_scenario_is_noop(tmp_path):
    config = BenchmarkConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[
            {
                "name": "profiling",
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32},
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 4,
                "requests": 100,
            }
        ],
    )
    run = _resolve(config, tmp_path)
    outcome = apply_scenario(run)
    assert outcome.submission_valid is None
    assert run.cfg.endpoint.extra.get("ignore_eos") is None
