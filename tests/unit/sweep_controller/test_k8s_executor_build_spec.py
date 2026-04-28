# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.config.benchmark import BenchmarkRun
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.sweep_controller.k8s_executor import K8sChildJobExecutor


def _sweep_cr() -> dict:
    return {
        "metadata": {"name": "test-sweep", "namespace": "default", "uid": "abc-123"},
        "spec": {
            "template": {
                "metadata": {"labels": {"team": "perf"}},
                "spec": {
                    "image": "test:latest",
                    "podTemplate": {},
                    "benchmark": {
                        "models": ["m"],
                        "endpoint": {"urls": ["http://x"], "type": "chat"},
                        "datasets": [{"name": "main", "type": "synthetic"}],
                        "phases": [
                            {
                                "name": "profiling",
                                "type": "concurrency",
                                "duration": 1,
                                "concurrency": 1,
                            }
                        ],
                    },
                },
            },
        },
    }


def _benchmark_config_for_run() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 64,
                }
            ],
        }
    )


def test_build_child_spec_overrides_benchmark():
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    run = BenchmarkRun(
        benchmark_id="x",
        cfg=_benchmark_config_for_run(),
        variation=SweepVariation(
            index=7,
            label="c=64",
            values={"phases.profiling.concurrency": 64},
        ),
        trial=2,
        label="run_0003",
        artifact_dir=Path("/results"),
    )
    spec = executor._build_child_spec(run)
    assert spec["image"] == "test:latest"
    # The variation-applied benchmark replaces the template's base benchmark.
    assert spec["benchmark"]["phases"][0]["concurrency"] == 64


def test_build_child_metadata_sets_owner_and_labels():
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    run = BenchmarkRun(
        benchmark_id="x",
        cfg=_benchmark_config_for_run(),
        variation=SweepVariation(index=7, label="c=64", values={}),
        trial=2,
        label="run_0003",
        artifact_dir=Path("/results"),
    )
    md = executor._build_child_metadata(run, "test-sweep-v07-t2")
    assert md["name"] == "test-sweep-v07-t2"
    assert md["namespace"] == "default"
    refs = md["ownerReferences"]
    assert len(refs) == 1
    assert refs[0]["uid"] == "abc-123"
    assert refs[0]["controller"] is True
    assert md["labels"]["aiperf.nvidia.com/sweep"] == "test-sweep"
    assert md["labels"]["aiperf.nvidia.com/sweep-uid"] == "abc-123"
    assert md["labels"]["aiperf.nvidia.com/variation-index"] == "07"
    assert md["labels"]["aiperf.nvidia.com/trial-index"] == "2"
    # Template metadata labels are merged
    assert md["labels"]["team"] == "perf"


def test_derive_id_uses_deterministic_naming():
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    assert executor.derive_id(plan=None, var_idx=7, trial=2) == "test-sweep-v07-t2"
