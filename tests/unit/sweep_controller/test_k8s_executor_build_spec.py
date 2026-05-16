# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.config import BenchmarkRun
from aiperf.config import BenchmarkConfig
from aiperf.config import SweepVariation
from aiperf.sweep_controller.k8s_executor import K8sChildJobExecutor


def _sweep_cr() -> dict:
    return {
        "metadata": {"name": "test-sweep", "namespace": "default", "uid": "abc-123"},
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
    # The variation-applied benchmark replaces the parent's base benchmark.
    assert spec["benchmark"]["phases"][0]["concurrency"] == 64
    # Sweep is cleared on the child so AIPerfJob doesn't try to fan out again.
    assert spec["sweep"] is None


def test_build_child_spec_strips_aiperfsweep_only_orchestration_fields():
    """sweep, convergence, failure_policy, cancel, ttl_seconds_after_finished
    are AIPerfSweep envelope-only and must NOT propagate to children.
    """
    cr = _sweep_cr()
    cr["spec"]["convergence"] = {"max_runs": 5}
    cr["spec"]["failure_policy"] = {"on_failure": "abort"}
    cr["spec"]["cancel"] = True
    cr["spec"]["ttl_seconds_after_finished"] = 60
    cr["spec"]["sweep"] = {"variables": [{"name": "c", "values": [1, 2]}]}
    executor = K8sChildJobExecutor(api=None, sweep=cr, with_trial_suffix=True)
    run = BenchmarkRun(
        benchmark_id="x",
        cfg=_benchmark_config_for_run(),
        variation=SweepVariation(index=0, label="c=1", values={}),
        trial=1,
        label="run_0001",
        artifact_dir=Path("/results"),
    )
    spec = executor._build_child_spec(run)
    for stripped in (
        "convergence",
        "failure_policy",
        "cancel",
        "ttl_seconds_after_finished",
    ):
        assert stripped not in spec, f"{stripped!r} must not propagate to child"
    assert spec["sweep"] is None


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


def test_derive_id_uses_deterministic_naming():
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    assert executor.derive_id(plan=None, var_idx=7, trial=2) == "test-sweep-v07-t2"


def test_build_child_metadata_merges_user_labels_and_annotations():
    """User-supplied childMetadata.labels/annotations propagate to children."""
    cr = _sweep_cr()
    cr["spec"]["childMetadata"] = {
        "labels": {"team": "perf", "cost-center": "ai-platform"},
        "annotations": {"runbook": "https://wiki/runbook"},
    }
    executor = K8sChildJobExecutor(
        api=None, sweep=cr, with_trial_suffix=True, sweep_run_epoch="3"
    )
    run = BenchmarkRun(
        benchmark_id="x",
        cfg=_benchmark_config_for_run(),
        variation=SweepVariation(index=1, label="c=1", values={}),
        trial=1,
        label="run_0001",
        artifact_dir=Path("/results"),
    )
    md = executor._build_child_metadata(run, "test-sweep-v01-t1")
    assert md["labels"]["team"] == "perf"
    assert md["labels"]["cost-center"] == "ai-platform"
    assert md["annotations"]["runbook"] == "https://wiki/runbook"
    # Sweep-tracking labels still authoritative.
    assert md["labels"]["aiperf.nvidia.com/sweep"] == "test-sweep"
    assert md["labels"]["aiperf.nvidia.com/sweep-uid"] == "abc-123"
    assert md["labels"]["aiperf.nvidia.com/sweep-run-epoch"] == "3"


def test_build_child_metadata_user_cannot_override_sweep_tracking_labels():
    """Sweep-tracking label keys (sweep, sweep-uid, variation-*, trial-index)
    must always be authoritative — letting users override them would silently
    break the label-selector queries that find children for status rollup.
    """
    cr = _sweep_cr()
    cr["spec"]["childMetadata"] = {
        "labels": {
            "aiperf.nvidia.com/sweep": "ATTACKER-NAME",
            "aiperf.nvidia.com/sweep-uid": "ATTACKER-UID",
            "aiperf.nvidia.com/sweep-run-epoch": "999",
            "aiperf.nvidia.com/variation-index": "99",
            "aiperf.nvidia.com/variation-label": "evil",
            "aiperf.nvidia.com/trial-index": "9",
            "user-allowed": "ok",
        },
    }
    executor = K8sChildJobExecutor(
        api=None, sweep=cr, with_trial_suffix=True, sweep_run_epoch="3"
    )
    run = BenchmarkRun(
        benchmark_id="x",
        cfg=_benchmark_config_for_run(),
        variation=SweepVariation(index=7, label="c=64", values={}),
        trial=2,
        label="run_0003",
        artifact_dir=Path("/results"),
    )
    md = executor._build_child_metadata(run, "test-sweep-v07-t2")
    assert md["labels"]["aiperf.nvidia.com/sweep"] == "test-sweep"
    assert md["labels"]["aiperf.nvidia.com/sweep-uid"] == "abc-123"
    assert md["labels"]["aiperf.nvidia.com/sweep-run-epoch"] == "3"
    assert md["labels"]["aiperf.nvidia.com/variation-index"] == "07"
    assert md["labels"]["aiperf.nvidia.com/variation-label"] == "c-64"
    assert md["labels"]["aiperf.nvidia.com/trial-index"] == "2"
    assert md["labels"]["user-allowed"] == "ok"


def test_build_child_metadata_accepts_snake_case_child_metadata():
    """Hand-built CRs / tests may use snake_case `child_metadata` instead of
    the canonical CRD camelCase `childMetadata` — both must work.
    """
    cr = _sweep_cr()
    cr["spec"]["child_metadata"] = {
        "labels": {"team": "perf"},
        "annotations": {"foo": "bar"},
    }
    executor = K8sChildJobExecutor(api=None, sweep=cr, with_trial_suffix=True)
    run = BenchmarkRun(
        benchmark_id="x",
        cfg=_benchmark_config_for_run(),
        variation=SweepVariation(index=0, label="c=1", values={}),
        trial=1,
        label="run_0001",
        artifact_dir=Path("/results"),
    )
    md = executor._build_child_metadata(run, "test-sweep-v00-t1")
    assert md["labels"]["team"] == "perf"
    assert md["annotations"] == {"foo": "bar"}


def test_build_child_metadata_no_user_metadata_keeps_empty_annotations():
    """When childMetadata is absent, annotations stay empty — no regression."""
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    run = BenchmarkRun(
        benchmark_id="x",
        cfg=_benchmark_config_for_run(),
        variation=SweepVariation(index=0, label="c=1", values={}),
        trial=1,
        label="run_0001",
        artifact_dir=Path("/results"),
    )
    md = executor._build_child_metadata(run, "test-sweep-v00-t1")
    assert md["annotations"] == {}


def test_build_child_spec_strips_child_metadata():
    """child_metadata is AIPerfSweep-only and must not propagate to children."""
    cr = _sweep_cr()
    cr["spec"]["sweep"] = {"variables": [{"name": "c", "values": [1, 2]}]}
    cr["spec"]["childMetadata"] = {"labels": {"team": "perf"}}
    executor = K8sChildJobExecutor(api=None, sweep=cr, with_trial_suffix=True)
    run = BenchmarkRun(
        benchmark_id="x",
        cfg=_benchmark_config_for_run(),
        variation=SweepVariation(index=0, label="c=1", values={}),
        trial=1,
        label="run_0001",
        artifact_dir=Path("/results"),
    )
    spec = executor._build_child_spec(run)
    assert "childMetadata" not in spec
    assert "child_metadata" not in spec
