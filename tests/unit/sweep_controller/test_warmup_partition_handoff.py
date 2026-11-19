# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve warmup reservations across the sweep child/operator boundary."""

from pathlib import Path
from unittest.mock import MagicMock

import kopf
import orjson
import pytest

from aiperf.cli_runner._strategy import build_strategy
from aiperf.common.enums import CreditPhase
from aiperf.config import BenchmarkRun
from aiperf.config.loader import load_config_from_mapping
from aiperf.config.sweep.multi_run import ConvergenceConfig
from aiperf.kubernetes.spec_converter import build_config_envelope
from aiperf.operator.handlers.create import _build_deployment, _validate_spec
from aiperf.orchestrator.models import RunResult
from aiperf.sweep_controller.k8s_executor import (
    SWEEP_UID_LABEL,
    TRIAL_INDEX_LABEL,
    K8sChildJobExecutor,
)
from aiperf.sweep_controller.plan_builder import build_plan_from_sweep
from aiperf.timing.config import TimingConfig


@pytest.fixture
def partitioned_sweep() -> dict:
    """Declare one variation with two trials and a real initial warmup."""
    return {
        "metadata": {"name": "partitioned", "namespace": "default", "uid": "parent"},
        "spec": {
            "image": "test:latest",
            "randomSeed": 42,
            "multiRun": {"numRuns": 2},
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1]},
            },
            "benchmark": {
                "models": ["test"],
                "endpoint": {"urls": ["http://localhost:8000"]},
                "dataset": {"type": "synthetic", "entries": 6, "warmup_entries": 2},
                "phases": [
                    {
                        "name": "warmup",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 5,
                        "exclude_from_results": True,
                    },
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 9,
                    },
                ],
            },
        },
    }


def _child_body(executor: K8sChildJobExecutor, run: BenchmarkRun) -> dict:
    """Round-trip the exact child spec and lineage emitted by the executor."""
    metadata = executor._build_child_metadata(run, f"trial-{run.trial}")
    metadata["uid"] = f"child-{run.trial}"
    return orjson.loads(
        orjson.dumps({"metadata": metadata, "spec": executor._build_child_spec(run)})
    )


@pytest.mark.parametrize("adaptive", [False, True])
def test_partition_survives_child_validation_and_serialized_deployment(
    partitioned_sweep: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adaptive: bool,
) -> None:
    """Exercise the real fixed/convergence child-to-operator config handoff."""
    monkeypatch.setattr(
        "aiperf.operator.handlers.create.events.spec_valid", MagicMock()
    )
    plan = build_plan_from_sweep(partitioned_sweep)
    if adaptive:
        plan.multi_run.convergence = ConvergenceConfig(metric="time_to_first_token")
    strategy = build_strategy(plan, MagicMock())
    executor = K8sChildJobExecutor(
        api=None, sweep=partitioned_sweep, with_trial_suffix=True
    )
    results = []
    for trial in range(2):
        cfg = strategy.get_next_config(plan.configs[0], results)
        run = BenchmarkRun(
            benchmark_id=f"trial-{trial}",
            cfg=cfg,
            trial=trial,
            variation=plan.variations[0],
            artifact_dir=tmp_path / str(trial),
            random_seed=42,
        )
        body = kopf.Body(_child_body(executor, run))
        validated = _validate_spec(body.spec, body, MagicMock())
        assert validated.benchmark.get_default_dataset().warmup_entries == 2
        deployment, _ = _build_deployment(
            body.spec,
            f"trial-{trial}",
            "default",
            f"trial-{trial}",
            body=body,
        )
        config_map = deployment.get_configmap_spec()
        assert deployment.get_jobset_spec().to_k8s_manifest()["kind"] == "JobSet"
        restored = BenchmarkRun.model_validate_json(config_map.data["run_config.json"])
        timing = TimingConfig.from_run(restored)
        assert timing.warmup_dataset_entries == 2
        assert [p.phase for p in timing.phase_configs].count(CreditPhase.WARMUP) == (
            1 if trial == 0 else 0
        )
        results.append(RunResult(label=f"trial-{trial}", success=True))


@pytest.fixture
def skipped_warmup_child(partitioned_sweep: dict, tmp_path: Path) -> dict:
    """Generate the second child after the existing strategy strips warmup."""
    plan = build_plan_from_sweep(partitioned_sweep)
    cfg = build_strategy(plan, MagicMock()).get_next_config(
        plan.configs[0], [RunResult(label="first", success=True)]
    )
    executor = K8sChildJobExecutor(
        api=None, sweep=partitioned_sweep, with_trial_suffix=True
    )
    return _child_body(
        executor,
        BenchmarkRun(
            benchmark_id="second",
            cfg=cfg,
            trial=1,
            artifact_dir=tmp_path,
        ),
    )


@pytest.mark.parametrize("lineage", ["none", "first-trial", "wrong-owner"])
def test_authored_or_unmatched_job_cannot_bypass_warmup(
    skipped_warmup_child: dict, monkeypatch: pytest.MonkeyPatch, lineage: str
) -> None:
    """Do not infer derived-trial eligibility from the benchmark payload alone."""
    monkeypatch.setattr(
        "aiperf.operator.handlers.create.events.spec_invalid", MagicMock()
    )
    body = skipped_warmup_child
    if lineage == "none":
        body["metadata"] = {}
    elif lineage == "first-trial":
        body["metadata"]["labels"][TRIAL_INDEX_LABEL] = "0"
    else:
        body["metadata"]["labels"][SWEEP_UID_LABEL] = "other"
    with pytest.raises(kopf.PermanentError, match="requires a warmup phase"):
        _validate_spec(body["spec"], body, MagicMock())
    with pytest.raises(ValueError, match="requires a warmup phase"):
        load_config_from_mapping(build_config_envelope(body["spec"]))


@pytest.mark.parametrize("invalid", ["accuracy", "pool-size"])
def test_derived_trial_retains_other_partition_guards(
    skipped_warmup_child: dict, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    """The lineage exemption only removes the already-completed warmup requirement."""
    monkeypatch.setattr(
        "aiperf.operator.handlers.create.events.spec_invalid", MagicMock()
    )
    benchmark = skipped_warmup_child["spec"]["benchmark"]
    if invalid == "accuracy":
        benchmark["accuracy"] = {"benchmark": "mmlu"}
        match = "accuracy"
    else:
        benchmark["datasets"][0]["warmupEntries"] = 6
        match = "smaller than"
    with pytest.raises(kopf.PermanentError, match=match):
        _validate_spec(skipped_warmup_child["spec"], skipped_warmup_child, MagicMock())
