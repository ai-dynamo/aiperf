# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes routing regressions for multi-run configs without a sweep axis."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiperf.cli_commands.kube import generate as generate_cmd
from aiperf.cli_commands.kube import sweep as sweep_cmd
from aiperf.config import AIPerfConfig
from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeOptions
from aiperf.kubernetes.crd_models import AIPerfJobSpec, AIPerfSweepSpec
from aiperf.sweep_controller.plan_builder import build_plan_from_sweep

_BENCHMARK: dict[str, object] = {
    "models": ["m"],
    "endpoint": {"urls": ["http://server:8000"], "type": "chat"},
    "datasets": [
        {
            "name": "main",
            "type": "synthetic",
            "prompts": {"isl": 64, "osl": 32},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
        }
    ],
}


def _write_config(path: Path, *, multi_run: dict[str, object] | None = None) -> Path:
    envelope: dict[str, object] = {"benchmark": _BENCHMARK}
    if multi_run is not None:
        envelope["multiRun"] = multi_run
    path.write_text(yaml.safe_dump(envelope, sort_keys=False))
    return path


def _kube_options() -> KubeOptions:
    return KubeOptions(image="aiperf:test", name="routing-test")


def _sweep_kwargs() -> dict[str, object]:
    return {
        "multi_run_trials": None,
        "cooldown_seconds": 0.0,
        "convergence_metric": None,
        "convergence_min_runs": 3,
        "convergence_max_runs": 10,
        "convergence_threshold": 0.05,
    }


def test_generate_multirun_only_routes_to_one_cell_sweep(tmp_path: Path) -> None:
    config_file = _write_config(tmp_path / "repeats.yaml", multi_run={"numRuns": 3})

    spec, config, _ = generate_cmd._resolve_spec_and_name(
        CLIConfig(config_file=config_file), _kube_options()
    )

    assert generate_cmd._choose_kind(config) == "AIPerfSweep"
    assert spec["sweep"] == {
        "type": "scenarios",
        "runs": [{"name": "base"}],
    }
    validated = AIPerfSweepSpec.model_validate(spec)
    assert validated.multi_run.num_runs == 3
    plan = build_plan_from_sweep({"spec": spec})
    assert plan.trials == 3
    assert len(plan.configs) == 1
    assert plan.is_sweep is False


def test_generate_convergence_only_uses_independent_one_cell_sweep(
    tmp_path: Path,
) -> None:
    config_file = _write_config(
        tmp_path / "convergence.yaml",
        multi_run={
            "numRuns": 5,
            "convergence": {
                "metric": "request_latency",
                "minRuns": 2,
                "threshold": 0.05,
            },
        },
    )

    spec, _, _ = generate_cmd._resolve_spec_and_name(
        CLIConfig(config_file=config_file), _kube_options()
    )

    assert spec["sweep"]["iterationOrder"] == "independent"
    plan = build_plan_from_sweep({"spec": spec})
    assert plan.trials == 5
    assert plan.use_adaptive is True
    assert len(plan.configs) == 1


def test_generate_multirun_aiperfjob_input_preserves_deployment_spec(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "job.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "aiperf.nvidia.com/v1alpha1",
                "kind": "AIPerfJob",
                "metadata": {"name": "repeats"},
                "spec": {
                    "image": "original:image",
                    "benchmark": _BENCHMARK,
                    "multiRun": {"numRuns": 3},
                    "sweep": None,
                    "connectionsPerWorker": 17,
                },
            },
            sort_keys=False,
        )
    )

    spec, config, name = generate_cmd._resolve_spec_and_name(
        CLIConfig(config_file=config_file), _kube_options()
    )

    assert generate_cmd._choose_kind(config) == "AIPerfSweep"
    assert name == "routing-test"
    assert spec["image"] == "aiperf:test"
    assert spec["connectionsPerWorker"] == 17
    assert spec["sweep"]["runs"] == [{"name": "base"}]
    AIPerfSweepSpec.model_validate(spec)


@pytest.mark.parametrize("multi_run", [None, {"numRuns": 1}])
def test_generate_single_run_stays_aiperfjob(
    tmp_path: Path, multi_run: dict[str, object] | None
) -> None:
    config_file = _write_config(
        tmp_path / "single.yaml",
        multi_run=multi_run,
    )

    spec, config, _ = generate_cmd._resolve_spec_and_name(
        CLIConfig(config_file=config_file), _kube_options()
    )

    assert generate_cmd._choose_kind(config) == "AIPerfJob"
    assert "sweep" not in spec
    AIPerfJobSpec.model_validate(spec)


def test_kube_sweep_multirun_only_builds_valid_one_cell_cr(tmp_path: Path) -> None:
    config_file = _write_config(tmp_path / "trials.yaml", multi_run={"numRuns": 4})

    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_sweep_kwargs(),
    )

    assert cr["kind"] == "AIPerfSweep"
    assert cr["spec"]["sweep"] == {
        "type": "scenarios",
        "runs": [{"name": "base"}],
    }
    AIPerfSweepSpec.model_validate(cr["spec"])
    plan = build_plan_from_sweep(cr)
    assert (len(plan.configs), plan.trials) == (1, 4)


def test_aiperfjob_cr_with_multirun_is_handed_off_from_profile(
    tmp_path: Path,
) -> None:
    from aiperf.cli_commands.kube.profile import _check_config_file_for_sweep_keys

    config_file = tmp_path / "job.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "aiperf.nvidia.com/v1alpha1",
                "kind": "AIPerfJob",
                "metadata": {"name": "repeats"},
                "spec": {
                    "image": "aiperf:test",
                    "benchmark": _BENCHMARK,
                    "multiRun": {"numRuns": 3},
                },
            },
            sort_keys=False,
        )
    )

    with pytest.raises(SystemExit):
        _check_config_file_for_sweep_keys(config_file)


def test_profile_allows_explicit_single_run_multirun_defaults() -> None:
    from aiperf.cli_commands.kube.profile import _check_no_sweep_keys

    _check_no_sweep_keys({"multiRun": {"numRuns": 1}}, source="single.yaml")


def test_programmatic_config_routing_matches_yaml_names() -> None:
    config = AIPerfConfig.model_validate(
        {"benchmark": _BENCHMARK, "multiRun": {"numRuns": 2}}
    )

    assert generate_cmd._choose_kind(config) == "AIPerfSweep"
