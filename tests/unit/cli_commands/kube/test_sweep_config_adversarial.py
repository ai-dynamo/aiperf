# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes sweep config and child building.

Focuses on:
- AIPerfSweep CR construction from kube sweep YAML and AIPerfJob CR input.
- Adaptive-search and convergence validator boundaries that fail before submit.
- Child metadata passthrough, reserved selector-label ownership, and name budgets.

Out of scope:
- Live CustomObjectsApi submission; covered by sibling kube CLI and sweep-controller tests.
- End-to-end sweep execution; covered by sweep-controller integration paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.cli_commands.kube import sweep as sweep_cmd
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeOptions
from aiperf.config.sweep import SweepVariation
from aiperf.kubernetes.crd_models import AIPerfSweepSpec
from aiperf.sweep_controller._naming import build_child_name
from aiperf.sweep_controller.k8s_executor import (
    SWEEP_LABEL,
    SWEEP_RUN_EPOCH_LABEL,
    SWEEP_UID_LABEL,
    TRIAL_INDEX_LABEL,
    VARIATION_INDEX_LABEL,
    VARIATION_LABEL_LABEL,
    VARIATION_VALUES_ANNOTATION,
    K8sChildJobExecutor,
)

# ============================================================================
# Helpers
# ============================================================================

_MIN_BENCHMARK_YAML = """\
models: [meta-llama/Llama-3-8B]
endpoint: {urls: [http://localhost:8000/v1], type: chat, streaming: true}
datasets: [{name: sharegpt-main, type: synthetic, prompts: {isl: 64, osl: 32}}]
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
"""


def _kube_options(**overrides: object) -> KubeOptions:
    base = {"image": "nvcr.io/nvidia/aiperf:branch-kube-sweep"}
    base.update(overrides)
    return KubeOptions(**base)


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "multi_run_trials": None,
        "cooldown_seconds": 0.0,
        "convergence_metric": None,
        "convergence_min_runs": 3,
        "convergence_max_runs": 10,
        "convergence_threshold": 0.05,
    }
    base.update(overrides)
    return base


def _write_config(tmp_path: Path, name: str, content: str) -> Path:
    config_file = tmp_path / name
    config_file.write_text(content)
    return config_file


def _yaml_with(extra: str = "") -> str:
    return _MIN_BENCHMARK_YAML + extra


def _build_cr(config_file: Path, **overrides: object) -> dict[str, object]:
    return sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(**overrides),
    )


def _benchmark_run() -> BenchmarkRun:
    benchmark = BenchmarkConfig.model_validate(
        {
            "models": ["meta-llama/Llama-3-8B"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1"],
                "type": "chat",
                "streaming": True,
            },
            "datasets": [
                {
                    "name": "sharegpt-main",
                    "type": "synthetic",
                    "prompts": {"isl": 64, "osl": 32},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 32,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id="sweep-conc-demo-v03-t7",
        cfg=benchmark,
        variation=SweepVariation(
            index=3,
            label="Concurrency / 32 + TTFT SLA",
            values={"phases.profiling.concurrency": 32},
        ),
        trial=7,
        artifact_dir=Path("/results/aiperf-benchmarks/sweep-conc-demo-v03-t7"),
        label="concurrency_32_trial_7",
        cli_command=None,
    )


# ============================================================================
# build_sweep_cr_dict — round-trip validation and adaptive constraints
# ============================================================================


class TestBuildSweepCrDictRoundTrip:
    """Validate emitted AIPerfSweep CRs against the real Pydantic model."""

    def test_build_sweep_cr_dict_grid_output_round_trips_as_aiperfsweep_spec(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "latency-grid.yaml",
            _yaml_with(
                """\
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [1, 16]
multi_run:
  num_runs: 2
  cooldown_seconds: 0.5
"""
            ),
        )

        cr = _build_cr(config_file)
        spec = AIPerfSweepSpec.model_validate(cr["spec"])

        assert cr["kind"] == "AIPerfSweep"
        assert cr["metadata"] == {"name": "latency-grid-sweep"}
        assert spec.sweep is not None
        assert spec.multi_run.num_runs == 2
        assert spec.multi_run.cooldown_seconds == 0.5

    def test_build_sweep_cr_dict_adaptive_output_preserves_discriminator_for_round_trip(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "adaptive-throughput.yaml",
            _yaml_with(
                """\
sweep:
  type: adaptive_search
  planner: bayesian
  search_space:
    - {path: phases.profiling.concurrency, lo: 1, hi: 256, kind: int}
  objectives:
    - {metric: output_token_throughput, stat: avg, direction: maximize}
  max_iterations: 8
  n_initial_points: 3
"""
            ),
        )

        cr = _build_cr(config_file)
        sweep = cr["spec"]["sweep"]
        spec = AIPerfSweepSpec.model_validate(cr["spec"])

        assert sweep["type"] == "adaptive_search"
        assert spec.sweep is not None
        assert spec.sweep.max_iterations == 8
        assert spec.sweep.search_space[0].path == "phases.profiling.concurrency"


class TestBuildSweepCrDictConvergenceCompatibility:
    """Convergence and iteration-order combinations accepted by the CR builder."""

    def test_build_sweep_cr_dict_repeated_iteration_with_convergence_rejects(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "repeated-convergence.yaml",
            _yaml_with(
                """\
sweep:
  type: grid
  iteration_order: repeated
  parameters: {phases.profiling.concurrency: [1, 2]}
"""
            ),
        )

        with pytest.raises(
            ValueError, match=r"iteration_order='repeated'.*convergence"
        ):
            _build_cr(
                config_file,
                convergence_metric="time_to_first_token",
                convergence_min_runs=2,
                convergence_max_runs=4,
            )

    def test_build_sweep_cr_dict_independent_iteration_with_convergence_accepts(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "independent-convergence.yaml",
            _yaml_with(
                """\
sweep:
  type: grid
  iteration_order: independent
  parameters: {phases.profiling.concurrency: [1, 2]}
"""
            ),
        )

        cr = _build_cr(
            config_file,
            convergence_metric="time_to_first_token",
            convergence_min_runs=2,
            convergence_max_runs=4,
        )

        assert cr["spec"]["sweep"]["iterationOrder"] == "independent"
        assert cr["spec"]["multiRun"]["convergence"]["metric"] == "time_to_first_token"
        assert (
            AIPerfSweepSpec.model_validate(cr["spec"]).multi_run.convergence is not None
        )

    def test_build_sweep_cr_dict_convergence_min_runs_above_num_runs_rejects(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "bad-convergence-min-runs.yaml",
            _yaml_with(
                """\
sweep:
  type: grid
  iteration_order: independent
  parameters: {phases.profiling.concurrency: [1, 2]}
"""
            ),
        )

        with pytest.raises(
            ValueError, match=r"convergence\.min_runs \(5\).*num_runs \(4\)"
        ):
            _build_cr(
                config_file,
                convergence_metric="time_to_first_token",
                convergence_min_runs=5,
                convergence_max_runs=4,
            )

    def test_build_sweep_cr_dict_adaptive_search_rejects_iteration_order_field(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "adaptive-with-grid-knob.yaml",
            _yaml_with(
                """\
sweep:
  type: adaptive_search
  iteration_order: independent
  search_space:
    - {path: phases.profiling.concurrency, lo: 1, hi: 64, kind: int}
  objectives:
    - {metric: output_token_throughput, stat: avg, direction: maximize}
  max_iterations: 8
  n_initial_points: 3
"""
            ),
        )

        with pytest.raises(ValidationError, match=r"iteration_order|extra|forbid"):
            _build_cr(config_file)


class TestBuildSweepCrDictAdaptiveValidation:
    """Adaptive-search constraints fail at config build time, not in the controller."""

    @pytest.mark.parametrize(
        "n_initial_points,max_iterations,ok",
        [
            (1, 2, True),
            param(2, 2, False, id="initial-points-equal-max-rejected"),
            param(3, 2, False, id="initial-points-above-max-rejected"),
        ],
    )  # fmt: skip
    def test_build_sweep_cr_dict_adaptive_initial_points_boundary_enforced(
        self, tmp_path: Path, n_initial_points: int, max_iterations: int, ok: bool
    ) -> None:
        config_file = _write_config(
            tmp_path,
            f"adaptive-init-{n_initial_points}-{max_iterations}.yaml",
            _yaml_with(
                f"""\
sweep:
  type: adaptive_search
  planner: bayesian
  search_space:
    - {{path: phases.profiling.concurrency, lo: 1, hi: 64, kind: int}}
  objectives:
    - {{metric: output_token_throughput, stat: avg, direction: maximize}}
  max_iterations: {max_iterations}
  n_initial_points: {n_initial_points}
"""
            ),
        )

        if ok:
            assert (
                _build_cr(config_file)["spec"]["sweep"]["maxIterations"]
                == max_iterations
            )
            return
        with pytest.raises(ValueError, match=r"n_initial_points.*max_iterations"):
            _build_cr(config_file)

    def test_build_sweep_cr_dict_adaptive_duplicate_search_paths_rejects(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "adaptive-duplicate-paths.yaml",
            _yaml_with(
                """\
sweep:
  type: adaptive_search
  planner: bayesian
  search_space:
    - {path: phases.profiling.concurrency, lo: 1, hi: 64, kind: int}
    - {path: phases.profiling.concurrency, lo: 65, hi: 256, kind: int}
  objectives:
    - {metric: output_token_throughput, stat: avg, direction: maximize}
  max_iterations: 8
  n_initial_points: 3
"""
            ),
        )

        with pytest.raises(
            ValueError, match=r"unique.*paths.*phases\.profiling\.concurrency"
        ):
            _build_cr(config_file)


# ============================================================================
# build_sweep_cr_dict — invalid cardinality and child metadata passthrough
# ============================================================================


class TestBuildSweepCrDictCardinality:
    """Reject variation/trial cardinalities that would create invalid plans."""

    @pytest.mark.parametrize(
        "trial_count,ok",
        [
            param(0, False, id="trials-zero-rejected"),
            (1, True),
            param(10, True, id="trials-max-boundary-accepted"),
            param(11, False, id="trials-eleven-rejected"),
        ],
    )  # fmt: skip
    def test_build_sweep_cr_dict_multi_run_trials_bounds_enforced(
        self, tmp_path: Path, trial_count: int, ok: bool
    ) -> None:
        config_file = _write_config(
            tmp_path,
            f"trials-{trial_count}.yaml",
            _yaml_with(
                """\
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2]}
"""
            ),
        )

        if ok:
            cr = _build_cr(config_file, multi_run_trials=trial_count)
            spec = AIPerfSweepSpec.model_validate(cr["spec"])
            assert spec.multi_run.num_runs == trial_count
            return
        with pytest.raises(
            ValidationError,
            match=r"numRuns|num_runs|greater than or equal|less than or equal",
        ):
            _build_cr(config_file, multi_run_trials=trial_count)

    def test_build_sweep_cr_dict_empty_grid_value_list_rejects_zero_variations(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "empty-variation-axis.yaml",
            _yaml_with(
                """\
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: []
"""
            ),
        )

        with pytest.raises(
            ValueError, match=r"phases\.profiling\.concurrency.*non-empty"
        ):
            _build_cr(config_file)

    def test_build_sweep_cr_dict_aiperfjob_cr_child_metadata_passthrough_preserved(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "job-cr-with-child-metadata.yaml",
            """\
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata: {name: latency-grid-template}
spec:
  childMetadata:
    labels:
      team.nvidia.com/owner: perf-lab
    annotations:
      runbook.nvidia.com/url: https://runbooks.example.nvidia.com/aiperf-sweep
  benchmark:
    models: [meta-llama/Llama-3-8B]
    endpoint: {urls: [http://localhost:8000/v1], type: chat, streaming: true}
    datasets: [{name: sharegpt-main, type: synthetic, prompts: {isl: 64, osl: 32}}]
    phases:
      - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
  sweep:
    type: grid
    parameters: {phases.profiling.concurrency: [1, 2]}
""",
        )

        cr = _build_cr(config_file)

        assert cr["spec"]["childMetadata"] == {
            "labels": {"team.nvidia.com/owner": "perf-lab"},
            "annotations": {
                "runbook.nvidia.com/url": "https://runbooks.example.nvidia.com/aiperf-sweep"
            },
        }

    def test_aiperfjob_cr_keeps_deployment_and_applies_benchmark_cli_overrides(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path,
            "job-cr-parity.yaml",
            """\
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata: {name: latency-grid-template}
spec:
  image: yaml:image
  resourceMode: none
  podTemplate:
    affinity: {nodeAffinity: {}}
    nodeSelector: {region: west}
  benchmark:
    models: [meta-llama/Llama-3-8B]
    endpoint: {urls: [http://localhost:8000/v1], type: chat, streaming: true}
    datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
    phases:
      - {name: profiling, type: concurrency, requests: 10, concurrency: 12}
  sweep:
    type: grid
    parameters: {phases.profiling.concurrency: [1, 2]}
""",
        )

        cr = sweep_cmd._build_sweep_cr_dict(
            config_file=config_file,
            cli_config=CLIConfig(config_file=config_file, request_count=37),
            kube_options=_kube_options(
                total_workers=6,
                node_selector_cli=["gpu=true"],
            ),
            **_kwargs(),
        )

        spec = cr["spec"]
        assert spec["resourceMode"] == "none"
        assert spec["benchmark"]["phases"][0]["requests"] == 37
        assert spec["benchmark"]["runtime"]["workers"] == 6
        assert spec["podTemplate"]["affinity"] == {"nodeAffinity": {}}
        assert spec["podTemplate"]["nodeSelector"] == {
            "region": "west",
            "gpu": "true",
        }


# ============================================================================
# Child metadata and selector-label contracts
# ============================================================================


class TestChildMetadataSelectorLabels:
    """Child labels remain selector-safe even with adversarial user metadata."""

    def test_build_child_metadata_reserved_selector_labels_override_user_values(
        self,
    ) -> None:
        executor = K8sChildJobExecutor(
            api=None,
            sweep={
                "metadata": {
                    "name": "sweep-conc-demo",
                    "namespace": "aiperf-benchmarks",
                    "uid": "uid-sweep-7f2a",
                },
                "spec": {
                    "childMetadata": {
                        "labels": {
                            SWEEP_LABEL: "attacker-sweep",
                            SWEEP_UID_LABEL: "uid-attacker",
                            SWEEP_RUN_EPOCH_LABEL: "9999999999",
                            VARIATION_INDEX_LABEL: "99",
                            VARIATION_LABEL_LABEL: "wrong-label",
                            TRIAL_INDEX_LABEL: "9",
                            "team.nvidia.com/owner": "perf-lab",
                        },
                        "annotations": {
                            "runbook.nvidia.com/url": "https://runbooks.example.nvidia.com/aiperf-sweep"
                        },
                    }
                },
            },
            with_trial_suffix=True,
            sweep_run_epoch="1778027130",
        )

        metadata = executor._build_child_metadata(
            _benchmark_run(), "sweep-conc-demo-v03-t7"
        )
        labels = metadata["labels"]

        assert labels[SWEEP_LABEL] == "sweep-conc-demo"
        assert labels[SWEEP_UID_LABEL] == "uid-sweep-7f2a"
        assert labels[SWEEP_RUN_EPOCH_LABEL] == "1778027130"
        assert labels[VARIATION_INDEX_LABEL] == "03"
        assert labels[VARIATION_LABEL_LABEL] == "concurrency-32-ttft-sla"
        assert labels[TRIAL_INDEX_LABEL] == "7"
        assert labels["team.nvidia.com/owner"] == "perf-lab"
        assert metadata["annotations"]["runbook.nvidia.com/url"].endswith(
            "aiperf-sweep"
        )
        assert metadata["annotations"][VARIATION_VALUES_ANNOTATION] == (
            '{"phases.profiling.concurrency":32}'
        )


class TestChildNameCardinality:
    """Child naming enforces the variation/trial budgets documented in the helper."""

    @pytest.mark.parametrize(
        "variation_index,trial_index,match",
        [
            param(200, 0, r"variation.*200.*0\.\.199", id="variation-index-over-budget"),
            param(-1, 0, r"variation.*-1.*0\.\.199", id="variation-index-negative"),
            param(0, 10, r"trial.*10.*0\.\.9", id="trial-index-over-budget"),
            param(0, -1, r"trial.*-1.*0\.\.9", id="trial-index-negative"),
        ],
    )  # fmt: skip
    def test_build_child_name_invalid_variation_or_trial_count_rejects(
        self, variation_index: int, trial_index: int, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            build_child_name(
                sweep_name="sweep-conc-demo",
                variation_index=variation_index,
                trial_index=trial_index,
            )
