# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BenchmarkPlan, BenchmarkRun, and build_benchmark_plan."""

from pathlib import Path

import orjson
import pytest
import yaml
from pydantic import ValidationError
from pytest import param

from aiperf.config import (
    AIPerfConfig,
    BenchmarkConfig,
    BenchmarkPlan,
    BenchmarkRun,
)
from aiperf.config.loader import build_benchmark_plan, load_benchmark_plan
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    GridSweep,
    Objective,
    SweepVariation,
)
from aiperf.config.sweep.multi_run import ConvergenceConfig, MultiRunConfig

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {"name": "profiling", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


_ENVELOPE_KEYS = {"sweep", "multi_run", "variables", "random_seed"}


def _make_aiperf_config(**overrides: object) -> AIPerfConfig:
    env_kwargs = {k: overrides.pop(k) for k in list(overrides) if k in _ENVELOPE_KEYS}
    body = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return AIPerfConfig(benchmark=body, **env_kwargs)


def _make_benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)


# ============================================================
# BenchmarkPlan Model
# ============================================================


class TestBenchmarkPlan:
    """Tests for BenchmarkPlan model."""

    def test_basic_construction(self) -> None:
        config = _make_benchmark_config()
        plan = BenchmarkPlan(
            configs=[config],
            variations=[SweepVariation(index=0, label="base", values={})],
        )
        assert len(plan.configs) == 1
        assert plan.trials == 1
        assert plan.is_single_run

    @pytest.mark.parametrize(
        "configs_count, trials, expected",
        [
            param(1, 1, True, id="single-config-single-trial"),
            param(2, 1, False, id="multiple-configs"),
            param(1, 3, False, id="multiple-trials"),
        ],
    )  # fmt: skip
    def test_is_single_run(
        self, configs_count: int, trials: int, expected: bool
    ) -> None:
        config = _make_benchmark_config()
        plan = BenchmarkPlan(configs=[config] * configs_count, trials=trials)
        assert plan.is_single_run is expected

    def test_default_values(self) -> None:
        config = _make_benchmark_config()
        plan = BenchmarkPlan(configs=[config])
        assert plan.trials == 1
        assert plan.cooldown_seconds == 0.0
        assert plan.confidence_level == 0.95
        assert plan.set_consistent_seed is True
        assert plan.disable_warmup_after_first is True

    @pytest.mark.parametrize(
        "trials, ok",
        [
            param(1, True, id="trials-min-bound-accepted"),
            param(10, True, id="trials-upper-bound-accepted"),
            param(11, False, id="trials-11-rejected"),
            param(0, False, id="trials-zero-rejected"),
        ],
    )  # fmt: skip
    def test_trials_bounds(self, trials: int, ok: bool) -> None:
        """``BenchmarkPlan.trials`` must accept ``1..10``."""
        config = _make_benchmark_config()
        if ok:
            plan = BenchmarkPlan(configs=[config], trials=trials)
            assert plan.trials == trials
        else:
            with pytest.raises(ValidationError):
                BenchmarkPlan(configs=[config], trials=trials)

    def test_requires_at_least_one_config(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkPlan(configs=[])

    def test_is_sweep_single_config_false(self) -> None:
        config = _make_benchmark_config()
        plan = BenchmarkPlan(configs=[config])
        assert plan.is_sweep is False

    def test_is_sweep_multiple_configs_true(self) -> None:
        config = _make_benchmark_config()
        plan = BenchmarkPlan(configs=[config, config])
        assert plan.is_sweep is True

    def test_sweep_block_defaults_to_none(self) -> None:
        """Sweep envelope sub-object defaults to None for non-sweep plans."""
        config = _make_benchmark_config()
        plan = BenchmarkPlan(configs=[config])
        assert plan.sweep is None

    def test_grid_sweep_carries_cooldown_and_same_seed(self) -> None:
        """Sweep-scoped knobs (cooldown_seconds / same_seed) live on GridSweep."""
        config = _make_benchmark_config()
        plan = BenchmarkPlan(
            configs=[config, config],
            sweep=GridSweep(
                parameters={"phases.profiling.concurrency": [1, 2]},
                cooldown_seconds=2.5,
                same_seed=True,
            ),
        )
        assert isinstance(plan.sweep, GridSweep)
        assert plan.sweep.cooldown_seconds == 2.5
        assert plan.sweep.same_seed is True

    def test_grid_sweep_negative_cooldown_rejected(self) -> None:
        """GridSweep.cooldown_seconds enforces ge=0 at the sub-object."""
        with pytest.raises(ValidationError):
            GridSweep(
                parameters={"phases.profiling.concurrency": [1, 2]},
                cooldown_seconds=-1.0,
            )


# ============================================================
# BenchmarkRun Model
# ============================================================


class TestBenchmarkRun:
    """Tests for BenchmarkRun model."""

    def test_basic_construction(self) -> None:
        config = _make_benchmark_config()
        run = BenchmarkRun(
            benchmark_id="abc123",
            cfg=config,
            artifact_dir=Path("/tmp/test"),
        )
        assert run.benchmark_id == "abc123"
        assert run.trial == 0
        assert run.variation is None

    def test_with_variation(self) -> None:
        config = _make_benchmark_config()
        variation = SweepVariation(
            index=1, label="concurrency=16", values={"phases.concurrency": 16}
        )
        run = BenchmarkRun(
            benchmark_id="abc",
            cfg=config,
            variation=variation,
            trial=2,
            artifact_dir=Path("/tmp/test"),
            label="concurrency=16 / trial_0003",
        )
        assert run.variation.label == "concurrency=16"
        assert run.trial == 2
        assert run.label == "concurrency=16 / trial_0003"

    def test_json_round_trip(self) -> None:
        """Test BenchmarkRun serialization/deserialization (critical for subprocess)."""
        config = _make_benchmark_config()
        run = BenchmarkRun(
            benchmark_id="test123",
            cfg=config,
            variation=SweepVariation(index=0, label="base", values={}),
            trial=0,
            artifact_dir=Path("/tmp/artifacts"),
            label="run_0001",
        )

        json_bytes = orjson.dumps(run.model_dump(mode="json", exclude_none=True))
        data = orjson.loads(json_bytes)
        restored = BenchmarkRun.model_validate(data)

        assert restored.benchmark_id == run.benchmark_id
        assert restored.trial == run.trial
        assert restored.label == run.label
        assert str(restored.artifact_dir) == str(run.artifact_dir)
        assert restored.cfg.get_model_names() == ["test-model"]


# ============================================================
# build_benchmark_plan
# ============================================================


class TestBuildBenchmarkPlan:
    """Tests for build_benchmark_plan."""

    def test_no_sweep_no_multi_run(self) -> None:
        config = _make_aiperf_config()
        plan = build_benchmark_plan(config)

        assert len(plan.configs) == 1
        assert plan.trials == 1
        assert plan.is_single_run
        assert isinstance(plan.configs[0], BenchmarkConfig)

    def test_multi_run_only(self) -> None:
        config = _make_aiperf_config(
            multi_run={"num_runs": 3, "cooldown_seconds": 1.0, "confidence_level": 0.99}
        )
        plan = build_benchmark_plan(config)

        assert len(plan.configs) == 1
        assert plan.trials == 3
        assert plan.cooldown_seconds == 1.0
        assert plan.confidence_level == 0.99
        assert not plan.is_single_run

    def test_grid_sweep(self) -> None:
        config = _make_aiperf_config(
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [8, 16, 32]},
            }
        )
        plan = build_benchmark_plan(config)

        assert len(plan.configs) == 3
        assert plan.trials == 1

        concurrencies = [
            next(p for p in c.phases if p.name == "profiling").concurrency
            for c in plan.configs
        ]
        assert concurrencies == [8, 16, 32]

    def test_scenario_sweep(self) -> None:
        config = _make_aiperf_config(
            sweep={
                "type": "scenarios",
                "runs": [
                    {
                        "name": "low",
                        "benchmark": {
                            "phases": [{"name": "profiling", "concurrency": 2}]
                        },
                    },
                    {
                        "name": "high",
                        "benchmark": {
                            "phases": [{"name": "profiling", "concurrency": 64}]
                        },
                    },
                ],
            }
        )
        plan = build_benchmark_plan(config)

        assert len(plan.configs) == 2
        assert plan.variations[0].label == "low"
        assert plan.variations[1].label == "high"
        assert (
            next(p for p in plan.configs[0].phases if p.name == "profiling").concurrency
            == 2
        )
        assert (
            next(p for p in plan.configs[1].phases if p.name == "profiling").concurrency
            == 64
        )

    def test_sweep_with_multi_run(self) -> None:
        config = _make_aiperf_config(
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [8, 16]},
            },
            multi_run={"num_runs": 3},
        )
        plan = build_benchmark_plan(config)

        assert len(plan.configs) == 2
        assert plan.trials == 3
        assert not plan.is_single_run

    def test_configs_are_benchmark_config_not_aiperf_config(self) -> None:
        """Expanded configs should be BenchmarkConfig (no sweep/multi_run)."""
        config = _make_aiperf_config(
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [8]},
            }
        )
        plan = build_benchmark_plan(config)

        for c in plan.configs:
            assert isinstance(c, BenchmarkConfig)
            assert not isinstance(c, AIPerfConfig)

    @pytest.mark.parametrize(
        "multi_run_kwargs, plan_attr, expected",
        [
            param(
                {"num_runs": 2, "set_consistent_seed": False},
                "set_consistent_seed",
                False,
                id="set-consistent-seed-false",
            ),
            param(
                {"num_runs": 2, "disable_warmup_after_first": False},
                "disable_warmup_after_first",
                False,
                id="disable-warmup-after-first-false",
            ),
        ],
    )  # fmt: skip
    def test_multi_run_field_propagated(
        self, multi_run_kwargs: dict, plan_attr: str, expected: object
    ) -> None:
        config = _make_aiperf_config(multi_run=multi_run_kwargs)
        plan = build_benchmark_plan(config)
        assert getattr(plan, plan_attr) == expected

    def test_defaults_when_multi_run_block_is_empty(self) -> None:
        config = _make_aiperf_config(multi_run={})
        plan = build_benchmark_plan(config)

        assert plan.trials == 1
        assert plan.cooldown_seconds == 0.0
        assert plan.confidence_level == 0.95
        assert plan.set_consistent_seed is True
        assert plan.disable_warmup_after_first is True


# ============================================================
# Config Hierarchy (BenchmarkConfig / AIPerfConfig)
# ============================================================


class TestConfigHierarchy:
    """Tests for BenchmarkConfig vs. AIPerfConfig envelope split."""

    @pytest.mark.parametrize(
        "attr",
        [
            param("sweep", id="sweep"),
            param("multi_run", id="multi-run"),
        ],
    )  # fmt: skip
    def test_benchmark_config_has_no_envelope_field(self, attr: str) -> None:
        config = _make_benchmark_config()
        assert not hasattr(config, attr)

    @pytest.mark.parametrize(
        "attr, expected",
        [
            param("sweep", None, id="sweep-default-none"),
        ],
    )  # fmt: skip
    def test_aiperf_config_has_field(self, attr: str, expected: object) -> None:
        config = _make_aiperf_config()
        assert hasattr(config, attr)
        assert getattr(config, attr) == expected

    def test_aiperf_config_multi_run_default(self) -> None:
        config = _make_aiperf_config()
        assert hasattr(config, "multi_run")
        assert config.multi_run.num_runs == 1

    def test_aiperf_config_wraps_benchmark_config(self) -> None:
        """Envelope holds the body — they're peers, not subclasses."""
        config = _make_aiperf_config()
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert not isinstance(config, BenchmarkConfig)

    @pytest.mark.parametrize(
        "extra_field, extra_value",
        [
            param("sweep", {"type": "grid"}, id="rejects-sweep"),
            param("multi_run", {"num_runs": 3}, id="rejects-multi-run"),
        ],
    )  # fmt: skip
    def test_benchmark_config_rejects_envelope_field(
        self, extra_field: str, extra_value: object
    ) -> None:
        """BenchmarkConfig with extra='forbid' rejects sweep/multi_run fields."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(**{**_MINIMAL_CONFIG_KWARGS, extra_field: extra_value})

    def test_validators_work_on_benchmark_config(self) -> None:
        config = BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)
        assert config.get_model_names() == ["test-model"]

    def test_validators_work_via_envelope(self) -> None:
        config = _make_aiperf_config()
        assert config.benchmark.get_model_names() == ["test-model"]

    def test_benchmark_config_normalizes_models(self) -> None:
        """model_validator normalizes string models to ModelsAdvanced."""
        config = _make_benchmark_config()
        assert len(config.models.items) == 1
        assert config.models.items[0].name == "test-model"


# ============================================================
# MultiRunConfig Validation
# ============================================================


class TestMultiRunConfigValidation:
    """Verify MultiRunConfig field constraints and defaults."""

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_field"):
            MultiRunConfig(extra_field="nope")


# ============================================================
# YAML File Round-Trip via load_benchmark_plan
# ============================================================


class TestLoadBenchmarkPlanYAMLRoundTrip:
    """Verify YAML file loading propagates multi_run fields correctly."""

    def _write_yaml(self, tmp_path: Path, data: dict) -> Path:
        file_path = tmp_path / "benchmark.yaml"
        file_path.write_text(yaml.dump(data, default_flow_style=False))
        return file_path

    def test_full_multi_run_block_from_yaml(self, tmp_path: Path) -> None:
        yaml_data = {
            "benchmark": _MINIMAL_CONFIG_KWARGS,
            "multi_run": {
                "num_runs": 5,
                "cooldown_seconds": 2.5,
                "confidence_level": 0.99,
                "set_consistent_seed": False,
                "disable_warmup_after_first": False,
            },
        }
        path = self._write_yaml(tmp_path, yaml_data)
        plan = load_benchmark_plan(path, substitute_env=False)

        assert plan.trials == 5
        assert plan.cooldown_seconds == 2.5
        assert plan.confidence_level == 0.99
        assert plan.set_consistent_seed is False
        assert plan.disable_warmup_after_first is False

    def test_minimal_yaml_no_multi_run_defaults(self, tmp_path: Path) -> None:
        path = self._write_yaml(tmp_path, {"benchmark": _MINIMAL_CONFIG_KWARGS})
        plan = load_benchmark_plan(path, substitute_env=False)

        assert plan.trials == 1


# ============================================================
# Adversarial regression-locks: BenchmarkPlan accepts the orchestration
# sub-objects (failure_policy, multi_run with convergence) cleanly.
# ============================================================


class TestBenchmarkPlanSweepAttachments:
    """Verify BenchmarkPlan accepts failure_policy + nested convergence cleanly."""

    def test_construct_with_failure_policy_and_convergence_round_trips(self) -> None:
        """Constructor accepts failure_policy + multi_run.convergence."""
        FailurePolicy = pytest.importorskip(
            "aiperf.kubernetes.sweep_models", reason="kubernetes module not ported"
        ).FailurePolicy

        config = _make_benchmark_config()
        fp = FailurePolicy(on_child_failure="abort", max_failures=2)
        mr = MultiRunConfig(
            num_runs=5,
            convergence=ConvergenceConfig(
                metric="ttft_p99", min_runs=3, threshold=0.05
            ),
        )
        plan = BenchmarkPlan(
            configs=[config],
            variations=[
                SweepVariation(index=0, label="base", values={}),
            ],
            failure_policy=fp,
            multi_run=mr,
        )
        assert plan.failure_policy is fp
        assert plan.multi_run.convergence is not None
        assert plan.multi_run.convergence.metric == "ttft_p99"

    def test_assigning_failure_policy_after_construction_does_not_raise(self) -> None:
        """Plain assignment must not raise."""
        FailurePolicy = pytest.importorskip(
            "aiperf.kubernetes.sweep_models", reason="kubernetes module not ported"
        ).FailurePolicy

        config = _make_benchmark_config()
        plan = BenchmarkPlan(
            configs=[config],
            variations=[SweepVariation(index=0, label="base", values={})],
        )
        plan.failure_policy = FailurePolicy(on_child_failure="abort", max_failures=1)
        assert plan.failure_policy.max_failures == 1

    def test_defaults_to_none_for_non_sweep_plans(self) -> None:
        """For plans not driven by an AIPerfSweep CR, failure_policy is None and
        multi_run.convergence is None."""
        config = _make_benchmark_config()
        plan = BenchmarkPlan(
            configs=[config],
            variations=[SweepVariation(index=0, label="base", values={})],
        )
        assert plan.failure_policy is None
        assert plan.multi_run.convergence is None


# ============================================================
# BenchmarkPlan.sweep (AdaptiveSearchSweep) + is_adaptive_search property
# ============================================================


class TestBenchmarkPlanAdaptiveSearch:
    """Verify the sweep field carrying an AdaptiveSearchSweep and is_adaptive_search."""

    def test_benchmark_plan_adaptive_search_default_none(self) -> None:
        config = _make_benchmark_config()
        plan = BenchmarkPlan(
            configs=[config],
            variations=[SweepVariation(index=0, label="base", values={})],
        )
        assert plan.sweep is None
        assert plan.is_adaptive_search is False

    def _adaptive_sweep(self) -> AdaptiveSearchSweep:
        from aiperf.common.enums import OptimizationDirection
        from aiperf.config.sweep.adaptive import SearchSpaceDimension

        return AdaptiveSearchSweep(
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objectives=[
                Objective(
                    metric="m", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=20,
        )

    def test_benchmark_plan_adaptive_search_set(self) -> None:
        config = _make_benchmark_config()
        sweep = self._adaptive_sweep()
        plan = BenchmarkPlan(
            configs=[config],
            variations=[SweepVariation(index=0, label="base", values={})],
            sweep=sweep,
        )
        assert plan.sweep is sweep
        assert plan.is_adaptive_search is True
        # is_sweep stays grid-only (length-1 variations).
        assert plan.is_sweep is False

    def test_benchmark_plan_adaptive_search_is_not_single_run(self) -> None:
        """Adaptive (BO) plans must not register as single-run plans.

        Regression: search-recipe BO plans carry exactly one starting config
        (the planner mutates it across iterations), so the prior
        ``is_single_run = len==1 and trials<=1`` rule routed them through
        ``_run_single_benchmark`` in cli_runner, bypassing the BO planner
        entirely.
        """
        config = _make_benchmark_config()
        plan = BenchmarkPlan(
            configs=[config],
            variations=[SweepVariation(index=0, label="base", values={})],
            sweep=self._adaptive_sweep(),
        )
        assert plan.is_adaptive_search is True
        assert plan.is_single_run is False

    def test_benchmark_plan_repeated_mode_rejects_convergence(self) -> None:
        """REPEATED grid + multi_run.convergence is rejected at plan-build time.

        Mirrors the CEL admission rule on AIPerfSweepSpec for the k8s path.
        Trial-outer iteration order has no place to evaluate convergence
        per-cell.
        """
        from aiperf.common.enums import SweepMode

        config = _make_benchmark_config()
        with pytest.raises(
            ValidationError,
            match=r"iteration_order='repeated' is incompatible with",
        ):
            BenchmarkPlan(
                configs=[config, config],
                variations=[
                    SweepVariation(index=0, label="v0", values={}),
                    SweepVariation(index=1, label="v1", values={}),
                ],
                multi_run=MultiRunConfig(
                    num_runs=3,
                    convergence=ConvergenceConfig(metric="request_throughput"),
                ),
                sweep=GridSweep(
                    parameters={"phases.profiling.concurrency": [1, 2]},
                    iteration_order=SweepMode.REPEATED,
                ),
            )
