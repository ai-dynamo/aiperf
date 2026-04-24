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
from aiperf.config.models import MultiRunConfig
from aiperf.config.sweep import SweepVariation

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": {
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    "phases": {
        "default": {"type": "concurrency", "requests": 10, "concurrency": 1},
    },
}


def _make_aiperf_config(**overrides: object) -> AIPerfConfig:
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return AIPerfConfig(**kwargs)


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

    def test_requires_at_least_one_config(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkPlan(configs=[])


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
                "variables": {"phases.default.concurrency": [8, 16, 32]},
            }
        )
        plan = build_benchmark_plan(config)

        assert len(plan.configs) == 3
        assert plan.trials == 1

        concurrencies = [c.phases["default"].concurrency for c in plan.configs]
        assert concurrencies == [8, 16, 32]

    def test_scenario_sweep(self) -> None:
        config = _make_aiperf_config(
            sweep={
                "type": "scenarios",
                "runs": [
                    {"name": "low", "phases": {"default": {"concurrency": 2}}},
                    {"name": "high", "phases": {"default": {"concurrency": 64}}},
                ],
            }
        )
        plan = build_benchmark_plan(config)

        assert len(plan.configs) == 2
        assert plan.variations[0].label == "low"
        assert plan.variations[1].label == "high"
        assert plan.configs[0].phases["default"].concurrency == 2
        assert plan.configs[1].phases["default"].concurrency == 64

    def test_sweep_with_multi_run(self) -> None:
        config = _make_aiperf_config(
            sweep={
                "type": "grid",
                "variables": {"phases.default.concurrency": [8, 16]},
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
                "variables": {"phases.default.concurrency": [8]},
            }
        )
        plan = build_benchmark_plan(config)

        for c in plan.configs:
            assert isinstance(c, BenchmarkConfig)
            assert not hasattr(c, "sweep") or not isinstance(c, AIPerfConfig)

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
    """Tests for BenchmarkConfig/AIPerfConfig inheritance."""

    @pytest.mark.parametrize(
        "attr",
        [
            param("sweep", id="sweep"),
            param("multi_run", id="multi-run"),
        ],
    )  # fmt: skip
    def test_benchmark_config_has_no_aiperf_only_field(self, attr: str) -> None:
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

    def test_aiperf_is_benchmark_config(self) -> None:
        config = _make_aiperf_config()
        assert isinstance(config, BenchmarkConfig)

    @pytest.mark.parametrize(
        "extra_field, extra_value",
        [
            param("sweep", {"type": "grid"}, id="rejects-sweep"),
            param("multi_run", {"num_runs": 3}, id="rejects-multi-run"),
        ],
    )  # fmt: skip
    def test_benchmark_config_rejects_aiperf_only_field(
        self, extra_field: str, extra_value: object
    ) -> None:
        """BenchmarkConfig with extra='forbid' rejects sweep/multi_run fields."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(**{**_MINIMAL_CONFIG_KWARGS, extra_field: extra_value})

    @pytest.mark.parametrize(
        "config_factory",
        [
            param(lambda: BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS), id="benchmark-config"),
            param(lambda: _make_aiperf_config(), id="aiperf-config"),
        ],
    )  # fmt: skip
    def test_validators_work(self, config_factory: object) -> None:
        """Validators inherited from BenchmarkConfig work on both types."""
        config = config_factory()
        assert config.get_model_names() == ["test-model"]

    def test_benchmark_config_normalizes_models(self) -> None:
        """model_validator normalizes string models to ModelsAdvanced."""
        config = _make_benchmark_config()
        assert len(config.models.items) == 1
        assert config.models.items[0].name == "test-model"

    def test_benchmark_config_benchmark_id_property(self) -> None:
        config = _make_benchmark_config()
        assert isinstance(config.benchmark_id, str)
        assert len(config.benchmark_id) > 0


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
            **_MINIMAL_CONFIG_KWARGS,
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
        path = self._write_yaml(tmp_path, _MINIMAL_CONFIG_KWARGS)
        plan = load_benchmark_plan(path, substitute_env=False)

        assert plan.trials == 1
