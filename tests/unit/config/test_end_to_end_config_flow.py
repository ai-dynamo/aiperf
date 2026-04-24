# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end config flow tests.

Focuses on:
- Full pipeline: YAML string -> AIPerfConfig -> BenchmarkPlan
- JSON round-trip serialization for subprocess communication
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.enums import GPUTelemetryMode
from aiperf.config import (
    AIPerfConfig,
    BenchmarkConfig,
    BenchmarkPlan,
    BenchmarkRun,
    ResolvedConfig,
    SweepVariation,
)
from aiperf.config.dataset import FileDataset
from aiperf.config.loader import (
    ConfigurationError,
    MissingEnvironmentVariableError,
    build_benchmark_plan,
    load_config_from_string,
)
from aiperf.config.sweep import expand_sweep

# ============================================================
# Shared YAML Helpers
# ============================================================

_MINIMAL_YAML = textwrap.dedent("""\
    models:
      - test-model
    endpoint:
      urls:
        - http://localhost:8000/v1/chat/completions
    datasets:
      default:
        type: synthetic
        entries: 100
        prompts:
          isl: 128
          osl: 64
    phases:
      default:
        type: concurrency
        requests: 10
        concurrency: 1
""")

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


def _yaml_to_plan(yaml_str: str) -> BenchmarkPlan:
    """Parse YAML string into a BenchmarkPlan via AIPerfConfig."""
    config = load_config_from_string(yaml_str, substitute_env=True)
    return build_benchmark_plan(config)


# ============================================================
# Class 1: TestYamlToBenchmarkPlan
# ============================================================


class TestYamlToBenchmarkPlan:
    """Full pipeline: YAML string -> AIPerfConfig -> BenchmarkPlan."""

    def test_minimal_yaml_produces_single_run_plan(self) -> None:
        plan = _yaml_to_plan(_MINIMAL_YAML)

        assert plan.is_single_run
        assert len(plan.configs) == 1
        assert len(plan.variations) == 1
        assert plan.variations[0].label == "base"

    def test_yaml_with_grid_sweep_expands_correctly(self) -> None:
        yaml_str = _MINIMAL_YAML + textwrap.dedent("""\
            sweep:
              type: grid
              variables:
                phases.default.concurrency:
                  - 8
                  - 16
                phases.default.requests:
                  - 100
                  - 200
                  - 300
        """)
        plan = _yaml_to_plan(yaml_str)

        assert len(plan.configs) == 6
        assert len(plan.variations) == 6

        concurrency_request_pairs = {
            (c.phases["default"].concurrency, c.phases["default"].requests)
            for c in plan.configs
        }
        assert concurrency_request_pairs == {
            (8, 100),
            (8, 200),
            (8, 300),
            (16, 100),
            (16, 200),
            (16, 300),
        }

    def test_yaml_with_scenario_sweep_preserves_base_fields(self) -> None:
        yaml_str = _MINIMAL_YAML + textwrap.dedent("""\
            sweep:
              type: scenarios
              runs:
                - name: low-concurrency
                  phases:
                    default:
                      concurrency: 2
                - name: high-concurrency
                  phases:
                    default:
                      concurrency: 64
        """)
        plan = _yaml_to_plan(yaml_str)

        assert len(plan.configs) == 2
        assert plan.variations[0].label == "low-concurrency"
        assert plan.variations[1].label == "high-concurrency"

        assert plan.configs[0].phases["default"].concurrency == 2
        assert plan.configs[1].phases["default"].concurrency == 64

        # Base fields preserved in both variations
        for cfg in plan.configs:
            assert cfg.get_model_names() == ["test-model"]
            assert cfg.phases["default"].requests == 10

    def test_yaml_with_multi_run_sets_plan_fields(self) -> None:
        yaml_str = _MINIMAL_YAML + textwrap.dedent("""\
            multi_run:
              num_runs: 5
              cooldown_seconds: 2.5
              confidence_level: 0.99
        """)
        plan = _yaml_to_plan(yaml_str)

        assert plan.trials == 5
        assert plan.cooldown_seconds == 2.5
        assert plan.confidence_level == 0.99
        assert not plan.is_single_run

    def test_yaml_with_sweep_and_multi_run_combined(self) -> None:
        yaml_str = _MINIMAL_YAML + textwrap.dedent("""\
            sweep:
              type: grid
              variables:
                phases.default.concurrency:
                  - 8
                  - 16
                  - 32
            multi_run:
              num_runs: 3
        """)
        plan = _yaml_to_plan(yaml_str)

        assert len(plan.configs) == 3
        assert plan.trials == 3
        assert not plan.is_single_run

    def test_yaml_with_magic_lists_auto_expands(self) -> None:
        # Magic lists are detected by expand_sweep on raw dicts before Pydantic
        # validation. AIPerfConfig rejects list-valued concurrency, so we test
        # via the dict-level expand_sweep API which is the actual detection path.
        import yaml

        yaml_str = textwrap.dedent("""\
            models:
              - test-model
            endpoint:
              urls:
                - http://localhost:8000/v1/chat/completions
            datasets:
              default:
                type: synthetic
                entries: 100
                prompts:
                  isl: 128
                  osl: 64
            phases:
              default:
                type: concurrency
                requests: 10
                concurrency:
                  - 8
                  - 16
                  - 32
        """)
        data = yaml.safe_load(yaml_str)
        variations = expand_sweep(data)

        assert len(variations) == 3
        concurrencies = [v[0]["phases"]["default"]["concurrency"] for v in variations]
        assert concurrencies == [8, 16, 32]

    def test_sweep_field_stripped_from_expanded_configs(self) -> None:
        yaml_str = _MINIMAL_YAML + textwrap.dedent("""\
            sweep:
              type: grid
              variables:
                phases.default.concurrency:
                  - 4
                  - 8
            multi_run:
              num_runs: 2
        """)
        plan = _yaml_to_plan(yaml_str)

        for cfg in plan.configs:
            assert not hasattr(cfg, "sweep") or not isinstance(cfg, AIPerfConfig)
            assert not hasattr(cfg, "multi_run") or not isinstance(cfg, AIPerfConfig)

    def test_expanded_configs_are_benchmark_config_not_aiperf_config(self) -> None:
        yaml_str = _MINIMAL_YAML + textwrap.dedent("""\
            sweep:
              type: grid
              variables:
                phases.default.concurrency:
                  - 1
        """)
        plan = _yaml_to_plan(yaml_str)

        for cfg in plan.configs:
            assert isinstance(cfg, BenchmarkConfig)
            assert not isinstance(cfg, AIPerfConfig)

    def test_env_var_substitution_in_yaml(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_URL", "http://gpu-server:9000/v1/chat/completions")
        yaml_str = textwrap.dedent("""\
            models:
              - test-model
            endpoint:
              urls:
                - ${TEST_URL}
            datasets:
              default:
                type: synthetic
                entries: 100
                prompts:
                  isl: 128
                  osl: 64
            phases:
              default:
                type: concurrency
                requests: 10
                concurrency: 1
        """)
        plan = _yaml_to_plan(yaml_str)

        assert (
            plan.configs[0].endpoint.urls[0]
            == "http://gpu-server:9000/v1/chat/completions"
        )

    def test_env_var_with_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_VAR", raising=False)
        yaml_str = textwrap.dedent("""\
            models:
              - test-model
            endpoint:
              urls:
                - ${MISSING_VAR:http://fallback:8000/v1/chat/completions}
            datasets:
              default:
                type: synthetic
                entries: 100
                prompts:
                  isl: 128
                  osl: 64
            phases:
              default:
                type: concurrency
                requests: 10
                concurrency: 1
        """)
        plan = _yaml_to_plan(yaml_str)

        assert (
            plan.configs[0].endpoint.urls[0]
            == "http://fallback:8000/v1/chat/completions"
        )

    def test_env_var_missing_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("REQUIRED_VAR", raising=False)
        yaml_str = textwrap.dedent("""\
            models:
              - test-model
            endpoint:
              urls:
                - ${REQUIRED_VAR}
            datasets:
              default:
                type: synthetic
                entries: 100
                prompts:
                  isl: 128
                  osl: 64
            phases:
              default:
                type: concurrency
                requests: 10
                concurrency: 1
        """)
        with pytest.raises(MissingEnvironmentVariableError, match="REQUIRED_VAR"):
            _yaml_to_plan(yaml_str)

    def test_yaml_parse_error_raises_configuration_error(self) -> None:
        bad_yaml = "models: [unclosed bracket"
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            _yaml_to_plan(bad_yaml)

    def test_empty_yaml_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="empty"):
            _yaml_to_plan("")

    def test_yaml_list_instead_of_dict_raises_configuration_error(self) -> None:
        yaml_str = "- item1\n- item2\n"
        with pytest.raises(ConfigurationError, match="mapping"):
            _yaml_to_plan(yaml_str)


# ============================================================
# Class 2: TestBenchmarkRunSerialization
# ============================================================


class TestBenchmarkRunSerialization:
    """JSON round-trip tests for subprocess communication."""

    def _make_run(self, **overrides) -> BenchmarkRun:
        """Build a BenchmarkRun with sensible defaults, accepting overrides."""
        defaults = {
            "benchmark_id": "test-run-001",
            "cfg": BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS),
            "artifact_dir": Path("/tmp/artifacts/test-run-001"),
            "label": "run_0001",
        }
        defaults.update(overrides)
        return BenchmarkRun(**defaults)

    def _round_trip(self, run: BenchmarkRun) -> BenchmarkRun:
        """Serialize to JSON bytes and back via model_validate."""
        json_bytes = orjson.dumps(run.model_dump(mode="json", exclude_none=True))
        data = orjson.loads(json_bytes)
        return BenchmarkRun.model_validate(data)

    def test_json_round_trip_minimal_run(self) -> None:
        original = self._make_run()
        restored = self._round_trip(original)

        assert restored.benchmark_id == original.benchmark_id
        assert restored.trial == original.trial
        assert restored.label == original.label
        assert restored.cfg.get_model_names() == ["test-model"]

    def test_json_round_trip_with_sweep_variation(self) -> None:
        variation = SweepVariation(
            index=2,
            label="concurrency=32",
            values={"phases.default.concurrency": 32},
        )
        original = self._make_run(variation=variation, trial=1)
        restored = self._round_trip(original)

        assert restored.variation is not None
        assert restored.variation.index == 2
        assert restored.variation.label == "concurrency=32"
        assert restored.variation.values == {"phases.default.concurrency": 32}

    def test_json_round_trip_with_resolved_config(self) -> None:
        resolved = ResolvedConfig(
            tokenizer_names={
                "test-model": "hf-internal-testing/tiny-random-LlamaForCausalLM"
            },
            gpu_telemetry_mode=GPUTelemetryMode.REALTIME_DASHBOARD,
            artifact_dir_created=True,
            total_expected_duration=300.0,
        )
        original = self._make_run(resolved=resolved)
        restored = self._round_trip(original)

        assert restored.resolved.tokenizer_names == {
            "test-model": "hf-internal-testing/tiny-random-LlamaForCausalLM"
        }
        assert (
            restored.resolved.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD
        )
        assert restored.resolved.artifact_dir_created is True
        assert restored.resolved.total_expected_duration == 300.0

    def test_json_round_trip_preserves_path_types(self) -> None:
        original = self._make_run(artifact_dir=Path("/data/benchmarks/run-42"))
        restored = self._round_trip(original)

        assert isinstance(restored.artifact_dir, Path)
        assert str(restored.artifact_dir) == "/data/benchmarks/run-42"

    def test_json_round_trip_with_all_phase_types(self) -> None:
        cfg_kwargs = {
            **_MINIMAL_CONFIG_KWARGS,
            "phases": {
                "warmup": {
                    "type": "concurrency",
                    "requests": 5,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                "profiling": {
                    "type": "poisson",
                    "rate": 10.0,
                    "duration": 60,
                },
                "constant_phase": {
                    "type": "constant",
                    "rate": 5.0,
                    "requests": 50,
                },
            },
        }
        config = BenchmarkConfig(**cfg_kwargs)
        original = self._make_run(cfg=config)
        restored = self._round_trip(original)

        assert set(restored.cfg.phases.keys()) == {
            "warmup",
            "profiling",
            "constant_phase",
        }
        assert restored.cfg.phases["warmup"].type == "concurrency"
        assert restored.cfg.phases["profiling"].type == "poisson"
        assert restored.cfg.phases["constant_phase"].type == "constant"

    def test_json_round_trip_with_file_dataset(self, tmp_path: Path) -> None:
        dataset_file = tmp_path / "prompts.jsonl"
        dataset_file.write_text('{"text": "hello"}\n')

        cfg_kwargs = {
            **_MINIMAL_CONFIG_KWARGS,
            "datasets": {
                "from_file": {
                    "type": "file",
                    "path": str(dataset_file),
                },
            },
            "phases": {
                "default": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                    "dataset": "from_file",
                },
            },
        }
        config = BenchmarkConfig(**cfg_kwargs)
        original = self._make_run(cfg=config)
        restored = self._round_trip(original)

        ds = restored.cfg.datasets["from_file"]
        assert isinstance(ds, FileDataset)
        assert str(ds.path) == str(dataset_file)

    def test_json_round_trip_with_enum_fields(self) -> None:
        resolved = ResolvedConfig(
            gpu_telemetry_mode=GPUTelemetryMode.SUMMARY,
        )
        original = self._make_run(resolved=resolved)
        restored = self._round_trip(original)

        assert restored.resolved.gpu_telemetry_mode == GPUTelemetryMode.SUMMARY
        assert isinstance(restored.resolved.gpu_telemetry_mode, GPUTelemetryMode)

    def test_json_round_trip_with_none_variation(self) -> None:
        original = self._make_run(variation=None)
        restored = self._round_trip(original)

        assert restored.variation is None

    def test_json_round_trip_with_nested_config(self) -> None:
        cfg_kwargs = {
            **_MINIMAL_CONFIG_KWARGS,
            "datasets": {
                "default": {
                    "type": "synthetic",
                    "entries": 500,
                    "prompts": {"isl": 256, "osl": 128},
                },
                "secondary": {
                    "type": "synthetic",
                    "entries": 200,
                    "prompts": {"isl": 64, "osl": 32},
                },
            },
            "phases": {
                "warmup": {
                    "type": "concurrency",
                    "requests": 5,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                "profiling": {
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 8,
                    "dataset": "secondary",
                },
            },
        }
        config = BenchmarkConfig(**cfg_kwargs)
        original = self._make_run(cfg=config)
        restored = self._round_trip(original)

        assert set(restored.cfg.datasets.keys()) == {"default", "secondary"}
        assert restored.cfg.phases["profiling"].dataset == "secondary"
        assert restored.cfg.phases["profiling"].concurrency == 8

    def test_model_validate_rejects_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError, match="benchmark_id"):
            BenchmarkRun.model_validate(
                {
                    "cfg": _MINIMAL_CONFIG_KWARGS,
                    "artifact_dir": "/tmp/test",
                }
            )

        with pytest.raises(ValidationError, match="cfg"):
            BenchmarkRun.model_validate(
                {
                    "benchmark_id": "abc",
                    "artifact_dir": "/tmp/test",
                }
            )

    def test_model_validate_rejects_invalid_trial(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRun(
                benchmark_id="abc",
                cfg=BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS),
                artifact_dir=Path("/tmp/test"),
                trial=-1,
            )


# ============================================================
# Class 3: TestFlatConfigYaml
# ============================================================


class TestFlatConfigYaml:
    """End-to-end: flat YAML -> AIPerfConfig -> BenchmarkPlan."""

    def test_minimal_flat_config(self) -> None:
        yaml_str = textwrap.dedent("""\
            model: test-model
            endpoint:
              url: http://localhost:8000/v1/chat/completions
            dataset:
              isl: 512
              osl: 128
            profiling:
              type: concurrency
              requests: 10
              concurrency: 1
        """)
        config = load_config_from_string(yaml_str)
        assert "default" in config.datasets
        assert "profiling" in config.phases

    def test_warmup_profiling_flat_config(self) -> None:
        yaml_str = textwrap.dedent("""\
            model: test-model
            endpoint:
              url: http://localhost:8000/v1/chat/completions
            datasets:
              warmup:
                isl: 256
                osl: 64
                entries: 100
              profiling:
                isl: {mean: 550, stddev: 50}
                osl: {mean: 150, stddev: 25}
                entries: 500
            warmup:
              type: concurrency
              dataset: warmup
              requests: 100
              concurrency: 8
            profiling:
              type: concurrency
              dataset: profiling
              requests: 10
              concurrency: 1
        """)
        config = load_config_from_string(yaml_str)
        assert list(config.phases.keys()) == ["warmup", "profiling"]
        assert config.phases["warmup"].exclude_from_results is True
        plan = build_benchmark_plan(config)
        assert len(plan.configs) == 1
