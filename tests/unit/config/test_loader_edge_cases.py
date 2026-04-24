# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config loader edge cases.

Focuses on:
- File-based loading (valid, missing, unreadable)
- String-based loading (valid, malformed, empty, wrong type)
- Environment variable substitution (simple, defaults, nested, errors)
- BenchmarkPlan construction from AIPerfConfig
- End-to-end file-to-plan pipeline
"""

from __future__ import annotations

import stat
import textwrap
from pathlib import Path

import pytest
from pytest import param

from aiperf.config import AIPerfConfig, BenchmarkConfig, BenchmarkPlan
from aiperf.config.loader import (
    ConfigurationError,
    MissingEnvironmentVariableError,
    build_benchmark_plan,
    load_benchmark_plan,
    load_config,
    load_config_from_string,
    substitute_env_vars,
)

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

_MINIMAL_CONFIG_KWARGS: dict = {
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


# ============================================================
# TestLoadConfig - file-based loading
# ============================================================


class TestLoadConfig:
    """Verify load_config reads YAML files and handles filesystem errors."""

    def test_load_from_valid_yaml_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(_MINIMAL_YAML)

        config = load_config(cfg_file)

        assert isinstance(config, AIPerfConfig)
        assert config.get_model_names() == ["test-model"]

    def test_load_from_nonexistent_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(missing)

    def test_load_from_unreadable_file_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "locked.yaml"
        cfg_file.write_text(_MINIMAL_YAML)
        cfg_file.chmod(0o000)

        try:
            with pytest.raises(ConfigurationError, match="Failed to read"):
                load_config(cfg_file)
        finally:
            cfg_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_load_from_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not a file"):
            load_config(tmp_path)

    def test_load_accepts_string_path(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(_MINIMAL_YAML)

        config = load_config(str(cfg_file))
        assert isinstance(config, AIPerfConfig)

    def test_load_with_env_substitution_disabled(self, tmp_path: Path) -> None:
        yaml_with_var = _MINIMAL_YAML.replace(
            "test-model", "${MODEL_NAME_THAT_DOES_NOT_EXIST}"
        )
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_with_var)

        # substitute_env=False skips env var expansion, so the raw string
        # passes through to Pydantic as the model name
        config = load_config(cfg_file, substitute_env=False)
        assert config.get_model_names() == ["${MODEL_NAME_THAT_DOES_NOT_EXIST}"]


# ============================================================
# TestLoadConfigFromString - string-based loading
# ============================================================


class TestLoadConfigFromString:
    """Verify load_config_from_string parses YAML strings correctly."""

    def test_valid_yaml_string(self) -> None:
        config = load_config_from_string(_MINIMAL_YAML)

        assert isinstance(config, AIPerfConfig)
        assert config.get_model_names() == ["test-model"]

    @pytest.mark.parametrize(
        "bad_yaml,match",
        [
            param("{{invalid", "Invalid YAML", id="broken-yaml-syntax"),
            param(":\n  :\n    - [", "Invalid YAML", id="malformed-nested"),
        ],
    )  # fmt: skip
    def test_invalid_yaml_syntax_raises(self, bad_yaml: str, match: str) -> None:
        with pytest.raises(ConfigurationError, match=match):
            load_config_from_string(bad_yaml)

    def test_null_yaml_content_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="empty"):
            load_config_from_string("null")

    def test_empty_string_content_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="empty"):
            load_config_from_string("")

    def test_list_yaml_content_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="mapping"):
            load_config_from_string("- item1\n- item2")

    def test_scalar_yaml_content_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="mapping"):
            load_config_from_string("just a string")

    def test_integer_yaml_content_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="mapping"):
            load_config_from_string("42")

    def test_file_path_included_in_error_context(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            load_config_from_string("null", file_path="/tmp/fake.yaml")
        assert "/tmp/fake.yaml" in str(exc_info.value)


# ============================================================
# TestSubstituteEnvVars - environment variable substitution
# ============================================================


class TestSubstituteEnvVars:
    """Verify recursive env var substitution with ${VAR} and ${VAR:default} syntax."""

    def test_simple_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_LOADER_VAR", "hello")
        assert substitute_env_vars("${TEST_LOADER_VAR}") == "hello"

    def test_default_value_used_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ABSENT_VAR_LOADER_TEST", raising=False)
        assert substitute_env_vars("${ABSENT_VAR_LOADER_TEST:fallback}") == "fallback"

    def test_empty_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ABSENT_VAR_LOADER_TEST", raising=False)
        assert substitute_env_vars("${ABSENT_VAR_LOADER_TEST:}") == ""

    def test_missing_required_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ABSENT_VAR_LOADER_TEST", raising=False)
        with pytest.raises(
            MissingEnvironmentVariableError, match="ABSENT_VAR_LOADER_TEST"
        ):
            substitute_env_vars("${ABSENT_VAR_LOADER_TEST}")

    def test_nested_dict_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INNER_VAL", "deep")
        result = substitute_env_vars({"a": {"b": "${INNER_VAL}"}})
        assert result == {"a": {"b": "deep"}}

    def test_list_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("V1", "alpha")
        monkeypatch.setenv("V2", "beta")
        result = substitute_env_vars(["${V1}", "${V2}"])
        assert result == ["alpha", "beta"]

    @pytest.mark.parametrize(
        "value",
        [42, True, None, 3.14],
    )  # fmt: skip
    def test_non_string_passthrough(self, value: object) -> None:
        assert substitute_env_vars(value) is value

    def test_multiple_vars_in_one_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST_VAR", "localhost")
        monkeypatch.setenv("PORT_VAR", "8080")
        result = substitute_env_vars("${HOST_VAR}:${PORT_VAR}")
        assert result == "localhost:8080"

    def test_partial_match_not_substituted(self) -> None:
        # $HOST without braces is NOT matched by the ${VAR} pattern
        assert substitute_env_vars("$HOST") == "$HOST"

    def test_env_value_takes_precedence_over_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRECEDENCE_VAR", "from_env")
        result = substitute_env_vars("${PRECEDENCE_VAR:ignored_default}")
        assert result == "from_env"

    def test_missing_var_in_nested_structure_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ABSENT_NESTED", raising=False)
        with pytest.raises(MissingEnvironmentVariableError, match="ABSENT_NESTED"):
            substitute_env_vars({"key": ["${ABSENT_NESTED}"]})

    def test_file_path_in_missing_var_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ABSENT_WITH_PATH", raising=False)
        with pytest.raises(MissingEnvironmentVariableError) as exc_info:
            substitute_env_vars("${ABSENT_WITH_PATH}", file_path="/cfg/test.yaml")
        assert "/cfg/test.yaml" in str(exc_info.value)


# ============================================================
# TestBuildBenchmarkPlan - plan construction from AIPerfConfig
# ============================================================


class TestBuildBenchmarkPlan:
    """Verify build_benchmark_plan expands sweeps and extracts multi_run settings."""

    def test_strips_sweep_from_expanded_configs(self) -> None:
        config = _make_aiperf_config(
            sweep={
                "type": "grid",
                "variables": {"phases.default.concurrency": [1, 2]},
            }
        )
        plan = build_benchmark_plan(config)

        for cfg in plan.configs:
            assert isinstance(cfg, BenchmarkConfig)
            assert not isinstance(cfg, AIPerfConfig)

    def test_strips_multi_run_from_expanded_configs(self) -> None:
        config = _make_aiperf_config(multi_run={"num_runs": 3})
        plan = build_benchmark_plan(config)

        for cfg in plan.configs:
            assert not hasattr(cfg, "multi_run") or not isinstance(cfg, AIPerfConfig)

    def test_multi_run_defaults(self) -> None:
        config = _make_aiperf_config()
        plan = build_benchmark_plan(config)

        assert plan.trials == 1
        assert plan.cooldown_seconds == 0.0
        assert plan.confidence_level == 0.95
        assert plan.set_consistent_seed is True
        assert plan.disable_warmup_after_first is True

    def test_variations_parallel_to_configs(self) -> None:
        config = _make_aiperf_config(
            sweep={
                "type": "grid",
                "variables": {"phases.default.concurrency": [4, 8, 16]},
            }
        )
        plan = build_benchmark_plan(config)

        assert len(plan.variations) == len(plan.configs)
        assert len(plan.configs) == 3

    def test_no_sweep_produces_single_base_variation(self) -> None:
        config = _make_aiperf_config()
        plan = build_benchmark_plan(config)

        assert len(plan.variations) == 1
        assert plan.variations[0].label == "base"
        assert plan.variations[0].values == {}

    def test_multi_run_values_propagated(self) -> None:
        config = _make_aiperf_config(
            multi_run={
                "num_runs": 5,
                "cooldown_seconds": 2.5,
                "confidence_level": 0.99,
                "set_consistent_seed": False,
                "disable_warmup_after_first": False,
            }
        )
        plan = build_benchmark_plan(config)

        assert plan.trials == 5
        assert plan.cooldown_seconds == 2.5
        assert plan.confidence_level == 0.99
        assert plan.set_consistent_seed is False
        assert plan.disable_warmup_after_first is False

    def test_expanded_config_preserves_model_names(self) -> None:
        config = _make_aiperf_config(
            sweep={
                "type": "grid",
                "variables": {"phases.default.concurrency": [1]},
            }
        )
        plan = build_benchmark_plan(config)

        assert plan.configs[0].get_model_names() == ["test-model"]


# ============================================================
# TestLoadBenchmarkPlan - end-to-end file to plan
# ============================================================


class TestLoadBenchmarkPlan:
    """Verify load_benchmark_plan wires file loading to plan construction."""

    def test_end_to_end_file_to_plan(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(_MINIMAL_YAML)

        plan = load_benchmark_plan(cfg_file)

        assert isinstance(plan, BenchmarkPlan)
        assert len(plan.configs) == 1
        assert plan.is_single_run

    def test_file_not_found_propagates(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yaml"
        with pytest.raises(ConfigurationError, match="not found"):
            load_benchmark_plan(missing)

    def test_plan_from_file_with_sweep(self, tmp_path: Path) -> None:
        yaml_with_sweep = _MINIMAL_YAML + textwrap.dedent("""\
            sweep:
              type: grid
              variables:
                phases.default.concurrency:
                  - 1
                  - 2
        """)
        cfg_file = tmp_path / "sweep.yaml"
        cfg_file.write_text(yaml_with_sweep)

        plan = load_benchmark_plan(cfg_file)

        assert len(plan.configs) == 2
        concurrencies = [c.phases["default"].concurrency for c in plan.configs]
        assert concurrencies == [1, 2]

    def test_plan_from_file_with_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOADER_TEST_MODEL", "env-model")
        yaml_with_env = _MINIMAL_YAML.replace("test-model", "${LOADER_TEST_MODEL}")
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text(yaml_with_env)

        plan = load_benchmark_plan(cfg_file)

        assert plan.configs[0].get_model_names() == ["env-model"]
