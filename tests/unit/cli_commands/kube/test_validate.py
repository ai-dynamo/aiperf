# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.validate module.

Focuses on:
- YAML structure validation: apiVersion, kind, metadata.name, nested spec.benchmark
- Kubernetes name validation: length, pattern
- Unknown spec field detection: warning vs strict error
- AIPerfConfig validation via AIPerfJobSpecConverter
- DeploymentConfig, worker count validation
- File-level error handling: missing, unreadable, invalid YAML
- Multi-file validation with pass/fail counts
- CLI entrypoint integration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pytest import param

from aiperf.kubernetes.validate import (
    KNOWN_SPEC_FIELDS,
    ValidationResult,
    validate_aiperf_config,
    validate_file,
    validate_files,
    validate_k8s_name,
    validate_unknown_spec_fields,
    validate_yaml_structure,
)


def _minimal_doc() -> dict:
    """Return a minimal valid AIPerfJob document (nested spec.benchmark format)."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "test-job"},
        "spec": {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {
                    "urls": ["http://localhost:8000/v1/chat/completions"],
                },
                "datasets": {
                    "main": {
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 32, "osl": 16},
                    },
                },
                "phases": {
                    "default": {
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 1,
                    },
                },
            },
        },
    }


def _write_yaml(path: Path, doc: dict) -> Path:
    """Write a YAML document to a file and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(doc))
    return path


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_passed_no_errors(self) -> None:
        result = ValidationResult(path=Path("test.yaml"))
        assert result.passed is True

    def test_passed_with_warnings_only(self) -> None:
        result = ValidationResult(path=Path("test.yaml"), warnings=["some warning"])
        assert result.passed is True

    def test_not_passed_with_errors(self) -> None:
        result = ValidationResult(path=Path("test.yaml"), errors=["some error"])
        assert result.passed is False


# ---------------------------------------------------------------------------
# validate_yaml_structure
# ---------------------------------------------------------------------------


class TestValidateYamlStructure:
    def test_valid_structure_returns_true(self) -> None:
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(_minimal_doc(), result) is True
        assert not result.errors

    def test_not_a_dict_returns_false(self) -> None:
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure("not a dict", result) is False
        assert "not a YAML mapping" in result.errors[0]

    @pytest.mark.parametrize(
        "api_version",
        [
            param(None, id="missing"),
            param("v1", id="wrong-version"),
            param("aiperf.nvidia.com/v1beta1", id="wrong-group-version"),
        ],
    )  # fmt: skip
    def test_wrong_api_version(self, api_version: str | None) -> None:
        doc = _minimal_doc()
        if api_version is None:
            del doc["apiVersion"]
        else:
            doc["apiVersion"] = api_version
        result = ValidationResult(path=Path("test.yaml"))
        validate_yaml_structure(doc, result)
        assert any("apiVersion" in e for e in result.errors)

    @pytest.mark.parametrize(
        "kind",
        [
            param(None, id="missing"),
            param("Job", id="wrong-kind"),
        ],
    )  # fmt: skip
    def test_wrong_kind(self, kind: str | None) -> None:
        doc = _minimal_doc()
        if kind is None:
            del doc["kind"]
        else:
            doc["kind"] = kind
        result = ValidationResult(path=Path("test.yaml"))
        validate_yaml_structure(doc, result)
        assert any("kind" in e for e in result.errors)

    def test_missing_metadata_returns_false(self) -> None:
        doc = _minimal_doc()
        del doc["metadata"]
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(doc, result) is False
        assert any("metadata" in e for e in result.errors)

    def test_metadata_not_dict_returns_false(self) -> None:
        doc = _minimal_doc()
        doc["metadata"] = "not-a-dict"
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(doc, result) is False

    def test_missing_metadata_name_returns_false(self) -> None:
        doc = _minimal_doc()
        doc["metadata"] = {}
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(doc, result) is False
        assert any("metadata.name" in e for e in result.errors)

    def test_missing_spec_returns_false(self) -> None:
        doc = _minimal_doc()
        del doc["spec"]
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(doc, result) is False
        assert any("spec" in e for e in result.errors)

    def test_spec_not_dict_returns_false(self) -> None:
        doc = _minimal_doc()
        doc["spec"] = "not-a-dict"
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(doc, result) is False

    def test_missing_benchmark_key_returns_false(self) -> None:
        """spec.benchmark is required."""
        doc = _minimal_doc()
        del doc["spec"]["benchmark"]
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(doc, result) is False
        assert any("benchmark" in e for e in result.errors)

    def test_benchmark_not_dict_returns_false(self) -> None:
        """spec.benchmark must be a mapping."""
        doc = _minimal_doc()
        doc["spec"]["benchmark"] = "not-a-dict"
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(doc, result) is False

    def test_missing_config_fields_returns_false(self) -> None:
        """spec.benchmark must have at least models or endpoint."""
        doc = _minimal_doc()
        del doc["spec"]["benchmark"]["models"]
        del doc["spec"]["benchmark"]["endpoint"]
        result = ValidationResult(path=Path("test.yaml"))
        assert validate_yaml_structure(doc, result) is False
        assert any("models" in e or "endpoint" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_k8s_name
# ---------------------------------------------------------------------------


class TestValidateK8sName:
    @pytest.mark.parametrize(
        "name",
        [
            param("my-job", id="simple"),
            param("a1", id="short"),
            param("a", id="single-char"),
            param("7", id="single-digit"),
            param("test-benchmark-123", id="with-numbers"),
        ],
    )  # fmt: skip
    def test_valid_names(self, name: str) -> None:
        result = ValidationResult(path=Path("test.yaml"))
        validate_k8s_name(name, result)
        assert not result.errors

    def test_name_too_long(self) -> None:
        result = ValidationResult(path=Path("test.yaml"))
        validate_k8s_name("a" * 254, result)
        assert any("exceeds max" in e for e in result.errors)

    @pytest.mark.parametrize(
        "name",
        [
            param("My-Job", id="uppercase"),
            param("-leading-dash", id="leading-dash"),
            param("trailing-dash-", id="trailing-dash"),
            param("has_underscore", id="underscore"),
            param("has.dot", id="dot"),
        ],
    )  # fmt: skip
    def test_invalid_names(self, name: str) -> None:
        result = ValidationResult(path=Path("test.yaml"))
        validate_k8s_name(name, result)
        assert any("not a valid Kubernetes resource name" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_unknown_spec_fields
# ---------------------------------------------------------------------------


class TestValidateUnknownSpecFields:
    def test_all_known_fields_no_warnings(self) -> None:
        spec = {k: {} for k in KNOWN_SPEC_FIELDS}
        result = ValidationResult(path=Path("test.yaml"))
        validate_unknown_spec_fields(spec, result, strict=False)
        assert not result.warnings
        assert not result.errors

    def test_unknown_top_level_field_warning(self) -> None:
        spec = {"benchmark": {"models": []}, "bogusField": "value"}
        result = ValidationResult(path=Path("test.yaml"))
        validate_unknown_spec_fields(spec, result, strict=False)
        assert len(result.warnings) == 1
        assert "bogusField" in result.warnings[0]
        assert not result.errors

    def test_unknown_top_level_field_strict_error(self) -> None:
        spec = {"benchmark": {"models": []}, "bogusField": "value"}
        result = ValidationResult(path=Path("test.yaml"))
        validate_unknown_spec_fields(spec, result, strict=True)
        assert len(result.errors) == 1
        assert "bogusField" in result.errors[0]
        assert not result.warnings

    def test_multiple_unknown_top_level_fields(self) -> None:
        spec = {"benchmark": {}, "foo": 1, "bar": 2}
        result = ValidationResult(path=Path("test.yaml"))
        validate_unknown_spec_fields(spec, result, strict=False)
        assert len(result.warnings) == 1
        assert "bar" in result.warnings[0]
        assert "foo" in result.warnings[0]

    def test_unknown_benchmark_field_warning(self) -> None:
        spec = {"benchmark": {"models": [], "bogusConfig": "value"}}
        result = ValidationResult(path=Path("test.yaml"))
        validate_unknown_spec_fields(spec, result, strict=False)
        assert any("bogusConfig" in w for w in result.warnings)

    def test_unknown_benchmark_field_strict_error(self) -> None:
        spec = {"benchmark": {"models": [], "bogusConfig": "value"}}
        result = ValidationResult(path=Path("test.yaml"))
        validate_unknown_spec_fields(spec, result, strict=True)
        assert any("bogusConfig" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_aiperf_config
# ---------------------------------------------------------------------------


class TestValidateAIPerfConfig:
    def test_valid_config(self) -> None:
        doc = _minimal_doc()
        result = ValidationResult(path=Path("test.yaml"))
        validate_aiperf_config(doc["spec"], "test-job", result)
        assert not result.errors

    def test_empty_models(self) -> None:
        doc = _minimal_doc()
        doc["spec"]["benchmark"]["models"] = []
        result = ValidationResult(path=Path("test.yaml"))
        validate_aiperf_config(doc["spec"], "test-job", result)
        assert any("models" in e for e in result.errors)

    def test_empty_urls(self) -> None:
        doc = _minimal_doc()
        doc["spec"]["benchmark"]["endpoint"]["urls"] = []
        result = ValidationResult(path=Path("test.yaml"))
        validate_aiperf_config(doc["spec"], "test-job", result)
        assert any("urls" in e for e in result.errors)

    def test_invalid_url_scheme(self) -> None:
        doc = _minimal_doc()
        doc["spec"]["benchmark"]["endpoint"]["urls"] = ["ftp://bad:8000"]
        result = ValidationResult(path=Path("test.yaml"))
        validate_aiperf_config(doc["spec"], "test-job", result)
        assert any("http://" in e for e in result.errors)

    def test_pydantic_validation_error(self) -> None:
        spec = {"benchmark": {"models": 123}}
        result = ValidationResult(path=Path("test.yaml"))
        validate_aiperf_config(spec, "test-job", result)
        assert any("Config validation failed" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_file (end-to-end single file)
# ---------------------------------------------------------------------------


class TestValidateFile:
    def test_valid_file(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path / "aiperfjob.yaml", _minimal_doc())
        result = validate_file(path)
        assert result.passed
        assert not result.errors
        assert not result.warnings

    def test_file_not_exists(self, tmp_path: Path) -> None:
        result = validate_file(tmp_path / "nonexistent.yaml")
        assert not result.passed
        assert any("does not exist" in e for e in result.errors)

    def test_not_a_file(self, tmp_path: Path) -> None:
        result = validate_file(tmp_path)
        assert not result.passed
        assert any("Not a file" in e for e in result.errors)

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("{{invalid yaml")
        result = validate_file(path)
        assert not result.passed
        assert any("YAML parse error" in e for e in result.errors)

    def test_wrong_api_version(self, tmp_path: Path) -> None:
        doc = _minimal_doc()
        doc["apiVersion"] = "wrong/v1"
        path = _write_yaml(tmp_path / "aiperfjob.yaml", doc)
        result = validate_file(path)
        assert not result.passed

    def test_unknown_fields_warning_default(self, tmp_path: Path) -> None:
        doc = _minimal_doc()
        doc["spec"]["extraField"] = "value"
        path = _write_yaml(tmp_path / "aiperfjob.yaml", doc)
        result = validate_file(path, strict=False)
        assert result.passed
        assert len(result.warnings) == 1

    def test_unknown_fields_error_strict(self, tmp_path: Path) -> None:
        doc = _minimal_doc()
        doc["spec"]["extraField"] = "value"
        path = _write_yaml(tmp_path / "aiperfjob.yaml", doc)
        result = validate_file(path, strict=True)
        assert not result.passed

    def test_all_validations_run(self, tmp_path: Path) -> None:
        doc = _minimal_doc()
        doc["spec"]["podTemplate"] = {
            "env": [{"name": "FOO", "value": "bar"}],
            "volumes": [],
            "volumeMounts": [],
        }
        doc["spec"]["connectionsPerWorker"] = 100
        path = _write_yaml(tmp_path / "aiperfjob.yaml", doc)
        result = validate_file(path)
        assert result.passed


# ---------------------------------------------------------------------------
# validate_files (multi-file)
# ---------------------------------------------------------------------------


class TestValidateFiles:
    @pytest.mark.asyncio
    async def test_all_pass(self, tmp_path: Path) -> None:
        paths = []
        for i in range(3):
            doc = _minimal_doc()
            doc["metadata"]["name"] = f"job-{i}x"
            paths.append(_write_yaml(tmp_path / f"job{i}.yaml", doc))

        with patch("aiperf.kubernetes.validate.kube_console"):
            passed, failed = await validate_files(paths)

        assert passed == 3
        assert failed == 0

    @pytest.mark.asyncio
    async def test_mixed_pass_fail(self, tmp_path: Path) -> None:
        good = _write_yaml(tmp_path / "good.yaml", _minimal_doc())
        bad_doc = _minimal_doc()
        del bad_doc["spec"]["benchmark"]["models"]
        del bad_doc["spec"]["benchmark"]["endpoint"]
        bad = _write_yaml(tmp_path / "bad.yaml", bad_doc)

        with patch("aiperf.kubernetes.validate.kube_console"):
            passed, failed = await validate_files([good, bad])

        assert passed == 1
        assert failed == 1

    @pytest.mark.asyncio
    async def test_all_fail(self, tmp_path: Path) -> None:
        paths = []
        for i in range(2):
            doc = _minimal_doc()
            del doc["spec"]
            paths.append(_write_yaml(tmp_path / f"bad{i}.yaml", doc))

        with patch("aiperf.kubernetes.validate.kube_console"):
            passed, failed = await validate_files(paths)

        assert passed == 0
        assert failed == 2

    @pytest.mark.asyncio
    async def test_strict_flag_forwarded(self, tmp_path: Path) -> None:
        doc = _minimal_doc()
        doc["spec"]["unknownField"] = "x"
        path = _write_yaml(tmp_path / "job.yaml", doc)

        with patch("aiperf.kubernetes.validate.kube_console"):
            passed_lax, failed_lax = await validate_files([path], strict=False)
            passed_strict, failed_strict = await validate_files([path], strict=True)

        assert passed_lax == 1 and failed_lax == 0
        assert passed_strict == 0 and failed_strict == 1

    @pytest.mark.asyncio
    async def test_nonexistent_file_counts_as_failure(self, tmp_path: Path) -> None:
        with patch("aiperf.kubernetes.validate.kube_console"):
            passed, failed = await validate_files([tmp_path / "nope.yaml"])

        assert passed == 0
        assert failed == 1


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


class TestValidateCLI:
    @pytest.mark.asyncio
    async def test_validate_success_no_exit(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path / "aiperfjob.yaml", _minimal_doc())

        with patch("aiperf.kubernetes.validate.kube_console"):
            from aiperf.cli_commands.kube.validate import validate

            await validate(files=[path])

    @pytest.mark.asyncio
    async def test_validate_failure_raises_system_exit(self, tmp_path: Path) -> None:
        doc = _minimal_doc()
        del doc["spec"]
        path = _write_yaml(tmp_path / "bad.yaml", doc)

        with (
            patch("aiperf.kubernetes.validate.kube_console"),
            pytest.raises(SystemExit) as exc_info,
        ):
            from aiperf.cli_commands.kube.validate import validate

            await validate(files=[path])

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_validate_strict_flag(self, tmp_path: Path) -> None:
        doc = _minimal_doc()
        doc["spec"]["bogus"] = "x"
        path = _write_yaml(tmp_path / "job.yaml", doc)

        with (
            patch("aiperf.kubernetes.validate.kube_console"),
            pytest.raises(SystemExit),
        ):
            from aiperf.cli_commands.kube.validate import validate

            await validate(files=[path], strict=True)
