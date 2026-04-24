# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.validate.

``validate_file`` is the engine behind ``aiperf kube validate``: it reads an
AIPerfJob YAML, checks apiVersion/kind/metadata/spec shape, the Kubernetes
name regex, unknown-field detection (warning vs. error under --strict), and
runs the spec through the operator's spec converter to catch schema errors
before ``kubectl apply``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION
from aiperf.kubernetes.validate import (
    K8S_NAME_MAX_LENGTH,
    KNOWN_SPEC_FIELDS,
    ValidationResult,
    validate_file,
    validate_k8s_name,
    validate_unknown_spec_fields,
    validate_yaml_structure,
)


def _valid_doc() -> dict:
    """Minimal AIPerfJob doc that passes every validator."""
    return {
        "apiVersion": AIPERF_API_VERSION,
        "kind": "AIPerfJob",
        "metadata": {"name": "my-bench"},
        "spec": {
            "benchmark": {
                "models": ["meta-llama/Llama-3.1-8B-Instruct"],
                "endpoint": {"urls": ["http://svc.ns.svc.cluster.local:8000"]},
                "datasets": {
                    "default": {
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128, "osl": 64},
                    }
                },
                "phases": {
                    "default": {
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 10,
                    }
                },
            }
        },
    }


def _write(tmp_path: Path, doc: dict, name: str = "job.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return path


class TestValidationResult:
    """ValidationResult.passed is derived from errors list."""

    def test_passed_when_no_errors(self) -> None:
        r = ValidationResult(path=Path("x"))
        assert r.passed

    def test_not_passed_when_any_errors(self) -> None:
        r = ValidationResult(path=Path("x"), errors=["something"])
        assert not r.passed

    def test_warnings_do_not_affect_passed(self) -> None:
        """Warnings are informational only in non-strict mode."""
        r = ValidationResult(path=Path("x"), warnings=["careful there"])
        assert r.passed


class TestValidateYamlStructure:
    """Top-level apiVersion/kind/metadata/spec/benchmark checks."""

    def test_valid_doc_passes(self) -> None:
        r = ValidationResult(path=Path("x"))
        assert validate_yaml_structure(_valid_doc(), r) is True
        assert r.errors == []

    def test_non_dict_document_fails_hard(self) -> None:
        r = ValidationResult(path=Path("x"))
        assert validate_yaml_structure("not a dict", r) is False
        assert any("not a YAML mapping" in e for e in r.errors)

    def test_wrong_api_version_reports_expected(self) -> None:
        doc = _valid_doc()
        doc["apiVersion"] = "wrong/v1"
        r = ValidationResult(path=Path("x"))

        # apiVersion error doesn't short-circuit if structure is otherwise intact
        validate_yaml_structure(doc, r)
        assert any("apiVersion" in e and "wrong/v1" in e for e in r.errors)

    def test_wrong_kind_reports_error(self) -> None:
        doc = _valid_doc()
        doc["kind"] = "Pod"
        r = ValidationResult(path=Path("x"))

        validate_yaml_structure(doc, r)
        assert any("kind" in e and "Pod" in e for e in r.errors)

    def test_missing_metadata_short_circuits(self) -> None:
        doc = _valid_doc()
        del doc["metadata"]
        r = ValidationResult(path=Path("x"))

        assert validate_yaml_structure(doc, r) is False
        assert any("metadata" in e for e in r.errors)

    def test_missing_metadata_name_short_circuits(self) -> None:
        doc = _valid_doc()
        doc["metadata"] = {}
        r = ValidationResult(path=Path("x"))

        assert validate_yaml_structure(doc, r) is False
        assert any("metadata.name" in e for e in r.errors)

    def test_missing_spec_short_circuits(self) -> None:
        doc = _valid_doc()
        del doc["spec"]
        r = ValidationResult(path=Path("x"))

        assert validate_yaml_structure(doc, r) is False
        assert any("spec" in e for e in r.errors)

    def test_missing_spec_benchmark_short_circuits(self) -> None:
        doc = _valid_doc()
        doc["spec"] = {"image": "placeholder"}
        r = ValidationResult(path=Path("x"))

        assert validate_yaml_structure(doc, r) is False
        assert any("spec.benchmark" in e for e in r.errors)

    def test_benchmark_without_models_or_endpoint_short_circuits(self) -> None:
        doc = _valid_doc()
        doc["spec"]["benchmark"] = {"datasets": {}}
        r = ValidationResult(path=Path("x"))

        assert validate_yaml_structure(doc, r) is False
        assert any(
            "must contain at least 'models' or 'endpoint'" in e for e in r.errors
        )


class TestValidateK8sName:
    """metadata.name must match DNS-1123 subdomain rules."""

    @pytest.mark.parametrize(
        "name",
        [
            "valid",
            "valid-name",
            "a",
            "my-bench-123",
            "abc123",
        ],
    )
    def test_valid_names_produce_no_error(self, name) -> None:
        r = ValidationResult(path=Path("x"))
        validate_k8s_name(name, r)
        assert r.errors == []

    @pytest.mark.parametrize(
        "name",
        [
            "Invalid",  # uppercase
            "-starts-with-hyphen",
            "ends-with-hyphen-",
            "has_underscore",
            "has.dot",
            "has spaces",
            "",  # empty
        ],
    )
    def test_invalid_names_produce_error(self, name) -> None:
        r = ValidationResult(path=Path("x"))
        validate_k8s_name(name, r)
        assert any("not a valid Kubernetes resource name" in e for e in r.errors)

    def test_too_long_name_produces_length_error(self) -> None:
        r = ValidationResult(path=Path("x"))
        validate_k8s_name("a" * (K8S_NAME_MAX_LENGTH + 1), r)
        assert any("exceeds max" in e for e in r.errors)

    def test_exactly_max_length_is_allowed(self) -> None:
        r = ValidationResult(path=Path("x"))
        validate_k8s_name("a" * K8S_NAME_MAX_LENGTH, r)
        assert all("exceeds max" not in e for e in r.errors)


class TestValidateUnknownSpecFields:
    """Unknown fields are warnings by default, errors under --strict."""

    def test_no_unknown_fields_is_clean(self) -> None:
        r = ValidationResult(path=Path("x"))
        validate_unknown_spec_fields({"benchmark": {"models": ["x"]}}, r, strict=False)
        assert r.errors == []
        assert r.warnings == []

    def test_unknown_top_level_is_warning_non_strict(self) -> None:
        r = ValidationResult(path=Path("x"))
        validate_unknown_spec_fields(
            {"benchmark": {"models": ["x"]}, "nonsense": "foo"}, r, strict=False
        )
        assert r.errors == []
        assert any("nonsense" in w for w in r.warnings)

    def test_unknown_top_level_is_error_strict(self) -> None:
        r = ValidationResult(path=Path("x"))
        validate_unknown_spec_fields(
            {"benchmark": {"models": ["x"]}, "nonsense": "foo"}, r, strict=True
        )
        assert any("nonsense" in e for e in r.errors)
        assert r.warnings == []

    def test_known_deployment_fields_are_accepted(self) -> None:
        """'image', 'podTemplate', etc. belong at top-level spec."""
        r = ValidationResult(path=Path("x"))
        validate_unknown_spec_fields(
            {
                "benchmark": {"models": ["x"]},
                "image": "my-img",
                "podTemplate": {},
                "ttlSecondsAfterFinished": 300,
            },
            r,
            strict=True,
        )
        assert r.errors == []

    def test_unknown_benchmark_field_is_warning_non_strict(self) -> None:
        r = ValidationResult(path=Path("x"))
        validate_unknown_spec_fields(
            {"benchmark": {"models": ["x"], "bogus_field": True}}, r, strict=False
        )
        assert any("spec.benchmark" in w and "bogus_field" in w for w in r.warnings)

    def test_unknown_benchmark_field_is_error_strict(self) -> None:
        r = ValidationResult(path=Path("x"))
        validate_unknown_spec_fields(
            {"benchmark": {"models": ["x"], "bogus_field": True}}, r, strict=True
        )
        assert any("spec.benchmark" in e and "bogus_field" in e for e in r.errors)

    def test_benchmark_field_is_in_known_spec_fields(self) -> None:
        """Regression — `benchmark` is the only known non-deployment key."""
        assert "benchmark" in KNOWN_SPEC_FIELDS


class TestValidateFile:
    """End-to-end: YAML on disk → ValidationResult."""

    def test_valid_file_passes(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _valid_doc())
        result = validate_file(path)
        assert result.passed, f"unexpected errors: {result.errors}"

    def test_missing_file_produces_error(self, tmp_path: Path) -> None:
        result = validate_file(tmp_path / "nope.yaml")
        assert not result.passed
        assert any("does not exist" in e for e in result.errors)

    def test_directory_instead_of_file_errors(self, tmp_path: Path) -> None:
        result = validate_file(tmp_path)
        assert not result.passed
        assert any("Not a file" in e for e in result.errors)

    def test_malformed_yaml_produces_parse_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("key: value\n  bad: indent\n    : not a key")
        result = validate_file(path)
        assert not result.passed
        assert any("YAML parse error" in e for e in result.errors)

    def test_invalid_k8s_name_surfaces(self, tmp_path: Path) -> None:
        doc = _valid_doc()
        doc["metadata"]["name"] = "Invalid_Name"
        path = _write(tmp_path, doc)

        result = validate_file(path)
        assert any("not a valid Kubernetes resource name" in e for e in result.errors)

    def test_non_http_endpoint_url_errors(self, tmp_path: Path) -> None:
        """Operator only speaks http(s); other schemes trip the validator."""
        doc = _valid_doc()
        doc["spec"]["benchmark"]["endpoint"] = {"urls": ["grpc://svc:8000"]}
        path = _write(tmp_path, doc)

        result = validate_file(path)
        assert any("http://" in e and "https://" in e for e in result.errors)

    def test_warnings_under_non_strict_do_not_fail(self, tmp_path: Path) -> None:
        doc = _valid_doc()
        doc["spec"]["nonsense"] = "foo"
        path = _write(tmp_path, doc)

        result = validate_file(path, strict=False)
        assert result.passed
        assert any("nonsense" in w for w in result.warnings)

    def test_warnings_under_strict_become_errors(self, tmp_path: Path) -> None:
        doc = _valid_doc()
        doc["spec"]["nonsense"] = "foo"
        path = _write(tmp_path, doc)

        result = validate_file(path, strict=True)
        assert not result.passed
        assert any("nonsense" in e for e in result.errors)

    def test_wrong_api_version_fails(self, tmp_path: Path) -> None:
        doc = _valid_doc()
        doc["apiVersion"] = "v1"
        path = _write(tmp_path, doc)

        result = validate_file(path)
        assert not result.passed
