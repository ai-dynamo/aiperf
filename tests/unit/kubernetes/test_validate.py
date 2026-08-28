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
from pytest import param

from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION
from aiperf.kubernetes.validate import (
    K8S_NAME_MAX_LENGTH,
    KNOWN_SPEC_FIELDS,
    ValidationResult,
    validate_file,
    validate_k8s_name,
    validate_manifest,
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
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128, "osl": 64},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "type": "concurrency",
                        "kind": "profiling",
                        "concurrency": 1,
                        "requests": 10,
                    }
                ],
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
        """Regression — `benchmark` stays in the known top-level spec fields."""
        assert "benchmark" in KNOWN_SPEC_FIELDS


class TestValidateFile:
    """End-to-end: YAML on disk → ValidationResult."""

    def test_valid_file_passes(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _valid_doc())
        result = validate_file(path)
        assert result.passed, f"unexpected errors: {result.errors}"

    def test_jinja_variables_are_rendered_before_kind_validation(
        self, tmp_path: Path
    ) -> None:
        doc = _valid_doc()
        doc["spec"]["variables"] = {
            "concurrency_per_worker": 4,
            "workers": 2,
            "total_concurrency": "{{ concurrency_per_worker * workers }}",
        }
        doc["spec"]["randomSeed"] = 42
        phase = doc["spec"]["benchmark"]["phases"][0]
        phase["concurrency"] = "{{ total_concurrency }}"
        path = _write(tmp_path, doc)

        result = validate_file(path, strict=True)

        assert result.passed, f"unexpected errors: {result.errors}"

    def test_unknown_jinja_variable_is_reported_as_validation_error(
        self, tmp_path: Path
    ) -> None:
        doc = _valid_doc()
        doc["spec"]["benchmark"]["phases"][0]["concurrency"] = "{{ typo }}"
        path = _write(tmp_path, doc)

        result = validate_file(path, strict=True)

        assert not result.passed
        assert any("typo" in error for error in result.errors)

    def test_literal_authorization_header_requires_secret_transport(
        self, tmp_path: Path
    ) -> None:
        doc = _valid_doc()
        doc["spec"]["benchmark"]["endpoint"]["headers"] = {
            "Authorization": "Bearer plaintext"
        }
        path = _write(tmp_path, doc)

        result = validate_file(path, strict=True)

        assert not result.passed
        assert any("AIPERF_INJECTED_HEADERS" in error for error in result.errors)

    def test_secret_backed_authorization_transport_passes(self, tmp_path: Path) -> None:
        doc = _valid_doc()
        doc["spec"]["benchmark"]["endpoint"]["headers"] = {
            "Authorization": "<redacted>"
        }
        doc["spec"]["podTemplate"] = {
            "env": [
                {
                    "name": "AIPERF_INJECTED_HEADERS",
                    "valueFrom": {
                        "secretKeyRef": {"name": "endpoint", "key": "headers"}
                    },
                }
            ]
        }
        path = _write(tmp_path, doc)

        result = validate_file(path, strict=True)

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
        # EndpointConfig rejects the scheme during model validation, so the
        # message names the offending scheme and the supported ones rather
        # than the literal "http://"/"https://" prefixes.
        assert any("grpc" in e and "http" in e for e in result.errors)

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


def _valid_sweep_doc() -> dict:
    """Minimal AIPerfSweep doc that should pass validation.

    Carries the same workload body as _valid_doc but flips kind to
    ``AIPerfSweep`` and adds a ``spec.sweep`` block (the whole reason an
    AIPerfSweep CR exists).
    """
    doc = _valid_doc()
    doc["kind"] = "AIPerfSweep"
    doc["metadata"]["name"] = "my-sweep"
    doc["spec"]["sweep"] = {
        "type": "grid",
        "parameters": {
            "phases.profiling.concurrency": [1, 2, 4],
        },
    }
    return doc


class TestValidateFileKindDispatch:
    """validate_file accepts both AIPerfJob and AIPerfSweep."""

    def test_validate_file_aiperfjob_with_sweep_block_rejected(
        self, tmp_path: Path
    ) -> None:
        """AIPerfJob.spec.sweep must be null; mirrors the CEL rule."""
        doc = _valid_doc()
        doc["spec"]["sweep"] = {
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [1, 2]},
        }
        path = _write(tmp_path, doc)

        result = validate_file(path)
        assert not result.passed
        assert any("spec.sweep" in e and "AIPerfJob" in e for e in result.errors), (
            f"errors: {result.errors}"
        )

    def test_validate_file_aiperfsweep_without_sweep_block_rejected(
        self, tmp_path: Path
    ) -> None:
        """AIPerfSweep.spec.sweep is required; absence trips local validation."""
        doc = _valid_sweep_doc()
        del doc["spec"]["sweep"]
        path = _write(tmp_path, doc)

        result = validate_file(path)
        assert not result.passed
        assert any("spec.sweep" in e and "required" in e for e in result.errors), (
            f"errors: {result.errors}"
        )

    def test_validate_file_aiperfsweep_with_valid_sweep_passes(
        self, tmp_path: Path
    ) -> None:
        """Happy path: an AIPerfSweep YAML with a sweep block validates clean."""
        path = _write(tmp_path, _valid_sweep_doc(), name="sweep.yaml")

        result = validate_file(path)
        assert result.passed, f"unexpected errors: {result.errors}"

    def test_validate_file_unknown_kind_rejected(self, tmp_path: Path) -> None:
        """`kind: Pod` and friends are rejected with a clear error."""
        doc = _valid_doc()
        doc["kind"] = "Pod"
        path = _write(tmp_path, doc)

        result = validate_file(path)
        assert not result.passed
        assert any(
            "kind" in e and "AIPerfJob" in e and "AIPerfSweep" in e
            for e in result.errors
        ), f"errors: {result.errors}"

    def test_validate_file_aiperfsweep_with_empty_sweep_rejected(
        self, tmp_path: Path
    ) -> None:
        """An empty `sweep: {}` block on AIPerfSweep is just as bad as missing."""
        doc = _valid_sweep_doc()
        doc["spec"]["sweep"] = {}
        path = _write(tmp_path, doc)

        result = validate_file(path)
        assert not result.passed
        assert any("spec.sweep" in e for e in result.errors), f"errors: {result.errors}"

    def test_validate_file_aiperfsweep_strict_passes(self, tmp_path: Path) -> None:
        """Strict mode does not flag the AIPerfSweep envelope keys (sweep, multiRun, ...)."""
        path = _write(tmp_path, _valid_sweep_doc(), name="sweep.yaml")

        result = validate_file(path, strict=True)
        assert result.passed, f"unexpected errors: {result.errors}"


# Divisors that used to make ``ceil(concurrency / connectionsPerWorker)`` raise
# out of the whole validation run. Two exception families: 0, 0.0, -0.0 and YAML
# ``false`` (a Python ``int``) divide as zero -> ZeroDivisionError; denormals
# overflow the quotient to ``inf``, which ``math.ceil`` cannot convert ->
# OverflowError. Neither is a ValueError subclass, so widening the caught tuple
# alone would not have closed the hole. The paired pydantic error tag is the one
# an already-completed earlier step (``validate_deployment_config``) appends, so
# each case has to keep reporting it.
_CRASHING_DIVISORS = [
    param(0, "greater_than_equal", id="int-zero"),
    param(0.0, "greater_than_equal", id="float-zero"),
    param(-0.0, "greater_than_equal", id="negative-float-zero"),
    param(False, "greater_than_equal", id="yaml-false"),
    param(1.0e-320, "int_from_float", id="denormal-1e-320"),
    param(5e-324, "int_from_float", id="smallest-subnormal"),
]


class TestConnectionsPerWorkerArithmetic:
    """A hostile ``connectionsPerWorker`` must be reported, never raised.

    ``DeploymentConfig`` bounds the field ``ge=1`` and the CRDs bound it
    ``minimum: 1``, but the validators divide the *raw* spec dict, so these
    values reach the arithmetic on every unvalidated entry point: ``aiperf kube
    validate``, ``POST /api/v1/validate``, ``kube profile --dry-run``, and
    ``kube generate --operator``.
    """

    @pytest.mark.parametrize(("divisor", "error_tag"), _CRASHING_DIVISORS)  # fmt: skip
    def test_manifest_reports_bad_divisor_instead_of_raising(
        self, divisor: object, error_tag: str
    ) -> None:
        doc = _valid_doc()
        doc["spec"]["connectionsPerWorker"] = divisor

        result = validate_manifest(doc)

        assert not result.passed
        assert any(error_tag in e for e in result.errors), f"errors: {result.errors}"

    @pytest.mark.parametrize(("divisor", "error_tag"), _CRASHING_DIVISORS)  # fmt: skip
    def test_file_reports_bad_divisor_instead_of_raising(
        self, tmp_path: Path, divisor: object, error_tag: str
    ) -> None:
        """The CLI path must not abort the run — a later file still gets validated."""
        doc = _valid_doc()
        doc["spec"]["connectionsPerWorker"] = divisor
        path = _write(tmp_path, doc)

        result = validate_file(path)

        assert not result.passed
        assert not any("Worker calculation failed" in e for e in result.errors), (
            f"the divisor must be neutralized, not caught: {result.errors}"
        )

    @pytest.mark.parametrize(
        "divisor",
        [
            param("abc", id="unparsable-string"),
            param(None, id="null"),
            param([], id="list"),
        ],
    )  # fmt: skip
    def test_non_numeric_divisor_still_reports_type_error(
        self, divisor: object
    ) -> None:
        """Non-numeric values pass through the neutralizer so the TypeError surfaces."""
        doc = _valid_doc()
        doc["spec"]["connectionsPerWorker"] = divisor

        result = validate_manifest(doc)

        assert not result.passed
        assert any("Worker calculation failed" in e for e in result.errors), (
            f"errors: {result.errors}"
        )

    @pytest.mark.parametrize(
        "divisor",
        [
            param(1, id="one"),
            param(100, id="default"),
            param(1000, id="above-concurrency"),
        ],
    )  # fmt: skip
    def test_valid_divisor_still_passes(self, divisor: int) -> None:
        doc = _valid_doc()
        doc["spec"]["connectionsPerWorker"] = divisor

        result = validate_manifest(doc)

        assert result.passed, f"unexpected errors: {result.errors}"

    def test_negative_divisor_verdict_unchanged(self) -> None:
        """``-1`` never crashed; it must keep the exact errors it produced before."""
        doc = _valid_doc()
        doc["spec"]["connectionsPerWorker"] = -1

        result = validate_manifest(doc)

        assert not result.passed
        assert not any("Worker calculation failed" in e for e in result.errors)
        assert sum("greater_than_equal" in e for e in result.errors) == 2, (
            f"errors: {result.errors}"
        )

    def test_credential_transport_gate_still_runs_on_a_clean_file(self) -> None:
        """Step order is load-bearing: the gate keys off steps 6-8 only.

        ``validate_endpoint_credential_transport`` is skipped when an earlier
        step already errored, so moving the kind-specific schema check ahead of
        it would silently drop this security check on files that reach it today.
        """
        doc = _valid_doc()
        doc["spec"]["benchmark"]["endpoint"]["api_key"] = "sk-literal-secret"

        result = validate_manifest(doc)

        assert not result.passed
        assert any(
            "Endpoint credential transport validation failed" in e
            for e in result.errors
        ), f"errors: {result.errors}"


class TestCredentialRedactionInValidationErrors:
    """``ValidationResult.errors`` crosses the unauthenticated
    ``POST /api/v1/validate`` boundary verbatim, so a malformed credential must
    never be echoed back -- while the field path and the structural reason,
    which are what make the error actionable, must survive."""

    @pytest.mark.parametrize(
        "endpoint,secret,loc",
        [
            param(
                {"urls": ["http://svc:8000"], "headers": ["Bearer SUPERSECRETTOKEN"]},
                "SUPERSECRETTOKEN",
                "endpoint.headers",
                id="malformed-authorization-header",
            ),
            param(
                {"urls": {"https://user:SUPERSECRETPW@example.com/v1": "x"}},
                "SUPERSECRETPW",
                "endpoint.urls",
                id="malformed-urls-with-userinfo",
            ),
            param(
                {"urls": ["http://svc:8000"], "apiKey": ["SUPERSECRETKEY"]},
                "SUPERSECRETKEY",
                "endpoint.apiKey",
                id="malformed-api-key",
            ),
        ],
    )  # fmt: skip
    def test_malformed_credentials_are_not_echoed(
        self, endpoint: dict, secret: str, loc: str
    ) -> None:
        doc = _valid_doc()
        doc["spec"]["benchmark"]["endpoint"] = endpoint

        result = validate_manifest(doc)

        assert not result.passed
        joined = "\n".join(result.errors)
        assert secret not in joined, f"credential leaked: {joined}"
        assert "<redacted>" in joined
        # Diagnostics preserved: the user still learns which field is wrong.
        assert loc in joined

    def test_non_credential_input_values_are_preserved(self) -> None:
        """Redaction is scoped to credential fields; an ordinary bad value keeps
        its ``input_value`` so a benchmark user can see what they typed."""
        doc = _valid_doc()
        doc["spec"]["benchmark"]["endpoint"]["timeout"] = "not-a-number"

        result = validate_manifest(doc)

        assert not result.passed
        joined = "\n".join(result.errors)
        assert "not-a-number" in joined
        assert "endpoint.timeout" in joined


_BARE_CONFIG = {
    "models": ["m"],
    "endpoint": {"urls": ["http://x"], "type": "chat", "streaming": True},
    "datasets": [
        {"name": "main", "type": "synthetic", "prompts": {"isl": 64, "osl": 32}}
    ],
    "phases": [
        {"name": "profiling", "type": "concurrency", "duration": 10, "concurrency": 8}
    ],
}


class TestBareConfigDocuments:
    """`aiperf kube sweep --config` accepts a bare Config-v2 document; validate
    must hold that same document to the same contract instead of reporting
    three unactionable missing-CR-field errors."""

    def test_bare_config_validates_without_cr_wrapper(self) -> None:
        result = validate_manifest(dict(_BARE_CONFIG))

        assert result.passed, result.errors
        assert not any("apiVersion" in e or "kind:" in e for e in result.errors)

    def test_bare_config_warns_which_contract_it_used(self) -> None:
        result = validate_manifest(dict(_BARE_CONFIG))

        assert any("bare AIPerf config" in w for w in result.warnings)
        assert any("AIPerfJob" in w for w in result.warnings)

    def test_bare_config_reports_stray_sibling_of_nested_benchmark(self) -> None:
        """Siblings of an explicit ``benchmark:`` must not be silently dropped."""
        doc = {"benchmark": dict(_BARE_CONFIG), "totallyBogus": 5}

        result = validate_manifest(doc)

        assert any("totallyBogus" in w for w in result.warnings)

    def test_bare_config_keeps_deployment_siblings_at_spec_level(self) -> None:
        """A known deployment field beside ``benchmark:`` is not an unknown field."""
        doc = {"benchmark": dict(_BARE_CONFIG), "image": "aiperf:latest"}

        result = validate_manifest(doc)

        assert not any("Unknown spec fields" in w for w in result.warnings)

    def test_bare_config_with_sweep_uses_the_sweep_contract(self) -> None:
        doc = dict(_BARE_CONFIG) | {
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [4, 8]},
            }
        }

        result = validate_manifest(doc)

        assert result.passed, result.errors
        assert any("AIPerfSweep" in w for w in result.warnings)

    def test_bare_config_snake_case_envelope_key_is_not_an_unknown_field(self) -> None:
        """`multi_run` must normalize to the `multiRun` wire alias, otherwise
        the spec-level unknown-field check flags a key the loader accepts."""
        doc = dict(_BARE_CONFIG) | {"multi_run": {"count": 2}}

        result = validate_manifest(doc)

        assert not any("multi_run" in w for w in result.warnings), result.warnings

    def test_bare_config_still_reports_real_errors(self) -> None:
        doc = dict(_BARE_CONFIG) | {
            "endpoint": {"urls": ["ftp://x"], "type": "chat", "streaming": True}
        }

        result = validate_manifest(doc)

        assert not result.passed

    def test_wrapped_cr_is_not_treated_as_a_bare_config(self) -> None:
        result = validate_manifest(_valid_doc())

        assert result.passed, result.errors
        assert not any("bare AIPerf config" in w for w in result.warnings)

    def test_document_with_neither_wrapper_nor_config_keys_still_errors(self) -> None:
        result = validate_manifest({"unrelated": True})

        assert not result.passed
        assert any("apiVersion" in e for e in result.errors)

    def test_bare_config_file_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.yaml"
        path.write_text(yaml.safe_dump(_BARE_CONFIG))

        result = validate_file(path)

        assert result.passed, result.errors
