# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core validation logic for AIPerfJob and AIPerfSweep YAML files.

Both CRD specs share the same nested shape: AIPerfConfig fields live under
spec.benchmark, deployment fields (image, podTemplate, etc.) live at the
spec level, and sweep envelope fields (sweep, multiRun, variables,
randomSeed) live at the spec level too. The kind toggles the cardinality
of `spec.sweep`: AIPerfJob forbids it, AIPerfSweep requires it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydantic
import yaml

from aiperf.kubernetes import console as kube_console
from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION
from aiperf.operator.models import AIPerfJobSpec, AIPerfSweepSpec
from aiperf.operator.spec_converter import CONFIG_FIELDS, AIPerfJobSpecConverter

EXPECTED_API_VERSION = AIPERF_API_VERSION
KIND_AIPERFJOB = "AIPerfJob"
KIND_AIPERFSWEEP = "AIPerfSweep"
SUPPORTED_KINDS: frozenset[str] = frozenset({KIND_AIPERFJOB, KIND_AIPERFSWEEP})
# Back-compat: callers that predate AIPerfSweep support expect a single
# EXPECTED_KIND constant.
EXPECTED_KIND = KIND_AIPERFJOB
K8S_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$")
K8S_NAME_MAX_LENGTH = 253

# Deployment/operator fields that live at the top-level spec (camelCase).
# Must stay in sync with AIPerfJobSpecConverter.to_deployment_config() and the
# CRD's spec.properties schema in deploy/helm/aiperf-operator/templates/crd-aiperfjob.yaml.
_DEPLOYMENT_FIELDS = {
    "image",
    "imagePullPolicy",
    "keepFailedPods",
    "resourceMode",
    "connectionsPerWorker",
    "timeoutSeconds",
    "ttlSecondsAfterFinished",
    "resultsTtlDays",
    "cancel",
    "podTemplate",
    "scheduling",
    "skipEndpointCheck",
    "failurePolicy",
}

# Envelope fields that AIPerfWorkloadSpec accepts at top-level spec (camelCase
# wire form). Both kinds accept these; AIPerfJob.sweep must be null while
# AIPerfSweep.sweep must be set — that distinction is enforced separately.
# childMetadata is AIPerfSweep-only; AIPerfJobSpec rejects it via
# ``extra='forbid'`` rather than via this gate.
_ENVELOPE_FIELDS = {
    "sweep",
    "multiRun",
    "variables",
    "randomSeed",
    "childMetadata",
}

# Top-level spec fields: deployment + envelope + benchmark.
KNOWN_SPEC_FIELDS = _DEPLOYMENT_FIELDS | _ENVELOPE_FIELDS | {"benchmark"}


def validate_cr(kind: str, spec: dict[str, Any]) -> AIPerfJobSpec | AIPerfSweepSpec:
    """Validate a CR spec dict against the kind-specific Pydantic schema.

    Routes to ``AIPerfJobSpec`` or ``AIPerfSweepSpec`` based on the CR's
    ``kind:`` line. Used by the validate CLI and any caller that has the
    raw ``kind`` + ``spec`` dict already split out (e.g. an admission path
    or a parsed ``aiperf kube apply`` payload).

    Args:
        kind: CR kind from the YAML's top-level ``kind:`` field
            (e.g. ``"AIPerfJob"`` or ``"AIPerfSweep"``).
        spec: Raw CR spec dict (the ``spec:`` mapping from the YAML).

    Returns:
        The validated ``AIPerfJobSpec`` or ``AIPerfSweepSpec`` instance.

    Raises:
        ValueError: ``kind`` is not one of the recognised CR kinds.
        pydantic.ValidationError: ``spec`` violates the kind-specific schema
            (e.g. AIPerfJob with a non-null ``sweep`` block, or AIPerfSweep
            without a ``sweep`` block).
    """
    if kind == "AIPerfJob":
        return AIPerfJobSpec.model_validate(spec)
    if kind == "AIPerfSweep":
        return AIPerfSweepSpec.model_validate(spec)
    raise ValueError(
        f"Unknown CR kind: {kind!r}. Expected 'AIPerfJob' or 'AIPerfSweep'."
    )


@dataclass(slots=True)
class ValidationResult:
    """Result of validating a single AIPerfJob YAML file."""

    path: Path
    """Filesystem path to the validated YAML file."""

    errors: list[str] = field(default_factory=list)
    """Validation errors that must be resolved before deployment."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal validation warnings."""

    @property
    def passed(self) -> bool:
        """Return True if no errors."""
        return len(self.errors) == 0


def validate_yaml_structure(doc: dict[str, Any], result: ValidationResult) -> bool:
    """Validate top-level YAML structure. Returns False if structure is too broken to continue.

    Accepts either ``kind: AIPerfJob`` or ``kind: AIPerfSweep``. Kind-specific
    invariants (sweep block presence/absence) are enforced by ``validate_file``.
    """
    if not isinstance(doc, dict):
        result.errors.append("Document is not a YAML mapping")
        return False

    api_version = doc.get("apiVersion")
    if api_version != EXPECTED_API_VERSION:
        result.errors.append(
            f"apiVersion: expected '{EXPECTED_API_VERSION}', got '{api_version}'"
        )

    kind = doc.get("kind")
    if kind not in SUPPORTED_KINDS:
        result.errors.append(
            f"kind: expected one of {sorted(SUPPORTED_KINDS)}, got '{kind}'"
        )

    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        result.errors.append("metadata: missing or not a mapping")
        return False

    if "name" not in metadata:
        result.errors.append("metadata.name: required field missing")
        return False

    spec = doc.get("spec")
    if not isinstance(spec, dict):
        result.errors.append("spec: missing or not a mapping")
        return False

    # spec must have a benchmark section
    benchmark = spec.get("benchmark")
    if not isinstance(benchmark, dict):
        result.errors.append(
            "spec.benchmark: required — put AIPerfConfig fields (models, endpoint, datasets, phases) here"
        )
        return False

    has_config = any(k in benchmark for k in ("models", "endpoint"))
    if not has_config:
        result.errors.append(
            "spec.benchmark: must contain at least 'models' or 'endpoint'"
        )
        return False

    return True


def validate_k8s_name(name: str, result: ValidationResult) -> None:
    """Validate metadata.name is a valid Kubernetes resource name."""
    if len(name) > K8S_NAME_MAX_LENGTH:
        result.errors.append(
            f"metadata.name: length {len(name)} exceeds max {K8S_NAME_MAX_LENGTH}"
        )
    if not K8S_NAME_PATTERN.match(name):
        result.errors.append(
            f"metadata.name: '{name}' is not a valid Kubernetes resource name "
            "(must match [a-z0-9][a-z0-9-]*[a-z0-9])"
        )


def validate_unknown_spec_fields(
    spec: dict[str, Any], result: ValidationResult, strict: bool
) -> None:
    """Check for unknown top-level spec fields and unknown spec.benchmark fields."""
    unknown_top = set(spec.keys()) - KNOWN_SPEC_FIELDS
    if unknown_top:
        msg = (
            f"Unknown spec fields (did you mean to put these under spec.benchmark?): "
            f"{', '.join(sorted(unknown_top))}"
        )
        if strict:
            result.errors.append(msg)
        else:
            result.warnings.append(msg)

    benchmark = spec.get("benchmark") or {}
    unknown_benchmark = set(benchmark.keys()) - CONFIG_FIELDS
    if unknown_benchmark:
        msg = f"Unknown spec.benchmark fields: {', '.join(sorted(unknown_benchmark))}"
        if strict:
            result.errors.append(msg)
        else:
            result.warnings.append(msg)


def validate_aiperf_config(
    spec: dict[str, Any], name: str, result: ValidationResult
) -> None:
    """Validate spec via AIPerfJobSpecConverter (flat spec, no userConfig wrapper)."""
    try:
        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        config = converter.to_aiperf_config()
    except (pydantic.ValidationError, ValueError, TypeError, KeyError) as e:
        result.errors.append(f"Config validation failed: {e}")
        return

    if not config.benchmark.get_model_names():
        result.errors.append("models: must not be empty")

    if not config.benchmark.endpoint.urls:
        result.errors.append("endpoint.urls: must not be empty")

    for url in config.benchmark.endpoint.urls:
        if not url.startswith(("http://", "https://")):
            result.errors.append(
                f"endpoint.urls: '{url}' must start with http:// or https://"
            )


def validate_deployment_config(
    spec: dict[str, Any], name: str, result: ValidationResult
) -> None:
    """Validate DeploymentConfig extraction."""
    try:
        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        converter.to_deployment_config()
    except (pydantic.ValidationError, ValueError, TypeError, KeyError) as e:
        result.errors.append(f"DeploymentConfig validation failed: {e}")


def validate_worker_count(
    spec: dict[str, Any], name: str, result: ValidationResult
) -> None:
    """Validate worker count calculation."""
    try:
        converter = AIPerfJobSpecConverter(spec=spec, name=name, namespace="default")
        workers = converter.calculate_workers()
        if workers < 1:
            result.errors.append(
                f"Worker count calculation returned {workers}, expected >= 1"
            )
    except (pydantic.ValidationError, ValueError, TypeError, KeyError) as e:
        result.errors.append(f"Worker calculation failed: {e}")


def validate_kind_sweep_cardinality(
    kind: str, spec: dict[str, Any], result: ValidationResult
) -> None:
    """Enforce the kind/sweep cardinality contract.

    AIPerfJob.spec.sweep MUST be null/omitted; AIPerfSweep.spec.sweep MUST be
    set and non-empty. This mirrors the CEL ``x-kubernetes-validations`` rules
    on each CRD so local ``aiperf kube validate`` surfaces the same error the
    apiserver would on apply.
    """
    sweep = spec.get("sweep")
    if kind == KIND_AIPERFJOB:
        if sweep is not None:
            result.errors.append(
                "spec.sweep: must be null on AIPerfJob; move the `sweep:` block "
                "to an AIPerfSweep CR or drop it."
            )
        return

    if kind == KIND_AIPERFSWEEP:
        if sweep is None:
            result.errors.append(
                "spec.sweep: required on AIPerfSweep; set a `sweep:` block "
                "(grid, scenarios, or sequential). For a single benchmark, use "
                "AIPerfJob instead."
            )
            return
        if not isinstance(sweep, dict) or not sweep:
            result.errors.append(
                "spec.sweep: AIPerfSweep requires a non-empty sweep mapping; "
                "got an empty/invalid value."
            )


def validate_kind_spec(
    kind: str, spec: dict[str, Any], result: ValidationResult
) -> None:
    """Validate the spec dict against the kind-specific Pydantic schema.

    Catches every cross-field invariant defined on ``AIPerfJobSpec`` /
    ``AIPerfSweepSpec`` (including the sweep-cardinality rule, image-non-empty,
    failure-policy shape, ...) by running ``model_validate`` over a copy of
    the spec stripped of unknown top-level keys (those are already surfaced
    by ``validate_unknown_spec_fields`` and would otherwise trigger redundant
    pydantic ``extra_forbidden`` errors).
    """
    filtered = {k: v for k, v in spec.items() if k in KNOWN_SPEC_FIELDS}
    try:
        validate_cr(kind, filtered)
    except pydantic.ValidationError as e:
        result.errors.append(f"{kind} spec validation failed: {e}")
    except ValueError as e:
        result.errors.append(str(e))


def validate_file(path: Path, *, strict: bool = False) -> ValidationResult:
    """Validate a single AIPerfJob or AIPerfSweep YAML file.

    Dispatches by ``doc['kind']``: shared checks (k8s name, unknown fields,
    AIPerfConfig benchmark body, deployment, worker count) run for both
    kinds; sweep-cardinality and per-kind Pydantic schema validation are
    layered on top.

    Args:
        path: Path to the YAML file.
        strict: If True, unknown spec fields are errors.

    Returns:
        ValidationResult with errors and warnings.
    """
    result = ValidationResult(path=path)

    if not path.exists():
        result.errors.append(f"File does not exist: {path}")
        return result

    if not path.is_file():
        result.errors.append(f"Not a file: {path}")
        return result

    try:
        text = path.read_text()
    except OSError as e:
        result.errors.append(f"Cannot read file: {e}")
        return result

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        result.errors.append(f"YAML parse error: {e}")
        return result

    if not validate_yaml_structure(doc, result):
        return result

    kind = doc.get("kind")
    name = doc["metadata"]["name"]
    spec = doc["spec"]

    validate_k8s_name(name, result)
    validate_unknown_spec_fields(spec, result, strict=strict)
    validate_aiperf_config(spec, name, result)
    validate_deployment_config(spec, name, result)
    validate_worker_count(spec, name, result)

    # Kind-specific checks come last so the user sees structural issues first.
    if kind in SUPPORTED_KINDS:
        validate_kind_sweep_cardinality(kind, spec, result)
        validate_kind_spec(kind, spec, result)

    return result


async def validate_files(files: list[Path], *, strict: bool = False) -> tuple[int, int]:
    """Validate multiple AIPerfJob YAML files and print results.

    Args:
        files: List of file paths to validate.
        strict: If True, unknown spec fields are errors.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    passed = 0
    failed = 0

    for path in files:
        result = validate_file(path, strict=strict)

        if result.passed:
            passed += 1
            kube_console.print_success(f"{path}")
        else:
            failed += 1
            kube_console.print_error(f"{path}")
            for error in result.errors:
                kube_console.logger.info(f"  [red]ERROR:[/red] {error}")

        for warning in result.warnings:
            kube_console.logger.info(f"  [yellow]WARN:[/yellow] {warning}")

    kube_console.logger.info("")
    total = passed + failed
    if failed == 0:
        kube_console.print_success(f"All {total} file(s) passed validation")
    else:
        kube_console.print_error(f"{failed}/{total} file(s) failed validation")

    return passed, failed
