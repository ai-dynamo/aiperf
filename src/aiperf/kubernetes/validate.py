# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core validation logic for AIPerfJob YAML files.

The CRD spec is nested: AIPerfConfig fields live under spec.benchmark and
deployment fields (image, podTemplate, etc.) live at the spec level.
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
from aiperf.operator.spec_converter import CONFIG_FIELDS, AIPerfJobSpecConverter

EXPECTED_API_VERSION = AIPERF_API_VERSION
EXPECTED_KIND = "AIPerfJob"
K8S_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$")
K8S_NAME_MAX_LENGTH = 253

# Deployment/operator fields that live at the top-level spec (camelCase)
_DEPLOYMENT_FIELDS = {
    "image",
    "imagePullPolicy",
    "resourceMode",
    "connectionsPerWorker",
    "timeoutSeconds",
    "ttlSecondsAfterFinished",
    "resultsTtlDays",
    "cancel",
    "podTemplate",
    "scheduling",
    "sweepExecution",
}

# Top-level spec fields: deployment fields + the benchmark key
KNOWN_SPEC_FIELDS = _DEPLOYMENT_FIELDS | {"benchmark"}


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
    """Validate top-level YAML structure. Returns False if structure is too broken to continue."""
    if not isinstance(doc, dict):
        result.errors.append("Document is not a YAML mapping")
        return False

    api_version = doc.get("apiVersion")
    if api_version != EXPECTED_API_VERSION:
        result.errors.append(
            f"apiVersion: expected '{EXPECTED_API_VERSION}', got '{api_version}'"
        )

    kind = doc.get("kind")
    if kind != EXPECTED_KIND:
        result.errors.append(f"kind: expected '{EXPECTED_KIND}', got '{kind}'")

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

    if not config.get_model_names():
        result.errors.append("models: must not be empty")

    if not config.endpoint.urls:
        result.errors.append("endpoint.urls: must not be empty")

    for url in config.endpoint.urls:
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


def validate_file(path: Path, *, strict: bool = False) -> ValidationResult:
    """Validate a single AIPerfJob YAML file.

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

    name = doc["metadata"]["name"]
    spec = doc["spec"]

    validate_k8s_name(name, result)
    validate_unknown_spec_fields(spec, result, strict=strict)
    validate_aiperf_config(spec, name, result)
    validate_deployment_config(spec, name, result)
    validate_worker_count(spec, name, result)

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
