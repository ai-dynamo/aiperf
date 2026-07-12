# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery and exact-binary capability checks for ``aiperf-runner``.

Endpoint identity is owned by the selected native runner. This module is the
only Python authority for locating that runner, verifying its advertised
identity against the selected executable's complete bytes, and reading its
catalog; it deliberately has no plugin-registry or endpoint-metadata fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from hmac import compare_digest
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from blake3 import blake3

from aiperf.common.redact import redact_string
from aiperf.orchestrator.rust_wire import (
    RUNNER_PROTOCOL_V2,
    RUNNER_PROTOCOL_VERSION,
    RunnerOperationV2,
    build_authored_run_request,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun

_RUNNER_ENV = "AIPERF_RUNNER_BIN"
_RUNNER_COMPANION_DISTRIBUTION = "aiperf-runner"
_RUNNER_COMMAND = "aiperf-runner"
_NATIVE_REPORT_SCHEMA_VERSION = "2.0"
_DISTRIBUTION_ID_DOMAIN = b"aiperf-runner-distribution-v1\0"
_DISTRIBUTION_ID_PREFIX = "blake3:"
_DISTRIBUTION_ID_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class RunnerInstallation:
    """One selected runner binary and the capabilities read from that binary."""

    binary: Path
    capabilities: dict[str, Any]

    @classmethod
    def resolve(cls, binary: Path | None = None) -> RunnerInstallation:
        """Discover one runner and negotiate its capability contract once."""
        resolved = _resolve_runner_binary(binary)
        return cls(binary=resolved, capabilities=_load_capabilities(resolved))

    def preflight_endpoint(self, endpoint_id: str) -> None:
        """Reject an endpoint absent from this exact compiled runner catalog."""
        available = self.capabilities.get("endpoint_types")
        if not isinstance(available, list) or not all(
            isinstance(value, str) and value for value in available
        ):
            raise ValueError(
                f"selected aiperf-runner {self.binary} does not publish a usable "
                "endpoint_types catalog; install a compatible runner. Python "
                "endpoint metadata is not used as a fallback."
            )
        if endpoint_id in available:
            return
        choices = ", ".join(sorted(available)) or "<none>"
        raise RuntimeError(
            f"endpoint {endpoint_id!r} is not compiled into selected "
            f"aiperf-runner {self.binary}; available endpoints: {choices}. "
            "Select a runner distribution containing that endpoint (for example "
            f"with {_RUNNER_ENV})."
        )

    @property
    def distribution_id(self) -> str | None:
        """Return the exact identity advertised by this binary, if available.

        Capability negotiation verifies this value against the complete bytes
        of the selected executable. Directly constructed test installations may
        omit it, but discovered installations never do.
        """
        value = self.capabilities.get("distribution_id")
        return value if isinstance(value, str) and value else None

    def verify_distribution_identity(self) -> None:
        """Reject replacement of the negotiated runner image before launch."""
        advertised = self.distribution_id
        if advertised is None:
            raise RuntimeError(
                f"selected aiperf-runner {self.binary} omitted distribution_id; "
                "install a runner that publishes executable-content identity"
            )
        actual = _runner_distribution_id(self.binary)
        if not compare_digest(advertised, actual):
            raise RuntimeError(
                f"selected aiperf-runner {self.binary} no longer matches its "
                "negotiated distribution_id; the executable was replaced"
            )

    def project_authored_request(
        self,
        run: BenchmarkRun,
        *,
        operation: RunnerOperationV2,
    ) -> dict[str, Any]:
        """Build a v2 request bound to this installation without executing it."""
        versions = self.capabilities.get("protocol_versions")
        if not isinstance(versions, list) or RUNNER_PROTOCOL_V2 not in versions:
            raise RuntimeError(
                f"selected aiperf-runner {self.binary} does not support protocol "
                f"{RUNNER_PROTOCOL_V2}; advertised {versions!r}"
            )
        distribution_id = self.distribution_id
        if distribution_id is None:
            raise RuntimeError(
                f"selected aiperf-runner {self.binary} advertises protocol "
                f"{RUNNER_PROTOCOL_V2} without distribution_id; upgrade the runner. "
                "Python will not invent a fallback identity."
            )
        return build_authored_run_request(
            run,
            operation=operation,
            expected_distribution_id=distribution_id,
        )

    def supports_pair(self, backend_id: str, workload_id: str) -> bool:
        """Return whether this exact image advertises an executable v2 pair."""
        versions = self.capabilities.get("protocol_versions")
        if not isinstance(versions, list) or RUNNER_PROTOCOL_V2 not in versions:
            return False
        supported = self.capabilities.get("supported_pairs")
        if not isinstance(supported, list):
            return False
        return [backend_id, workload_id] in supported

    def preflight_plan(self, plan: BenchmarkPlan) -> None:
        """Validate every distinct fixed-plan endpoint before its first run."""
        endpoint_ids = {str(config.endpoint.type) for config in plan.configs}
        endpoint_ids.update(
            str(profile.type)
            for config in plan.configs
            for profile in config.endpoint_profiles.values()
        )
        for endpoint_id in sorted(endpoint_ids):
            self.preflight_endpoint(endpoint_id)

    def preflight_request(self, request: dict[str, Any]) -> None:
        """Validate a projected request against this installation's inventory."""
        protocol_version = request.get("protocol_version")
        if protocol_version == RUNNER_PROTOCOL_VERSION:
            _require_request_capabilities(self.capabilities, request)
            return
        if protocol_version == RUNNER_PROTOCOL_V2:
            _require_v2_request_capabilities(self.capabilities, request)
            return
        raise ValueError(
            f"native request protocol_version must be {RUNNER_PROTOCOL_VERSION} or "
            f"{RUNNER_PROTOCOL_V2}, got {protocol_version!r}"
        )

    def execute(self, request: dict[str, Any]) -> subprocess.CompletedProcess[bytes]:
        """Run one request with the same binary whose catalog was negotiated."""
        self.preflight_request(request)
        self.verify_distribution_identity()
        return subprocess.run(
            [str(self.binary)],
            input=orjson.dumps(request),
            capture_output=True,
            check=False,
        )

    def validate_authored_run(self, run: BenchmarkRun) -> dict[str, Any]:
        """Run strict side-effect-free validation in one fresh native child."""
        request = self.project_authored_request(run, operation="validate")
        return self.validate_authored_request(
            request,
            benchmark_id=run.benchmark_id,
        )

    def validate_authored_request(
        self,
        request: dict[str, Any],
        *,
        benchmark_id: str,
    ) -> dict[str, Any]:
        """Validate one already-projected v2 request without projecting it again."""
        if request.get("protocol_version") != RUNNER_PROTOCOL_V2:
            raise ValueError("authored validation requires a protocol-v2 request")
        if request.get("operation") != "validate":
            raise ValueError("authored validation requires operation='validate'")
        completed = self.execute(request)
        response = _parse_validation_response(
            completed.stdout,
            benchmark_id=benchmark_id,
            distribution_id=request["expected_distribution_id"],
            returncode=completed.returncode,
            stderr=completed.stderr,
        )
        if response["success"]:
            return response
        messages = [error["message"] for error in response["errors"]]
        detail = redact_string("; ".join(messages) or "native validation failed")
        stderr = redact_string(completed.stderr.decode(errors="replace")).strip()
        if stderr:
            detail = f"{detail}; Rust stderr: {stderr[-4000:]}"
        raise RuntimeError(
            f"aiperf-runner rejected authored run {benchmark_id!r} "
            f"(exit {completed.returncode}): {detail}"
        )


def _resolve_runner_binary(explicit: Path | None) -> Path:
    if explicit is not None:
        return _require_runner_binary(Path(explicit), "explicit --runner-bin")

    configured = os.environ.get(_RUNNER_ENV)
    if configured:
        return _require_runner_binary(Path(configured), _RUNNER_ENV)

    companion = _installed_companion_binary()
    if companion is not None:
        return companion

    discovered = shutil.which(_RUNNER_COMMAND)
    if discovered:
        return _require_runner_binary(Path(discovered), "PATH")

    raise FileNotFoundError(
        "aiperf-runner executable was not found; install the platform companion "
        f"package {_RUNNER_COMPANION_DISTRIBUTION!r}, pass --runner-bin, set "
        f"{_RUNNER_ENV}, or place {_RUNNER_COMMAND} on PATH for development"
    )


def _installed_companion_binary() -> Path | None:
    """Locate the native script installed by the platform companion wheel.

    The wheel contains the Rust executable as wheel ``scripts`` data. Discovery
    reads its installed RECORD through ``importlib.metadata``; it never imports
    a Python shim and does not use PATH for this precedence tier.
    """
    try:
        distribution = metadata.distribution(_RUNNER_COMPANION_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None

    files = distribution.files
    if files is None:
        raise RuntimeError(
            f"installed companion package {_RUNNER_COMPANION_DISTRIBUTION!r} "
            "does not expose its installed file RECORD"
        )
    filenames = {_RUNNER_COMMAND, f"{_RUNNER_COMMAND}.exe"}
    entries = sorted(
        (entry for entry in files if Path(str(entry)).name in filenames),
        key=str,
    )
    if len(entries) != 1:
        raise RuntimeError(
            f"installed companion package {_RUNNER_COMPANION_DISTRIBUTION!r} "
            f"must contain exactly one native {_RUNNER_COMMAND} executable; "
            f"found {len(entries)}"
        )
    candidate = Path(distribution.locate_file(entries[0]))
    return _require_runner_binary(
        candidate,
        f"installed companion package {_RUNNER_COMPANION_DISTRIBUTION!r}",
    )


def _require_runner_binary(candidate: Path, source: str) -> Path:
    """Validate one selected precedence tier without falling through."""
    resolved = candidate.expanduser().resolve()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return resolved
    raise FileNotFoundError(
        f"{source} selected aiperf-runner {resolved}, but it is not an "
        "executable file; refusing to substitute a lower-precedence runner"
    )


def _load_capabilities(binary: Path) -> dict[str, Any]:
    expected_distribution_id = _runner_distribution_id(binary)
    completed = subprocess.run(
        [str(binary), "--capabilities"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = redact_string(completed.stderr.decode(errors="replace")).strip()
        raise RuntimeError(
            f"aiperf-runner capability negotiation failed (exit "
            f"{completed.returncode}): {stderr or 'no diagnostic'}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            "aiperf-runner --capabilities must write exactly one JSON line; "
            f"received {len(lines)}"
        )
    try:
        capabilities = orjson.loads(lines[0])
    except orjson.JSONDecodeError as error:
        raise ValueError(
            f"aiperf-runner returned invalid capability JSON: {error}"
        ) from error
    if not isinstance(capabilities, dict):
        raise ValueError("aiperf-runner capabilities must be an object")
    if capabilities.get("event") != "runner_capabilities":
        raise ValueError("aiperf-runner returned an unknown capability response")
    distribution_id = capabilities.get("distribution_id")
    if not _is_distribution_id(distribution_id):
        raise ValueError(
            "aiperf-runner capability distribution_id must be 'blake3:' followed "
            "by exactly 64 lowercase hexadecimal characters"
        )
    if not compare_digest(distribution_id, expected_distribution_id):
        raise RuntimeError(
            f"aiperf-runner capability distribution_id does not match the exact "
            f"selected executable bytes at {binary}; refusing a mixed runner "
            "distribution"
        )
    versions = capabilities.get("protocol_versions")
    if not isinstance(versions, list) or RUNNER_PROTOCOL_VERSION not in versions:
        raise RuntimeError(
            f"aiperf-runner does not support protocol {RUNNER_PROTOCOL_VERSION}: "
            f"advertised {versions!r}"
        )
    if RUNNER_PROTOCOL_V2 in versions:
        _validate_v2_capabilities(capabilities)
    schema = capabilities.get("report_schema_version")
    if schema != _NATIVE_REPORT_SCHEMA_VERSION:
        raise RuntimeError(
            f"aiperf-runner report schema {schema!r} is incompatible; "
            f"expected {_NATIVE_REPORT_SCHEMA_VERSION!r}"
        )
    for field in (
        "endpoint_types",
        "dataset_types",
        "phase_types",
        "phase_features",
        "run_features",
        "telemetry_source_types",
        "server_metrics_formats",
    ):
        values = capabilities.get(field)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            detail = (
                "; install a compatible catalog-publishing runner because Python "
                "endpoint metadata is not used as a fallback"
                if field == "endpoint_types"
                else ""
            )
            raise ValueError(
                f"aiperf-runner capability {field} must be an array of "
                f"non-empty strings{detail}"
            )
    return capabilities


def _validate_v2_capabilities(capabilities: dict[str, Any]) -> None:
    """Validate inventories required to select a protocol-v2 execution pair."""
    if capabilities.get("capabilities_schema_version") != 2:
        raise RuntimeError(
            "aiperf-runner advertises protocol 2 without capability schema 2"
        )
    for field in ("supported_pairs", "statically_compatible_pairs"):
        pairs = capabilities.get(field)
        if not isinstance(pairs, list) or not all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(value, str) and value for value in pair)
            for pair in pairs
        ):
            raise ValueError(
                f"aiperf-runner capability {field} must be an array of "
                "[backend, workload] string pairs"
            )
    for field in ("backends", "workloads", "endpoints"):
        descriptors = capabilities.get(field)
        if not isinstance(descriptors, list) or not all(
            isinstance(descriptor, dict)
            and isinstance(descriptor.get("id"), str)
            and bool(descriptor["id"])
            for descriptor in descriptors
        ):
            raise ValueError(
                f"aiperf-runner capability {field} must be an array of "
                "descriptors with non-empty id fields"
            )
    extensions = capabilities.get("extensions")
    if not isinstance(extensions, list) or not all(
        isinstance(extension, str) and extension for extension in extensions
    ):
        raise ValueError(
            "aiperf-runner capability extensions must be an array of non-empty strings"
        )
    _validate_evaluation_capabilities(capabilities)


def _validate_evaluation_capabilities(capabilities: dict[str, Any]) -> None:
    """Validate executable-only evaluator/provider capability inventories."""
    workloads = capabilities.get("workloads", [])
    evaluation_registered = any(
        isinstance(workload, dict) and workload.get("id") == "evaluation"
        for workload in workloads
    )
    fields = (
        "evaluation_providers",
        "evaluation_host_operations",
        "supported_evaluation_combinations",
    )
    if not evaluation_registered and not any(field in capabilities for field in fields):
        # Additive compatibility for protocol-v2 runners predating the
        # evaluation workload. They can never pass evaluation preflight.
        return
    for field in fields:
        if not isinstance(capabilities.get(field), list):
            raise ValueError(f"aiperf-runner capability {field} must be an array")

    for provider in capabilities["evaluation_providers"]:
        if not isinstance(provider, dict):
            raise ValueError("evaluation provider capabilities must be objects")
        _require_nonempty_strings(
            provider,
            ("id", "display_name", "config_schema_sha256", "isolation_profile_id"),
            "evaluation provider",
        )
        if not isinstance(provider.get("config_schema_version"), int) or (
            provider["config_schema_version"] <= 0
        ):
            raise ValueError(
                "evaluation provider config_schema_version must be positive"
            )
        for field in (
            "worker_protocol_versions",
            "execution_granularities",
            "scheduling_modes",
            "declared_operations",
            "distributions",
        ):
            if not isinstance(provider.get(field), list) or not provider[field]:
                raise ValueError(
                    f"evaluation provider capability {field} must be a non-empty array"
                )
        for distribution in provider["distributions"]:
            if not isinstance(distribution, dict):
                raise ValueError("evaluation distributions must be objects")
            _require_nonempty_strings(
                distribution,
                (
                    "id",
                    "package",
                    "package_version",
                    "provider_source_sha256",
                    "worker_source_sha256",
                    "dependency_lock_sha256",
                    "launch_closure_sha256",
                ),
                "evaluation distribution",
            )

    for operation in capabilities["evaluation_host_operations"]:
        if not isinstance(operation, dict):
            raise ValueError("evaluation host operation capabilities must be objects")
        _require_nonempty_strings(
            operation,
            ("id", "family", "request_schema_sha256", "response_schema_sha256"),
            "evaluation host operation",
        )
        if not isinstance(operation.get("true_streaming"), bool):
            raise ValueError(
                "evaluation host operation true_streaming must be a boolean"
            )
        if not isinstance(operation.get("endpoint_capabilities"), list) or not all(
            isinstance(value, str) and value
            for value in operation["endpoint_capabilities"]
        ):
            raise ValueError(
                "evaluation host operation endpoint_capabilities must be strings"
            )

    for combination in capabilities["supported_evaluation_combinations"]:
        if not isinstance(combination, dict):
            raise ValueError("supported evaluation combinations must be objects")
        _require_nonempty_strings(
            combination,
            (
                "backend",
                "workload",
                "provider",
                "distribution",
                "isolation_profile_id",
            ),
            "supported evaluation combination",
        )
        for field in ("operations", "resources"):
            if not isinstance(combination.get(field), list) or not all(
                isinstance(value, str) and value for value in combination[field]
            ):
                raise ValueError(
                    f"supported evaluation combination {field} must be strings"
                )


def _require_nonempty_strings(
    value: dict[str, Any], fields: tuple[str, ...], label: str
) -> None:
    for field in fields:
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"{label} {field} must be a non-empty string")


def _parse_validation_response(
    stdout: bytes,
    *,
    benchmark_id: str,
    distribution_id: str,
    returncode: int,
    stderr: bytes = b"",
) -> dict[str, Any]:
    """Decode and bind the exactly-one-line protocol-v2 validation response."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        diagnostic = redact_string(stderr.decode(errors="replace")).strip()
        detail = f"; stderr: {diagnostic}" if diagnostic else ""
        raise ValueError(
            "aiperf-runner validate must write exactly one JSON line; "
            f"received {len(lines)}; child exit code {returncode}{detail}"
        )
    try:
        response = orjson.loads(lines[0])
    except orjson.JSONDecodeError as error:
        raise ValueError(
            f"aiperf-runner returned invalid validation JSON: {error}"
        ) from error
    if not isinstance(response, dict):
        raise ValueError("aiperf-runner validation response must be an object")
    expected: dict[str, object] = {
        "protocol_version": RUNNER_PROTOCOL_V2,
        "event": "run_validation",
        "distribution_id": distribution_id,
        "benchmark_id": benchmark_id,
    }
    for field, value in expected.items():
        if response.get(field) != value:
            raise ValueError(
                f"aiperf-runner validation {field}={response.get(field)!r}; "
                f"expected {value!r}"
            )
    success = response.get("success")
    if not isinstance(success, bool):
        raise ValueError("aiperf-runner validation success must be a boolean")
    if response.get("completeness") not in {"static", "complete"}:
        raise ValueError(
            "aiperf-runner validation completeness must be 'static' or 'complete'"
        )
    for field in ("deferred_checks", "errors"):
        entries = response.get(field, [])
        if not isinstance(entries, list) or not all(
            isinstance(entry, dict) for entry in entries
        ):
            raise ValueError(
                f"aiperf-runner validation {field} must be an array of objects"
            )
    errors = response.get("errors", [])
    if not all(
        isinstance(error.get("code"), str)
        and bool(error["code"])
        and isinstance(error.get("message"), str)
        and bool(error["message"])
        for error in errors
    ):
        raise ValueError(
            "aiperf-runner validation errors require non-empty code and message strings"
        )
    if success != (returncode == 0):
        raise ValueError(
            "aiperf-runner validation success disagrees with child exit code "
            f"{returncode}"
        )
    if success and errors:
        raise ValueError("successful aiperf-runner validation cannot contain errors")
    if not success and not errors:
        raise ValueError("failed aiperf-runner validation must contain errors")
    return response


def _runner_distribution_id(binary: Path) -> str:
    """Hash one opened runner image with the native versioned BLAKE3 contract."""
    digest = blake3()
    digest.update(_DISTRIBUTION_ID_DOMAIN)
    try:
        with Path(binary).open("rb", buffering=0) as image:
            while chunk := image.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError(
            f"cannot read selected aiperf-runner image {binary} for distribution "
            f"identity: {error}"
        ) from error
    return f"{_DISTRIBUTION_ID_PREFIX}{digest.hexdigest()}"


def _is_distribution_id(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_DISTRIBUTION_ID_PREFIX):
        return False
    hexadecimal = value[len(_DISTRIBUTION_ID_PREFIX) :]
    return len(hexadecimal) == _DISTRIBUTION_ID_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in hexadecimal
    )


def _require_v2_request_capabilities(
    capabilities: dict[str, Any], request: dict[str, Any]
) -> None:
    """Fail before launch unless this image advertises the exact v2 pair."""
    run = request.get("run")
    if not isinstance(run, dict):
        raise ValueError("protocol-v2 request omitted its run object")
    backend = run.get("backend")
    workload = run.get("workload")
    if not isinstance(backend, dict) or not isinstance(backend.get("type"), str):
        raise ValueError("protocol-v2 request omitted run.backend.type")
    if not isinstance(workload, dict) or not isinstance(workload.get("type"), str):
        raise ValueError("protocol-v2 request omitted run.workload.type")
    pair = [backend["type"], workload["type"]]
    supported = capabilities.get("supported_pairs")
    if not isinstance(supported, list) or pair not in supported:
        raise RuntimeError(
            "selected aiperf-runner does not contain executable protocol-v2 pair "
            f"({pair[0]!r}, {pair[1]!r}); advertised {supported!r}"
        )

    if workload["type"] == "evaluation":
        _require_evaluation_selection_capability(
            capabilities, backend["type"], workload
        )

    resources = run.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("protocol-v2 request omitted run.resources")
    endpoints = resources.get("endpoints")
    if endpoints is None:
        return
    profiles = endpoints.get("profiles") if isinstance(endpoints, dict) else None
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("protocol-v2 request requires at least one endpoint profile")
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict) or not isinstance(profile.get("type"), str):
            raise ValueError(f"protocol-v2 endpoint profile {index} omitted type")
        _require_capability(capabilities, "endpoint_types", profile["type"])


def _require_evaluation_selection_capability(
    capabilities: dict[str, Any], backend_id: str, workload: dict[str, Any]
) -> None:
    """Require one exact executable provider/distribution combination."""
    config = workload.get("config")
    provider = config.get("provider") if isinstance(config, dict) else None
    if not isinstance(provider, dict):
        raise ValueError("evaluation workload omitted config.provider")
    provider_id = provider.get("type")
    distribution_id = provider.get("distribution")
    if not isinstance(provider_id, str) or not provider_id:
        raise ValueError("evaluation workload provider.type must be a non-empty string")
    if not isinstance(distribution_id, str) or not distribution_id:
        raise ValueError(
            "evaluation workload provider.distribution must be a non-empty string"
        )
    combinations = capabilities.get("supported_evaluation_combinations")
    if not isinstance(combinations, list):
        raise RuntimeError(
            "selected aiperf-runner does not publish executable evaluation combinations"
        )
    selected = next(
        (
            combination
            for combination in combinations
            if isinstance(combination, dict)
            and combination.get("backend") == backend_id
            and combination.get("workload") == "evaluation"
            and combination.get("provider") == provider_id
            and combination.get("distribution") == distribution_id
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(
            "selected aiperf-runner does not contain executable evaluation "
            f"provider/distribution ({provider_id!r}, {distribution_id!r}); no "
            "benchmark-name or provider fallback is permitted"
        )
    resources = config.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("evaluation workload resources must be an object")
    advertised_resources = selected.get("resources", [])
    for name, resource in resources.items():
        if not isinstance(resource, dict) or not isinstance(resource.get("type"), str):
            raise ValueError(f"evaluation resource {name!r} omitted type")
        if resource["type"] not in advertised_resources:
            raise RuntimeError(
                f"evaluation resource adapter {resource['type']!r} is not executable "
                f"for provider/distribution ({provider_id!r}, {distribution_id!r})"
            )


def _require_request_capabilities(
    capabilities: dict[str, Any], request: dict[str, Any]
) -> None:
    """Fail before launch when a resolved run exceeds the child contract."""
    run, endpoint, dataset, phases, artifacts = _request_components(request)
    _require_capability(capabilities, "run_features", "thread_per_core_execution")
    _require_endpoint_capabilities(capabilities, endpoint)
    _require_capability(capabilities, "dataset_types", dataset["type"])
    _require_phase_capabilities(capabilities, phases)
    _require_output_capabilities(capabilities, run, artifacts)
    _require_gpu_capabilities(capabilities, run.get("gpu_telemetry"))
    _require_network_capabilities(capabilities, run.get("network_latency"))
    _require_server_metrics_capabilities(capabilities, run.get("server_metrics"))
    _require_live_streaming_capabilities(capabilities, run.get("live_streaming"))


def _request_components(
    request: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[Any],
    dict[str, Any],
]:
    """Extract and structurally validate the native request components."""
    run = request.get("run")
    if not isinstance(run, dict):
        raise ValueError("native run request omitted its run object")
    endpoint = run.get("endpoint")
    dataset = run.get("dataset")
    phases = run.get("phases")
    artifacts = run.get("artifacts", {})
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get("type"), str):
        raise ValueError("native run request omitted endpoint.type")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("type"), str):
        raise ValueError("native run request omitted dataset.type")
    if not isinstance(phases, list) or not phases:
        raise ValueError("native run request must contain at least one phase")
    if not isinstance(artifacts, dict):
        raise ValueError("native run artifacts must be an object")
    workers = run.get("workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("native run workers must be a positive integer")
    return run, endpoint, dataset, phases, artifacts


def _require_endpoint_capabilities(
    capabilities: dict[str, Any], endpoint: dict[str, Any]
) -> None:
    """Validate endpoint identity and optional transport policy inventory."""
    _require_capability(capabilities, "endpoint_types", endpoint["type"])
    if any(
        field in endpoint
        for field in (
            "timeout_seconds",
            "connection_reuse",
            "request_content_type",
            "download_video_content",
            "session_header",
        )
    ):
        _require_capability(capabilities, "run_features", "http_transport_policy")


def _require_phase_capabilities(
    capabilities: dict[str, Any], phases: list[Any]
) -> None:
    """Validate each phase kind and its optional native control features."""
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict) or not isinstance(phase.get("type"), str):
            raise ValueError(f"native run phase {index} omitted type")
        _require_capability(capabilities, "phase_types", phase["type"])
        if "adaptive_scale" in phase:
            _require_capability(capabilities, "phase_features", "adaptive_scale")
        if any(
            field in phase
            for field in ("concurrency_ramp", "prefill_ramp", "rate_ramp")
        ):
            _require_capability(capabilities, "phase_features", "ramps")
        if "cancellation" in phase:
            _require_capability(capabilities, "phase_features", "request_cancellation")


def _require_output_capabilities(
    capabilities: dict[str, Any],
    run: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    """Validate accuracy and artifact features selected by the run."""
    if "accuracy" in run:
        _require_capability(capabilities, "run_features", "python_accuracy_evaluator")
    if "outputs_path" in artifacts:
        _require_capability(capabilities, "run_features", "outputs_json")
    if "raw_path" in artifacts:
        _require_capability(capabilities, "run_features", "raw_records")


def _require_gpu_capabilities(capabilities: dict[str, Any], gpu_telemetry: Any) -> None:
    """Validate optional GPU telemetry sources."""
    if gpu_telemetry is None:
        return
    _require_capability(capabilities, "run_features", "gpu_telemetry")
    if not isinstance(gpu_telemetry, dict):
        raise ValueError("native run gpu_telemetry must be an object")
    sources = gpu_telemetry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("native run GPU telemetry requires at least one source")
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not isinstance(source.get("type"), str):
            raise ValueError(f"native GPU telemetry source {index} omitted type")
        _require_capability(capabilities, "telemetry_source_types", source["type"])


def _require_network_capabilities(
    capabilities: dict[str, Any], network_latency: Any
) -> None:
    """Validate optional network-latency policy."""
    if network_latency is None:
        return
    if not isinstance(network_latency, dict):
        raise ValueError("native run network_latency must be an object")
    _require_capability(capabilities, "run_features", "network_latency")


def _require_server_metrics_capabilities(
    capabilities: dict[str, Any], server_metrics: Any
) -> None:
    """Validate optional server-metrics formats."""
    if server_metrics is None:
        return
    _require_capability(capabilities, "run_features", "server_metrics")
    if not isinstance(server_metrics, dict):
        raise ValueError("native run server_metrics must be an object")
    formats = server_metrics.get("formats")
    if not isinstance(formats, list) or not formats:
        raise ValueError("native run server_metrics requires at least one format")
    for format_name in formats:
        if not isinstance(format_name, str):
            raise ValueError("native server metrics formats must be strings")
        _require_capability(capabilities, "server_metrics_formats", format_name)


def _require_live_streaming_capabilities(
    capabilities: dict[str, Any], live_streaming: Any
) -> None:
    """Validate the optional supervised Python streaming extension."""
    if live_streaming is None:
        return
    if not isinstance(live_streaming, dict):
        raise ValueError("native run live_streaming must be an object")
    _require_capability(capabilities, "run_features", "python_live_streaming")


def _require_capability(
    capabilities: dict[str, Any], field: str, required: str
) -> None:
    advertised = capabilities[field]
    if required not in advertised:
        raise RuntimeError(
            f"aiperf-runner does not support {field}.{required}; "
            f"advertised {advertised!r}"
        )
