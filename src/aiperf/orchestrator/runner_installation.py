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
        completed = self.execute(request)
        response = _parse_validation_response(
            completed.stdout,
            benchmark_id=run.benchmark_id,
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
            f"aiperf-runner rejected authored run {run.benchmark_id!r} "
            f"(exit {completed.returncode}): {detail}"
        )


def _resolve_runner_binary(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    configured = os.environ.get(_RUNNER_ENV)
    if configured:
        candidates.append(Path(configured))
    discovered = shutil.which("aiperf-runner")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise FileNotFoundError(
        "aiperf-runner executable was not found; install the native runner beside "
        f"aiperf or set {_RUNNER_ENV} to its absolute path"
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
        raise ValueError(f"aiperf-runner returned invalid validation JSON: {error}") from error
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
            raise ValueError(f"aiperf-runner validation {field} must be an array of objects")
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

    endpoints = run.get("endpoints")
    profiles = endpoints.get("profiles") if isinstance(endpoints, dict) else None
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("protocol-v2 request requires at least one endpoint profile")
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict) or not isinstance(profile.get("type"), str):
            raise ValueError(f"protocol-v2 endpoint profile {index} omitted type")
        _require_capability(capabilities, "endpoint_types", profile["type"])


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
