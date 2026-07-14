# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery and exact-binary catalog checks for ``aiperf-runner``.

Endpoint identity is owned by the selected native runner. This module is the
only Python authority for locating that runner and reading its linked
plugins.yaml-shaped catalog; it deliberately has no plugin-registry or
endpoint-metadata fallback. Distribution-id pinning is not part of the wire
contract.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from blake3 import blake3

from aiperf.common.redact import redact_string
from aiperf.orchestrator.rust_wire import (
    RUNNER_PROTOCOL_V2,
    RunnerOperationV2,
    build_authored_run_request,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun

_CAPABILITIES_TIMEOUT_SECONDS = 30.0
_RUNNER_ENV = "AIPERF_RUNNER_BIN"
_RUNNER_COMPANION_DISTRIBUTION = "aiperf-runner"
_RUNNER_COMMAND = "aiperf-runner"
_PROVIDER_ROOTS_ENV = "AIPERF_EVALUATOR_PROVIDER_ROOTS"
_PROVIDER_ROOTS_SCHEMA = "aiperf-stock-evaluator-roots-v1"
_PROVIDER_ROOTS_REGISTRY = "evaluator-roots-v1.json"
_PROVIDER_ROOTS_WHEEL_PREFIX = "_aiperf_runner/evaluator-roots"
_PROVIDER_ROOTS_SIDECAR_SUFFIX = ".evaluator-roots"
_PROVIDER_ROOT_SPECS = (
    ("cpython_3_12_10_linux_x86_64", "python_runtime", "runtime"),
    ("nvidia_nemo_evaluator_0_4_locked", "python_environment", "nemo"),
    (
        "groq_openbench_0_5_3_inspect_0_3_141_locked",
        "python_environment",
        "openbench",
    ),
    ("system_linux_x86_64", "system", "system"),
)
_EVALUATION_UNAVAILABLE_REASON_CODES = frozenset(
    {
        "provider_roots_unavailable",
        "unsupported_platform",
        "isolation_unavailable",
    }
)
_DISTRIBUTION_ID_DOMAIN = b"aiperf-runner-distribution-v1\0"
_DISTRIBUTION_ID_PREFIX = "blake3:"
_DISTRIBUTION_ID_HEX_LENGTH = 64
_REQUIRED_CATALOG_CATEGORIES = ("endpoint", "transport")
_OPTIONAL_CATALOG_CATEGORIES = (
    "custom_dataset_loader",
    "public_dataset_loader",
    "dataset_sampler",
    "synthetic",
)


@dataclass(frozen=True, slots=True)
class RunnerInstallation:
    """One selected runner binary and the catalog read from that binary."""

    binary: Path
    capabilities: dict[str, Any]
    provider_roots: tuple[Path, ...] = ()

    @classmethod
    def resolve(
        cls,
        binary: Path | None = None,
        *,
        provider_roots: Sequence[Path] | None = None,
    ) -> RunnerInstallation:
        """Discover one runner and negotiate its catalog once.

        ``provider_roots`` is an explicit test/deployment injection for
        mutually independent evaluator environments.  Product discovery does
        not inspect the active Python prefix or ambient root variables: an
        installed companion uses only its own wheel RECORD, while an explicit,
        environment-selected, or PATH runner uses only its generated adjacent
        sidecar.  Ambient child variables never broaden either selection.
        """
        if provider_roots is None:
            deployment = _resolve_runner_deployment(binary)
            resolved = deployment.binary
            selected_provider_roots = _deployment_provider_roots(deployment)
        else:
            resolved = _resolve_runner_binary(binary)
            selected_provider_roots = _normalize_provider_roots(provider_roots)
        return cls(
            binary=resolved,
            capabilities=_load_capabilities(resolved, selected_provider_roots),
            provider_roots=selected_provider_roots,
        )

    def preflight_endpoint(self, endpoint_id: str) -> None:
        """Reject an endpoint absent from this exact compiled runner catalog."""
        available = self.capabilities.get("endpoint")
        if not isinstance(available, dict) or not available:
            raise ValueError(
                f"selected aiperf-runner {self.binary} does not publish a usable "
                "endpoint catalog; install a compatible runner. Python "
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
        """Optional diagnostic hash; not part of the wire pin contract."""
        value = self.capabilities.get("distribution_id")
        return value if isinstance(value, str) and value else None

    def verify_distribution_identity(self) -> None:
        """No-op: distribution-id is not a wire pin for BenchmarkRun requests."""

    def project_authored_request(
        self,
        run: BenchmarkRun,
        *,
        operation: RunnerOperationV2,
    ) -> dict[str, Any]:
        """Build a v2 BenchmarkRun envelope without executing it."""
        return build_authored_run_request(run, operation=operation)

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
        """Validate a projected request against this installation's catalog."""
        protocol_version = request.get("protocol_version")
        if protocol_version != RUNNER_PROTOCOL_V2:
            raise ValueError(
                f"native request protocol_version must be {RUNNER_PROTOCOL_V2}, "
                f"got {protocol_version!r}"
            )
        _require_v2_request_capabilities(self.capabilities, request)

    def execute(self, request: dict[str, Any]) -> subprocess.CompletedProcess[bytes]:
        """Run one request with the same binary whose catalog was negotiated."""
        child = self.spawn(request)
        stdout, stderr = child.communicate(orjson.dumps(request))
        return subprocess.CompletedProcess(
            child.args,
            child.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def spawn(self, request: dict[str, Any]) -> subprocess.Popen[bytes]:
        """Start one request so a lifecycle owner can forward signals.

        The returned child has private stdin/stdout/stderr pipes. Callers must
        send exactly ``orjson.dumps(request)`` to ``communicate``. Lifecycle
        owners use this seam to forward the first SIGINT/SIGTERM for native
        graceful shutdown.
        """
        self.preflight_request(request)
        return subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_runner_subprocess_environment(self.provider_roots),
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


@dataclass(frozen=True, slots=True)
class _RunnerDeployment:
    """One selected executable and its sole deployment-metadata authority."""

    binary: Path
    companion_distribution: metadata.Distribution | None = None


def _resolve_runner_deployment(explicit: Path | None) -> _RunnerDeployment:
    """Resolve one precedence tier while retaining its exact metadata owner."""
    if explicit is not None:
        return _RunnerDeployment(
            _require_runner_binary(Path(explicit), "explicit --runner-bin")
        )

    configured = os.environ.get(_RUNNER_ENV)
    if configured:
        return _RunnerDeployment(_require_runner_binary(Path(configured), _RUNNER_ENV))

    distribution = _installed_companion_distribution()
    if distribution is not None:
        return _RunnerDeployment(
            _companion_binary_from_distribution(distribution),
            companion_distribution=distribution,
        )

    discovered = shutil.which(_RUNNER_COMMAND)
    if discovered:
        return _RunnerDeployment(_require_runner_binary(Path(discovered), "PATH"))

    raise FileNotFoundError(
        "aiperf-runner executable was not found; install the platform companion "
        f"package {_RUNNER_COMPANION_DISTRIBUTION!r}, pass --runner-bin, set "
        f"{_RUNNER_ENV}, or place {_RUNNER_COMMAND} on PATH for development"
    )


def _resolve_runner_binary(explicit: Path | None) -> Path:
    """Compatibility helper returning only the selected executable path."""
    return _resolve_runner_deployment(explicit).binary


def _installed_companion_binary() -> Path | None:
    """Locate the native script installed by the platform companion wheel.

    The wheel contains the Rust executable as wheel ``scripts`` data. Discovery
    reads its installed RECORD through ``importlib.metadata``; it never imports
    a Python shim and does not use PATH for this precedence tier.
    """
    distribution = _installed_companion_distribution()
    if distribution is None:
        return None

    return _companion_binary_from_distribution(distribution)


def _installed_companion_distribution() -> metadata.Distribution | None:
    """Return the selected companion distribution without importing from it."""
    try:
        return metadata.distribution(_RUNNER_COMPANION_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def _companion_binary_from_distribution(
    distribution: metadata.Distribution,
) -> Path:
    """Resolve the sole native script recorded by one companion distribution."""

    files = distribution.files
    if files is None:
        raise RuntimeError(
            f"installed companion package {_RUNNER_COMPANION_DISTRIBUTION!r} "
            "does not expose its installed file RECORD"
        )
    filenames = {_RUNNER_COMMAND, f"{_RUNNER_COMMAND}.exe"}
    entries = sorted(
        (
            entry
            for entry in files
            if not str(entry)
            .replace("\\", "/")
            .startswith(f"{_PROVIDER_ROOTS_WHEEL_PREFIX}/")
            and str(entry).replace("\\", "/").rsplit("/", 1)[-1] in filenames
        ),
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


def _deployment_provider_roots(
    deployment: _RunnerDeployment,
) -> tuple[Path, ...]:
    """Discover roots owned by the selected runner deployment only.

    An implicit installed-companion selection consumes only that same wheel's
    RECORD-owned payload. Explicit, ``AIPERF_RUNNER_BIN``, and PATH selections
    consume only a generated directory adjacent to the selected executable.
    Missing or invalid deployment metadata intentionally produces no roots.
    """
    if deployment.companion_distribution is not None:
        return _installed_companion_provider_roots(deployment.companion_distribution)
    binary = deployment.binary
    sidecar = binary.with_name(f"{binary.name}{_PROVIDER_ROOTS_SIDECAR_SUFFIX}")
    return _provider_roots_from_registry(sidecar)


def _installed_companion_provider_roots(
    distribution: metadata.Distribution,
) -> tuple[Path, ...]:
    """Validate exact payload membership in the selected wheel RECORD."""
    files = distribution.files
    if files is None:
        return ()
    prefix = f"{_PROVIDER_ROOTS_WHEEL_PREFIX}/"
    registry_relative = f"{prefix}{_PROVIDER_ROOTS_REGISTRY}"
    entries: dict[str, metadata.PackagePath] = {}
    for entry in files:
        normalized = str(entry).replace("\\", "/")
        if not normalized.startswith(prefix):
            continue
        if normalized in entries:
            return ()
        entries[normalized] = entry
    registry_entry = entries.get(registry_relative)
    if registry_entry is None:
        return ()
    try:
        located_registry = Path(distribution.locate_file(registry_entry))
        if located_registry.is_symlink() or located_registry.parent.is_symlink():
            raise ValueError("evaluator payload RECORD root cannot be a symlink")
        registry_path = located_registry.resolve(strict=True)
        base = registry_path.parent
        recorded: dict[str, tuple[str, int]] = {}
        for relative, entry in entries.items():
            logical = relative.removeprefix(prefix)
            if not logical or logical in recorded:
                raise ValueError("duplicate or empty evaluator payload RECORD path")
            file_hash = entry.hash
            if file_hash is None or file_hash.mode != "sha256":
                raise ValueError("evaluator payload RECORD requires SHA-256")
            if entry.size is None or entry.size < 0:
                raise ValueError("evaluator payload RECORD requires a byte length")
            digest = base64.b64decode(
                file_hash.value + "=" * (-len(file_hash.value) % 4),
                altchars=b"-_",
                validate=True,
            ).hex()
            if len(digest) != 64:
                raise ValueError("evaluator payload RECORD SHA-256 is malformed")
            located = Path(distribution.locate_file(entry)).resolve(strict=True)
            if not located.is_relative_to(base):
                raise ValueError("evaluator payload RECORD escaped its owned root")
            recorded[logical] = (digest, entry.size)
        return _provider_roots_from_registry(base, recorded=recorded)
    except (OSError, TypeError, ValueError):
        return ()


def _provider_roots_from_registry(
    base: Path,
    *,
    recorded: dict[str, tuple[str, int]] | None = None,
) -> tuple[Path, ...]:
    """Validate one canonical provider-root registry and its complete payload."""
    try:
        if base.is_symlink():
            return ()
        base = base.resolve(strict=True)
        if not base.is_dir():
            return ()
        registry_path = base / _PROVIDER_ROOTS_REGISTRY
        registry_bytes = registry_path.read_bytes()
        value = json.loads(registry_bytes)
        if _canonical_provider_registry(value) != registry_bytes:
            return ()
        roots = _validate_provider_registry(value)
        physical = _physical_provider_payload(base)
        if recorded is not None:
            if set(recorded) != set(physical):
                return ()
            if any(
                path.stat().st_size != recorded[relative][1]
                for relative, path in physical.items()
            ):
                return ()
        actual_digests = {
            relative: _file_sha256(path) for relative, path in physical.items()
        }
        if recorded is not None:
            for relative in physical:
                expected_digest, _ = recorded[relative]
                if actual_digests[relative] != expected_digest:
                    return ()
            registry_digest, _ = recorded[_PROVIDER_ROOTS_REGISTRY]
            if hashlib.sha256(registry_bytes).hexdigest() != registry_digest:
                return ()
        selected = []
        for root in roots:
            prefix = f"{root['path']}/"
            members = {
                relative.removeprefix(prefix): actual_digests[relative]
                for relative in physical
                if relative.startswith(prefix)
            }
            if len(members) != root["file_count"]:
                return ()
            if _provider_tree_sha256(members) != root["tree_sha256"]:
                return ()
            selected.append((base / root["path"]).resolve(strict=True))
        return _normalize_provider_roots(selected)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return ()


def _canonical_provider_registry(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _validate_provider_registry(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "platform",
        "roots",
        "schema_version",
    }:
        raise ValueError("invalid evaluator root registry object")
    if value["schema_version"] != _PROVIDER_ROOTS_SCHEMA:
        raise ValueError("unsupported evaluator root registry schema")
    if value["platform"] != "linux-x86_64":
        raise ValueError("unsupported evaluator root platform")
    roots = value["roots"]
    if not isinstance(roots, list) or len(roots) != len(_PROVIDER_ROOT_SPECS):
        raise ValueError("incomplete evaluator root registry")
    result: list[dict[str, Any]] = []
    for entry, (expected_id, expected_kind, expected_path) in zip(
        roots, _PROVIDER_ROOT_SPECS, strict=True
    ):
        if not isinstance(entry, dict) or set(entry) != {
            "file_count",
            "id",
            "kind",
            "path",
            "tree_sha256",
        }:
            raise ValueError("invalid evaluator root entry")
        if (
            entry["id"] != expected_id
            or entry["kind"] != expected_kind
            or entry["path"] != expected_path
            or not isinstance(entry["file_count"], int)
            or isinstance(entry["file_count"], bool)
            or entry["file_count"] <= 0
            or not _is_sha256(entry["tree_sha256"])
        ):
            raise ValueError("evaluator root entry drifted")
        result.append(entry)
    return result


def _physical_provider_payload(base: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in base.rglob("*"):
        metadata_value = path.lstat()
        if stat.S_ISDIR(metadata_value.st_mode):
            if path.is_symlink():
                raise ValueError("evaluator payload contains a symlink directory")
            continue
        if not stat.S_ISREG(metadata_value.st_mode) or path.is_symlink():
            raise ValueError("evaluator payload contains a special file")
        relative = path.relative_to(base).as_posix()
        if relative in files:
            raise ValueError("duplicate evaluator payload path")
        files[relative] = path
    expected_top_level = {
        _PROVIDER_ROOTS_REGISTRY,
        *(path for _, _, path in _PROVIDER_ROOT_SPECS),
    }
    actual_top_level = {relative.split("/", 1)[0] for relative in files}
    if actual_top_level != expected_top_level:
        raise ValueError("evaluator payload has an incomplete root set")
    return files


def _provider_tree_sha256(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, content_sha256 in sorted(files.items()):
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(content_sha256))
    return f"sha256:{digest.hexdigest()}"


def _file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        metadata_value = os.fstat(descriptor)
        if not stat.S_ISREG(metadata_value.st_mode):
            raise ValueError("evaluator payload file is not regular")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _normalize_provider_roots(provider_roots: Sequence[Path]) -> tuple[Path, ...]:
    """Freeze one explicit deployment-owned evaluator-root selection."""
    authored = tuple(Path(root) for root in provider_roots)
    if any(root.is_symlink() for root in authored):
        raise RuntimeError("runner evaluator provider roots cannot be symlinks")
    normalized = tuple(root.resolve(strict=True) for root in authored)
    if len(set(normalized)) != len(normalized) or not all(
        root.is_dir() for root in normalized
    ):
        raise RuntimeError("runner evaluator provider roots are invalid or duplicated")
    return normalized


def _runner_subprocess_environment(provider_roots: tuple[Path, ...]) -> dict[str, str]:
    """Bind child discovery to deployment-owned roots, overriding ambient input."""
    environment = os.environ.copy()
    if provider_roots:
        normalized = _normalize_provider_roots(provider_roots)
        environment[_PROVIDER_ROOTS_ENV] = os.pathsep.join(map(os.fspath, normalized))
    else:
        environment.pop(_PROVIDER_ROOTS_ENV, None)
    return environment


def _load_capabilities(
    binary: Path, provider_roots: tuple[Path, ...] = ()
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(binary), "--capabilities"],
            capture_output=True,
            check=False,
            env=_runner_subprocess_environment(provider_roots),
            timeout=_CAPABILITIES_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"aiperf-runner capability negotiation timed out after "
            f"{_CAPABILITIES_TIMEOUT_SECONDS:g}s"
        ) from error
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
    _validate_v2_capabilities(capabilities)
    return capabilities


def _is_catalog_shape(capabilities: dict[str, Any]) -> bool:
    return (
        isinstance(capabilities.get("schema_version"), str)
        and bool(capabilities["schema_version"])
        and isinstance(capabilities.get("endpoint"), dict)
        and isinstance(capabilities.get("transport"), dict)
    )


def _validate_v2_capabilities(capabilities: dict[str, Any]) -> None:
    """Validate the plugins.yaml-shaped linked runner catalog."""
    if not _is_catalog_shape(capabilities):
        raise ValueError(
            "aiperf-runner catalog must include non-empty schema_version plus "
            "endpoint and transport category maps"
        )
    for category in _REQUIRED_CATALOG_CATEGORIES:
        _require_category_map(capabilities, category, required=True)
    for category in _OPTIONAL_CATALOG_CATEGORIES:
        if category in capabilities:
            _require_category_map(capabilities, category, required=False)


def _require_category_map(
    capabilities: dict[str, Any], category: str, *, required: bool
) -> None:
    value = capabilities.get(category)
    if not isinstance(value, dict):
        raise ValueError(f"aiperf-runner catalog {category} must be an object")
    if required and not value:
        raise ValueError(
            f"aiperf-runner catalog {category} must contain at least one type id"
        )
    for type_id, entry in value.items():
        if not isinstance(type_id, str) or not type_id:
            raise ValueError(
                f"aiperf-runner catalog {category} keys must be non-empty strings"
            )
        if not isinstance(entry, dict):
            raise ValueError(
                f"aiperf-runner catalog {category}.{type_id} must be an object"
            )


def _parse_validation_response(
    stdout: bytes,
    *,
    benchmark_id: str,
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
    """Hash one opened runner image with the native versioned BLAKE3 contract.

    Retained for release/packaging diagnostics; not used as a wire pin.
    """
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
    """Fail before launch unless this image catalogs the requested Config ids."""
    run = request.get("run")
    if not isinstance(run, dict):
        raise ValueError("protocol-v2 request omitted its run object")
    cfg = run.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("protocol-v2 request omitted run.cfg")

    transport = cfg.get("transport")
    if not isinstance(transport, dict) or not isinstance(transport.get("type"), str):
        raise ValueError("protocol-v2 request omitted run.cfg.transport.type")
    _require_catalog_id(capabilities, "transport", transport["type"])

    endpoint = cfg.get("endpoint")
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get("type"), str):
        raise ValueError("protocol-v2 request omitted run.cfg.endpoint.type")
    _require_catalog_id(capabilities, "endpoint", endpoint["type"])

    profiles = cfg.get("endpoint_profiles")
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, dict):
        raise ValueError("protocol-v2 run.cfg.endpoint_profiles must be an object")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, dict) or not isinstance(profile.get("type"), str):
            raise ValueError(
                f"protocol-v2 endpoint profile {profile_id!r} omitted type"
            )
        _require_catalog_id(capabilities, "endpoint", profile["type"])

    datasets = cfg.get("datasets")
    if datasets is None:
        return
    if not isinstance(datasets, list):
        raise ValueError("protocol-v2 run.cfg.datasets must be an array")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"protocol-v2 dataset {index} must be an object")
        _require_dataset_catalog(capabilities, dataset, index)


def _require_dataset_catalog(
    capabilities: dict[str, Any], dataset: dict[str, Any], index: int
) -> None:
    dataset_type = dataset.get("type")
    if dataset_type == "file":
        format_id = dataset.get("format")
        if isinstance(format_id, str) and format_id:
            _require_optional_catalog_id(
                capabilities, "custom_dataset_loader", format_id
            )
        return
    if dataset_type == "public":
        public_id = dataset.get("dataset")
        if isinstance(public_id, str) and public_id:
            _require_optional_catalog_id(
                capabilities, "public_dataset_loader", public_id
            )
        return
    if dataset_type == "synthetic":
        _require_optional_catalog_id(capabilities, "synthetic", "synthetic")
        return
    if not isinstance(dataset_type, str) or not dataset_type:
        raise ValueError(f"protocol-v2 dataset {index} omitted type")


def _require_optional_catalog_id(
    capabilities: dict[str, Any], category: str, required: str
) -> None:
    advertised = capabilities.get(category)
    if not isinstance(advertised, dict) or not advertised:
        return
    if required not in advertised:
        raise RuntimeError(
            f"aiperf-runner does not support {category}.{required}; "
            f"advertised {sorted(advertised)!r}"
        )


def _require_catalog_id(
    capabilities: dict[str, Any], category: str, required: str
) -> None:
    advertised = capabilities.get(category)
    if not isinstance(advertised, dict) or required not in advertised:
        raise RuntimeError(
            f"aiperf-runner does not support {category}.{required}; "
            f"advertised {sorted(advertised) if isinstance(advertised, dict) else advertised!r}"
        )
