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

import contextlib
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Callable
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

    @classmethod
    def resolve(cls, binary: Path | None = None) -> RunnerInstallation:
        """Discover one runner and negotiate its catalog once."""
        resolved = _resolve_runner_binary(binary)
        return cls(binary=resolved, capabilities=_load_capabilities(resolved))

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

    def execute(
        self,
        request: dict[str, Any],
        *,
        on_stderr_line: Callable[[bytes], None] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one request with the same binary whose catalog was negotiated.

        When invoked on the main thread this forwards the first SIGINT/SIGTERM
        to the child, so a Ctrl+C during a benchmark triggers the runner's
        graceful phase cancellation (partial results written, native summary
        ``was_cancelled=true``) instead of tearing the child down before it can
        write results. Off the main thread (e.g. the sweep orchestrator's
        ``asyncio.to_thread`` worker, where ``signal.signal`` raises) the child
        runs without local forwarding.

        ``on_stderr_line`` is invoked for each captured stderr line as it
        arrives (bytes, newline stripped). The run path uses this to surface the
        runner's live lifecycle/readiness trace - including the profiling banner
        a Ctrl+C harness waits for - instead of only forwarding it after the
        child exits. The full stderr is still captured and returned.
        """
        child = self.spawn(request)
        stdout, stderr = _communicate_forwarding_signals(
            child, orjson.dumps(request), on_stderr_line
        )
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


def _communicate_forwarding_signals(
    child: subprocess.Popen[bytes],
    payload: bytes,
    on_stderr_line: Callable[[bytes], None] | None = None,
) -> tuple[bytes, bytes]:
    """Drive one runner child to completion, forwarding the first Ctrl+C/term.

    On the main thread this installs temporary SIGINT/SIGTERM handlers that send
    SIGINT to the child exactly once and then keep waiting for its graceful exit
    and report. The handler swallows the signal (never raises
    ``KeyboardInterrupt``) so the parent does not tear the child down before it
    writes its partial results; the runner cancels the active phase, drains
    in-flight requests, and exits 0 after writing ``native-v2.json`` with
    ``was_cancelled=true``. Prior handlers are always restored.

    Signal installation only works on the main thread, so off-main-thread
    callers (the sweep orchestrator's ``asyncio.to_thread`` worker) run without
    local forwarding.

    stdout and stderr are drained on dedicated reader threads so the main thread
    stays free to run the Python signal handler while the child is still alive.
    When ``on_stderr_line`` is provided each captured stderr line is delivered to
    it live (bytes, newline stripped), which is how the runner's profiling banner
    reaches a signal-forwarding lifecycle owner before the run finishes. The full
    stderr is still captured and returned.
    """
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def _drain_stdout() -> None:
        if child.stdout is not None:
            stdout_chunks.append(child.stdout.read())

    def _drain_stderr() -> None:
        if child.stderr is None:
            return
        for raw in iter(child.stderr.readline, b""):
            stderr_chunks.append(raw)
            if on_stderr_line is not None:
                line = raw.rstrip(b"\r\n")
                if line:
                    with contextlib.suppress(Exception):
                        on_stderr_line(line)

    # Send the request and close stdin so the runner can begin. Writing before
    # the reader threads consume output is safe: the payload is small and the
    # runner reads its whole request before emitting anything.
    with contextlib.suppress(BrokenPipeError, OSError):
        if child.stdin is not None:
            child.stdin.write(payload)
            child.stdin.close()

    out_thread = threading.Thread(target=_drain_stdout, daemon=True)
    err_thread = threading.Thread(target=_drain_stderr, daemon=True)
    out_thread.start()
    err_thread.start()

    forwarded = False

    def _forward(_signum: int, _frame: object) -> None:
        nonlocal forwarded
        if forwarded:
            return
        forwarded = True
        # The child may already have exited or be otherwise unreachable.
        with contextlib.suppress(ProcessLookupError, ValueError, OSError):
            child.send_signal(signal.SIGINT)

    on_main = threading.current_thread() is threading.main_thread()
    installed: list[tuple[int, Any]] = []
    try:
        if on_main:
            for signum in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
                if signum is None:
                    continue
                try:
                    previous = signal.signal(signum, _forward)
                except (ValueError, OSError):
                    # Not the main thread, or not installable here.
                    continue
                installed.append((signum, previous))
        child.wait()
        out_thread.join()
        err_thread.join()
    finally:
        for signum, previous in installed:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signum, previous)

    return b"".join(stdout_chunks), b"".join(stderr_chunks)


def _resolve_runner_binary(explicit: Path | None) -> Path:
    """Resolve one runner by precedence: explicit → env → companion → PATH."""
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
            if str(entry).replace("\\", "/").rsplit("/", 1)[-1] in filenames
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


def _load_capabilities(binary: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(binary), "--capabilities"],
            capture_output=True,
            check=False,
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
