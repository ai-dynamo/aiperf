# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate and drive the unified ``aiperf`` execution binary.

There is no separate ``aiperf-runner`` executable anymore: the one native
``aiperf`` binary is BOTH the front door and the execution engine. When the
Python orchestrator path is active (``AIPERF_NATIVE=0``) it drives one run by
spawning that same binary in its internal ``--execute`` mode over stdio.

There is no capability negotiation and no preflight: the binary's own
``validate``/``execute`` operation is the single source of truth and fails
closed on an unsupported id. The catalog (`--capabilities`) subprocess mode was
removed — capabilities is an in-process function inside the native binary.

The binary is **interned** in the one maturin-built ``aiperf`` wheel as package
data at ``aiperf/_bin/aiperf-native``. Discovery precedence is
``explicit --exec-bin -> AIPERF_EXEC_BIN -> interned package data -> PATH``.
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
from importlib import resources
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
    from aiperf.config.resolution.plan import BenchmarkRun

# The internal re-exec flag: `aiperf --execute` reads one protocol-v2 request
# (validate or execute) from stdin and runs it. The same flag serves both
# operations; the envelope's `operation` field selects behaviour.
_EXECUTE_FLAG = "--execute"
_EXEC_ENV = "AIPERF_EXEC_BIN"
_EXEC_COMMAND = "aiperf-native"
# Package data location of the interned binary inside the installed `aiperf`
# wheel (maturin `include` glob in pyproject.toml + Makefile `bundle-cli`).
_INTERNED_PACKAGE = "aiperf"
_INTERNED_SUBDIR = "_bin"
_DISTRIBUTION_ID_DOMAIN = b"aiperf-runner-distribution-v1\0"
_DISTRIBUTION_ID_PREFIX = "blake3:"
_DISTRIBUTION_ID_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class RunnerInstallation:
    """One selected ``aiperf`` execution binary."""

    binary: Path

    @classmethod
    def resolve(cls, binary: Path | None = None) -> RunnerInstallation:
        """Discover the execution binary (no capability negotiation)."""
        return cls(binary=_resolve_exec_binary(binary))

    def project_authored_request(
        self,
        run: BenchmarkRun,
        *,
        operation: RunnerOperationV2,
    ) -> dict[str, Any]:
        """Build a v2 BenchmarkRun envelope without executing it."""
        return build_authored_run_request(run, operation=operation)

    def execute(
        self,
        request: dict[str, Any],
        *,
        on_stderr_line: Callable[[bytes], None] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one request in a fresh ``aiperf --execute`` child.

        When invoked on the main thread this forwards the first SIGINT/SIGTERM
        to the child, so a Ctrl+C during a benchmark triggers graceful phase
        cancellation (partial results written, native summary
        ``was_cancelled=true``) instead of tearing the child down before it can
        write results. Off the main thread (e.g. the sweep orchestrator's
        ``asyncio.to_thread`` worker, where ``signal.signal`` raises) the child
        runs without local forwarding.

        ``on_stderr_line`` is invoked for each captured stderr line as it
        arrives (bytes, newline stripped). The run path uses this to surface the
        child's live lifecycle/readiness trace — including the profiling banner a
        Ctrl+C harness waits for — instead of only forwarding it after the child
        exits. The full stderr is still captured and returned.
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
        graceful shutdown. The binary is invoked in its internal ``--execute``
        mode; the envelope's ``operation`` selects validate vs execute.
        """
        return subprocess.Popen(
            [str(self.binary), _EXECUTE_FLAG],
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
            f"aiperf rejected authored run {benchmark_id!r} "
            f"(exit {completed.returncode}): {detail}"
        )


def _communicate_forwarding_signals(
    child: subprocess.Popen[bytes],
    payload: bytes,
    on_stderr_line: Callable[[bytes], None] | None = None,
) -> tuple[bytes, bytes]:
    """Drive one execution child to completion, forwarding the first Ctrl+C/term.

    On the main thread this installs temporary SIGINT/SIGTERM handlers that send
    SIGINT to the child exactly once and then keep waiting for its graceful exit
    and report. The handler swallows the signal (never raises
    ``KeyboardInterrupt``) so the parent does not tear the child down before it
    writes its partial results; the child cancels the active phase, drains
    in-flight requests, and exits 0 after writing ``native-v2.json`` with
    ``was_cancelled=true``. Prior handlers are always restored.

    Signal installation only works on the main thread, so off-main-thread
    callers (the sweep orchestrator's ``asyncio.to_thread`` worker) run without
    local forwarding.

    stdout and stderr are drained on dedicated reader threads so the main thread
    stays free to run the Python signal handler while the child is still alive.
    When ``on_stderr_line`` is provided each captured stderr line is delivered to
    it live (bytes, newline stripped), which is how the child's profiling banner
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

    # Send the request and close stdin so the child can begin. Writing before
    # the reader threads consume output is safe: the payload is small and the
    # child reads its whole request before emitting anything.
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


def _resolve_exec_binary(explicit: Path | None) -> Path:
    """Resolve the execution binary: explicit → env → interned → PATH."""
    if explicit is not None:
        return _require_exec_binary(Path(explicit), "explicit --exec-bin")

    configured = os.environ.get(_EXEC_ENV)
    if configured:
        return _require_exec_binary(Path(configured), _EXEC_ENV)

    interned = _interned_binary()
    if interned is not None:
        return interned

    discovered = shutil.which(_EXEC_COMMAND) or shutil.which("aiperf")
    if discovered:
        return _require_exec_binary(Path(discovered), "PATH")

    raise FileNotFoundError(
        f"aiperf execution binary was not found; the installed {_INTERNED_PACKAGE!r} "
        f"package did not intern {_INTERNED_SUBDIR}/{_EXEC_COMMAND} (build the wheel "
        "with `make wheel`/`make bundle-cli`), or pass --exec-bin, set "
        f"{_EXEC_ENV}, or place {_EXEC_COMMAND} on PATH for development"
    )


def _interned_binary() -> Path | None:
    """Locate the binary interned as package data in the installed wheel.

    Resolves ``aiperf/_bin/aiperf-native`` through ``importlib.resources`` — the
    absolute install path, independent of PATH. The ``aiperf`` distribution is
    always installed unpacked (it carries a compiled extension module and this
    executable), so ``files()`` yields a concrete filesystem path. Returns
    ``None`` when the package data is absent (e.g. a source checkout that has not
    run ``make bundle-cli``), so lower precedence tiers can still resolve.
    """
    try:
        base = resources.files(_INTERNED_PACKAGE)
    except (ModuleNotFoundError, TypeError):
        return None
    candidate = base.joinpath(_INTERNED_SUBDIR, _EXEC_COMMAND)
    try:
        path = Path(str(candidate))
    except (TypeError, ValueError):
        return None
    if path.is_file() and os.access(path, os.X_OK):
        return path.resolve()
    return None


def _require_exec_binary(candidate: Path, source: str) -> Path:
    """Validate one selected precedence tier without falling through."""
    resolved = candidate.expanduser().resolve()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return resolved
    raise FileNotFoundError(
        f"{source} selected aiperf execution binary {resolved}, but it is not an "
        "executable file; refusing to substitute a lower-precedence binary"
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
            "aiperf validate must write exactly one JSON line; "
            f"received {len(lines)}; child exit code {returncode}{detail}"
        )
    try:
        response = orjson.loads(lines[0])
    except orjson.JSONDecodeError as error:
        raise ValueError(
            f"aiperf returned invalid validation JSON: {error}"
        ) from error
    if not isinstance(response, dict):
        raise ValueError("aiperf validation response must be an object")
    expected: dict[str, object] = {
        "protocol_version": RUNNER_PROTOCOL_V2,
        "event": "run_validation",
        "benchmark_id": benchmark_id,
    }
    for field, value in expected.items():
        if response.get(field) != value:
            raise ValueError(
                f"aiperf validation {field}={response.get(field)!r}; "
                f"expected {value!r}"
            )
    success = response.get("success")
    if not isinstance(success, bool):
        raise ValueError("aiperf validation success must be a boolean")
    if response.get("completeness") not in {"static", "complete"}:
        raise ValueError(
            "aiperf validation completeness must be 'static' or 'complete'"
        )
    for field in ("deferred_checks", "errors"):
        entries = response.get(field, [])
        if not isinstance(entries, list) or not all(
            isinstance(entry, dict) for entry in entries
        ):
            raise ValueError(
                f"aiperf validation {field} must be an array of objects"
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
            "aiperf validation errors require non-empty code and message strings"
        )
    if success != (returncode == 0):
        raise ValueError(
            "aiperf validation success disagrees with child exit code "
            f"{returncode}"
        )
    if success and errors:
        raise ValueError("successful aiperf validation cannot contain errors")
    if not success and not errors:
        raise ValueError("failed aiperf validation must contain errors")
    return response


def _runner_distribution_id(binary: Path) -> str:
    """Hash one opened execution image with the native versioned BLAKE3 contract.

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
            f"cannot read selected aiperf image {binary} for distribution "
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
