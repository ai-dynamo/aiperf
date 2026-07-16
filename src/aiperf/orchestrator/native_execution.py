# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the native ``aiperf`` binary for one benchmark run from Python.

The one native ``aiperf`` binary is BOTH the front door and the execution engine.
When a Python frontend entry point needs to execute a run it does so here: resolve
the binary, project the authored Config-v2 run to a protocol-v2 ``execute``
envelope, drive one ``aiperf --execute`` child over stdio, and parse its terminal
line + ``native-v2.json`` into a :class:`RunResult`. Kubernetes cell pods launch a
sliced ``aiperf --cell`` child (:func:`run_cell_process`).

This consolidates and replaces the former ``runner_installation`` +
``rust_executor`` modules. It is execute-only: there is no ``validate`` operation,
no capability negotiation, no interned-binary discovery (the wheel installs the
binary onto PATH as the ``aiperf`` command), and no distribution-id hashing. The
binary's own ``execute`` operation is the single source of truth and fails closed
on an unsupported id.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import orjson

from aiperf.common.redact import redact_string
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.native_report import (
    export_python_compatibility_reports,
    load_native_report,
    project_native_summary,
)
from aiperf.orchestrator.rust_wire import build_authored_run_request

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun

logger = logging.getLogger(__name__)

__all__ = ["NativeExecutor", "resolve_native_binary", "run_cell_process"]

# The internal re-exec flags. `aiperf --execute` reads one protocol-v2 execute
# envelope from stdin and runs it; `aiperf --cell` fetches its sliced envelope
# from the cellular controller over the wire (no stdin, no config file).
_EXECUTE_FLAG = "--execute"
_CELL_FLAG = "--cell"
_EXEC_ENV = "AIPERF_EXEC_BIN"
_EXEC_COMMAND = "aiperf"
# The binary writes exactly this authoritative report into ``run.artifact_dir``.
_NATIVE_REPORT_NAME = "native-v2.json"


def resolve_native_binary(explicit: Path | None = None) -> Path:
    """Resolve the ``aiperf`` execution binary.

    Precedence: explicit path -> ``AIPERF_EXEC_BIN`` env -> ``aiperf`` on PATH ->
    the binary installed alongside the current interpreter (``sys.executable``'s
    bin dir, where the wheel's ``.data/scripts/aiperf`` lands). Each selected tier
    is required to be an executable file; there is no silent fall-through to a
    lower-precedence binary.
    """
    if explicit is not None:
        return _require_binary(Path(explicit), "explicit --exec-bin")

    configured = os.environ.get(_EXEC_ENV)
    if configured:
        return _require_binary(Path(configured), _EXEC_ENV)

    discovered = shutil.which(_EXEC_COMMAND)
    if discovered:
        return _require_binary(Path(discovered), "PATH")

    alongside = Path(sys.executable).resolve().parent / _EXEC_COMMAND
    if alongside.is_file() and os.access(alongside, os.X_OK):
        return alongside

    raise FileNotFoundError(
        f"aiperf execution binary was not found; install the aiperf wheel (which "
        f"puts {_EXEC_COMMAND} on PATH), pass --exec-bin, or set {_EXEC_ENV}"
    )


def _require_binary(candidate: Path, source: str) -> Path:
    """Validate one selected precedence tier without falling through."""
    resolved = candidate.expanduser().resolve()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return resolved
    raise FileNotFoundError(
        f"{source} selected aiperf execution binary {resolved}, but it is not an "
        "executable file; refusing to substitute a lower-precedence binary"
    )


def run_cell_process(binary: Path | None = None) -> int:
    """Launch ``aiperf --cell`` and wait, returning its exit code.

    A Kubernetes cell reads neither config nor stdin: it fetches its ``(cell_id,
    cell_count)`` slice + sliced execute envelope from the controller over the
    wire using the operator-set ``AIPERF_CELL_*`` env, which flows straight
    through this inherited-env subprocess.
    """
    exec_binary = binary or resolve_native_binary()
    completed = subprocess.run([str(exec_binary), _CELL_FLAG], check=False)  # noqa: S603
    return completed.returncode


class NativeExecutor(RunExecutor):
    """Execute each fully planned run through one fresh ``aiperf --execute`` child."""

    def __init__(self, base_dir: Path, *, binary: Path | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.binary = resolve_native_binary(binary)

    def derive_id(self, plan: BenchmarkPlan, var_idx: int, trial: int) -> str:
        return uuid4().hex

    async def execute(self, run: BenchmarkRun) -> RunResult:
        """Run the blocking child outside the event loop."""
        return await asyncio.to_thread(self.execute_sync, run)

    def execute_sync(self, run: BenchmarkRun) -> RunResult:
        """Execute one run and return its orchestrator-facing metric projection."""
        try:
            request = self._request_for_run(run)
            _clear_prior_report(run.artifact_dir)
            # Surface the child's control-plane stderr (endpoint readiness probe
            # progress, the profiling banner) in the run log live, line by line:
            # stdout is reserved for the single terminal JSON line. Forwarding live
            # - rather than after the child exits - also lets a signal-forwarding
            # parent observe the profiling banner while the run is still in flight.
            completed = self._run_execute_child(
                request, on_stderr_line=_forward_child_stderr_line
            )
            terminal = _parse_terminal(
                completed.stdout,
                run,
                protocol_version=request["protocol_version"],
                returncode=completed.returncode,
                stderr=completed.stderr,
            )
            if completed.returncode != 0 or not terminal["success"]:
                return _failure(completed, terminal, run)
            report_path = _validated_report_path(terminal, run.artifact_dir)
            native_report = load_native_report(report_path)
            summary = project_native_summary(native_report)
            export_python_compatibility_reports(native_report, summary, run)
            return _classify(summary, run)
        except Exception as error:
            logger.exception("Error executing native run %s", run.label)
            return RunResult(
                label=run.label or f"run_{run.trial:04d}",
                success=False,
                error=redact_string(str(error)),
                artifacts_path=run.artifact_dir,
            )

    def _request_for_run(self, run: BenchmarkRun) -> dict[str, Any]:
        """Project one authored v2 run into an ``execute`` envelope.

        Two pre-projection resolvers run first (the only ``run.resolved`` /
        ``run.cfg`` mutations Python still owns before the strict, side-effect-free
        projection):

        * ``GpuMetricsResolver`` validates the ``--gpu-telemetry`` custom-metrics
          CSV into ``run.resolved.gpu_custom_metrics`` — the binary has no CSV
          metric loader, yet its accumulator must register each custom field's
          name+unit to summarize the values the Python telemetry worker scrapes.
        * ``apply_scenario`` applies the ``--scenario`` invariant lock (force
          streaming, inject ignore_eos, auto-fill the trajectory-start t* window
          and per-trace idle-gap cap) into ``run.cfg`` before projection; no-op
          unless ``run.cfg.scenario`` is set. Raises ``ScenarioLockError`` on an
          unresolved conflict unless ``--unsafe-override`` downgrades it.
        """
        from aiperf.common.scenario import apply_scenario
        from aiperf.config.resolution.resolvers import GpuMetricsResolver

        GpuMetricsResolver().resolve(run)
        apply_scenario(run)
        return build_authored_run_request(run, operation="execute")

    def _run_execute_child(
        self,
        request: dict[str, Any],
        *,
        on_stderr_line: Callable[[bytes], None] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Drive one ``aiperf --execute`` child to completion over stdio.

        On the main thread the first SIGINT/SIGTERM is forwarded to the child as a
        single SIGINT, so a Ctrl+C during a benchmark triggers graceful phase
        cancellation (partial results written, ``was_cancelled=true``) instead of
        tearing the child down. Off the main thread (the sweep orchestrator's
        ``asyncio.to_thread`` worker, where ``signal.signal`` raises) the child
        runs without local forwarding.
        """
        child = subprocess.Popen(
            [str(self.binary), _EXECUTE_FLAG],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = _communicate_forwarding_signals(
            child, orjson.dumps(request), on_stderr_line
        )
        return subprocess.CompletedProcess(
            child.args, child.returncode, stdout=stdout, stderr=stderr
        )


def _communicate_forwarding_signals(
    child: subprocess.Popen[bytes],
    payload: bytes,
    on_stderr_line: Callable[[bytes], None] | None = None,
) -> tuple[bytes, bytes]:
    """Drive one execution child to completion, forwarding the first Ctrl+C/term.

    On the main thread this installs temporary SIGINT/SIGTERM handlers that send
    SIGINT to the child exactly once and then keep waiting for its graceful exit;
    the handler swallows the signal (never raises ``KeyboardInterrupt``) so the
    parent does not tear the child down before it writes its partial results.
    Prior handlers are always restored. Signal installation only works on the main
    thread, so off-main-thread callers run without local forwarding.

    stdout and stderr are drained on dedicated reader threads so the main thread
    stays free to run the Python signal handler while the child is still alive.
    When ``on_stderr_line`` is provided each captured stderr line is delivered to
    it live (bytes, newline stripped); the full stderr is still captured/returned.
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

    # Send the request and close stdin so the child can begin. The payload is
    # small and the child reads its whole request before emitting anything.
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


def _forward_child_stderr_line(raw: bytes) -> None:
    """Re-emit one ``aiperf --execute`` child stderr line through the logger, live."""
    text = redact_string(raw.decode(errors="replace")).strip()
    if text:
        logger.info("aiperf-exec: %s", text)


def _clear_prior_report(artifact_dir: Path) -> None:
    """Remove a prior run's authoritative report before launching a fresh child.

    The binary is write-once by design: it refuses to overwrite ``native-v2.json``
    so a failed execution can never replace a good report mid-run. Re-running into
    the same artifact dir is a legitimate user action, and the orchestrator - not
    the binary - owns artifact-dir lifecycle, so the prior report is cleared here,
    immediately before launching the fresh child.
    """
    (artifact_dir / _NATIVE_REPORT_NAME).unlink(missing_ok=True)


def _parse_terminal(
    stdout: bytes,
    run: BenchmarkRun,
    *,
    protocol_version: int,
    returncode: int | None = None,
    stderr: bytes = b"",
) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        diagnostic = redact_string(stderr.decode(errors="replace")).strip()
        process = "" if returncode is None else f"; child exit code {returncode}"
        detail = "" if not diagnostic else f"; stderr: {diagnostic}"
        raise ValueError(
            "native binary must write exactly one terminal JSON line to stdout; "
            f"received {len(lines)} non-empty lines{process}{detail}"
        )
    try:
        terminal = orjson.loads(lines[0])
    except orjson.JSONDecodeError as error:
        raise ValueError(
            f"native binary returned invalid terminal JSON: {error}"
        ) from error
    if not isinstance(terminal, dict):
        raise ValueError("native binary terminal response must be an object")
    expected = {
        "protocol_version": protocol_version,
        "event": "run_terminal",
        "benchmark_id": run.benchmark_id,
    }
    for field, value in expected.items():
        if terminal.get(field) != value:
            raise ValueError(
                f"native binary terminal {field}={terminal.get(field)!r}; "
                f"expected {value!r}"
            )
    if not isinstance(terminal.get("success"), bool):
        raise ValueError("native binary terminal success must be a boolean")
    return terminal


def _validated_report_path(terminal: dict[str, Any], artifact_dir: Path) -> Path:
    authored = terminal.get("report_path")
    if not isinstance(authored, str) or not authored:
        raise ValueError("successful native terminal response omitted report_path")
    report = Path(authored).resolve()
    root = artifact_dir.resolve()
    if report.parent != root or report.name != _NATIVE_REPORT_NAME:
        raise ValueError(
            f"native binary returned report outside its run contract: {report}"
        )
    return report


def _failure(
    completed: subprocess.CompletedProcess[bytes],
    terminal: dict[str, Any],
    run: BenchmarkRun,
) -> RunResult:
    detail = redact_string(_terminal_error(terminal))
    stderr = redact_string(completed.stderr.decode(errors="replace")).strip()
    if stderr:
        detail = f"{detail}\nRust stderr: {stderr[-4000:]}"
    return RunResult(
        label=run.label,
        success=False,
        error=f"native benchmark failed (exit {completed.returncode}): {detail}",
        artifacts_path=run.artifact_dir,
    )


def _terminal_error(terminal: dict[str, Any]) -> str:
    """Render the typed failure without using stderr."""
    error = terminal.get("error")
    if isinstance(error, str) and error:
        return error
    errors = terminal.get("errors")
    if isinstance(errors, list):
        messages = [
            diagnostic.get("message")
            for diagnostic in errors
            if isinstance(diagnostic, dict)
            and isinstance(diagnostic.get("message"), str)
            and diagnostic["message"]
        ]
        if messages:
            stage = terminal.get("stage")
            prefix = f"{stage}: " if isinstance(stage, str) and stage else ""
            return prefix + "; ".join(messages)
    return "native binary failed without an error message"


def _classify(summary: dict[str, Any], run: BenchmarkRun) -> RunResult:
    request_count = summary.get("request_count")
    error_count = summary.get("error_request_count")
    if request_count is None or request_count.avg in (None, 0):
        if error_count is not None and error_count.avg not in (None, 0):
            error = f"All {int(error_count.avg)} requests failed"
        else:
            error = "No requests completed"
        return RunResult(
            label=run.label,
            success=False,
            error=error,
            artifacts_path=run.artifact_dir,
        )
    return RunResult(
        label=run.label,
        success=True,
        summary_metrics=summary,
        artifacts_path=run.artifact_dir,
    )
