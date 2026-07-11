# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical local executor backed by one fresh Rust process per run."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import orjson

from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.native_report import load_native_summary
from aiperf.orchestrator.rust_wire import (
    RUNNER_PROTOCOL_VERSION,
    build_run_request,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun


logger = logging.getLogger(__name__)

__all__ = ["RustSubprocessExecutor"]

_RUNNER_ENV = "AIPERF_RUNNER_BIN"


class RustSubprocessExecutor(RunExecutor):
    """Execute each fully planned run through ``aiperf-runner`` over stdio."""

    def __init__(self, base_dir: Path, *, binary: Path | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.binary = _resolve_runner_binary(binary)

    def derive_id(self, plan: BenchmarkPlan, var_idx: int, trial: int) -> str:
        return uuid4().hex

    async def execute(self, run: BenchmarkRun) -> RunResult:
        """Resolve Config v2 and run the blocking child outside the event loop."""
        return await asyncio.to_thread(self.execute_sync, run)

    def execute_sync(self, run: BenchmarkRun) -> RunResult:
        """Execute one run and return its orchestrator-facing metric projection."""
        try:
            self._resolve_run(run)
            request = build_run_request(run)
            completed = subprocess.run(
                [str(self.binary)],
                input=orjson.dumps(request),
                capture_output=True,
                check=False,
            )
            terminal = _parse_terminal(completed.stdout, run)
            if completed.returncode != 0 or not terminal["success"]:
                return _failure(completed, terminal, run)
            report_path = _validated_report_path(terminal, run.artifact_dir)
            summary = load_native_summary(report_path)
            return _classify(summary, run)
        except Exception as error:
            logger.exception("Error executing native run %s", run.label)
            return RunResult(
                label=run.label or f"run_{run.trial:04d}",
                success=False,
                error=str(error),
                artifacts_path=run.artifact_dir,
            )

    @staticmethod
    def _resolve_run(run: BenchmarkRun) -> None:
        # Config resolution remains Python-owned. It resolves artifact paths,
        # tokenizer aliases/cache, dataset paths/format metadata, and timing
        # validation before the explicit Rust projection is built.
        from aiperf.config.resolution.resolvers import build_default_resolver_chain

        build_default_resolver_chain().resolve_all(run)


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
        "aiperf-runner executable was not found; install it beside aiperf or "
        f"set {_RUNNER_ENV} to its absolute path"
    )


def _parse_terminal(stdout: bytes, run: BenchmarkRun) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            "native runner must write exactly one terminal JSON line to stdout; "
            f"received {len(lines)} non-empty lines"
        )
    try:
        terminal = orjson.loads(lines[0])
    except orjson.JSONDecodeError as error:
        raise ValueError(
            f"native runner returned invalid terminal JSON: {error}"
        ) from error
    if not isinstance(terminal, dict):
        raise ValueError("native runner terminal response must be an object")
    expected = {
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "event": "run_terminal",
        "benchmark_id": run.benchmark_id,
    }
    for field, value in expected.items():
        if terminal.get(field) != value:
            raise ValueError(
                f"native runner terminal {field}={terminal.get(field)!r}; expected {value!r}"
            )
    if not isinstance(terminal.get("success"), bool):
        raise ValueError("native runner terminal success must be a boolean")
    return terminal


def _validated_report_path(terminal: dict[str, Any], artifact_dir: Path) -> Path:
    authored = terminal.get("report_path")
    if not isinstance(authored, str) or not authored:
        raise ValueError("successful native terminal response omitted report_path")
    report = Path(authored).resolve()
    root = artifact_dir.resolve()
    if report.parent != root or report.name != "native-v2.json":
        raise ValueError(
            f"native runner returned report outside its run contract: {report}"
        )
    return report


def _failure(
    completed: subprocess.CompletedProcess[bytes],
    terminal: dict[str, Any],
    run: BenchmarkRun,
) -> RunResult:
    detail = terminal.get("error") or "native runner failed without an error message"
    stderr = completed.stderr.decode(errors="replace").strip()
    if stderr:
        detail = f"{detail}\nRust stderr: {stderr[-4000:]}"
    return RunResult(
        label=run.label,
        success=False,
        error=f"native benchmark failed (exit {completed.returncode}): {detail}",
        artifacts_path=run.artifact_dir,
    )


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
