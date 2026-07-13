# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical local executor backed by one fresh Rust process per run."""

from __future__ import annotations

import asyncio
import logging
import subprocess
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
from aiperf.orchestrator.runner_installation import RunnerInstallation

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun


logger = logging.getLogger(__name__)

__all__ = ["RunnerInstallation", "RustSubprocessExecutor"]


class RustSubprocessExecutor(RunExecutor):
    """Execute each fully planned run through ``aiperf-runner`` over stdio."""

    def __init__(
        self,
        base_dir: Path,
        *,
        installation: RunnerInstallation | None = None,
        binary: Path | None = None,
    ) -> None:
        if installation is not None and binary is not None:
            raise ValueError("pass either installation or binary, not both")
        self.base_dir = Path(base_dir)
        self.installation = installation or RunnerInstallation.resolve(binary)
        # Retain these read-only aliases for callers that inspect executor
        # identity while the explicit RunnerInstallation seam rolls out.
        self.binary = self.installation.binary
        self.capabilities = self.installation.capabilities

    def derive_id(self, plan: BenchmarkPlan, var_idx: int, trial: int) -> str:
        return uuid4().hex

    def preflight_plan(self, plan: BenchmarkPlan) -> None:
        """Reject fixed-plan endpoint IDs unavailable in this installation."""
        self.installation.preflight_plan(plan)

    async def execute(self, run: BenchmarkRun) -> RunResult:
        """Resolve Config v2 and run the blocking child outside the event loop."""
        return await asyncio.to_thread(self.execute_sync, run)

    def execute_sync(self, run: BenchmarkRun) -> RunResult:
        """Execute one run and return its orchestrator-facing metric projection."""
        try:
            request = self._request_for_run(run)
            completed = self.installation.execute(request)
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
        """Project one authored v2 run and bind it to an executable pair.

        Python never resolves runner-owned inputs and never projects a second
        protocol shape. The selected pair adapter is therefore the only load
        boundary between Config v2 and its prepared Rust harness.
        """
        authored = self.installation.project_authored_request(
            run,
            operation="execute",
        )
        self.installation.preflight_request(authored)
        return authored


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
            "native runner must write exactly one terminal JSON line to stdout; "
            f"received {len(lines)} non-empty lines{process}{detail}"
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
        "protocol_version": protocol_version,
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
    """Render either protocol generation's typed failure without using stderr."""
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
    return "native runner failed without an error message"


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
