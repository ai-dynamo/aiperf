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
from aiperf.orchestrator.rust_wire import (
    RUNNER_PROTOCOL_VERSION,
    build_run_request,
    validate_v1_selection,
)

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
            # This must precede config resolvers: they may create artifact
            # directories or warm tokenizer caches. Unknown endpoint identity
            # is an exact-runner compatibility error, not a Python plugin lookup.
            self.installation.preflight_endpoint(str(run.cfg.endpoint.type))
            validate_v1_selection(run.cfg)
            self._resolve_run(run)
            request = build_run_request(run)
            completed = self.installation.execute(request)
            terminal = _parse_terminal(
                completed.stdout,
                run,
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

    @staticmethod
    def _resolve_run(run: BenchmarkRun) -> None:
        # ``dag_jsonl`` is an authored graph program, not a linear Python
        # dataset. The runner's direct GraphInputAdapter owns its sole parse,
        # topology validation, root selection, and Graph-IR lowering. Running
        # DatasetResolver here would parse it once in Python, replace authored
        # sequential sampling with the legacy loader preference, then make Rust
        # parse it again. TimingResolver depends on that legacy dataset result,
        # and CommConfigResolver exists only for the removed ZMQ execution path.
        dataset = run.cfg.get_default_dataset()
        if str(getattr(dataset, "format", "")) == "dag_jsonl":
            from aiperf.config.resolution.resolvers import (
                ArtifactDirResolver,
                ConfigResolverChain,
                GpuMetricsResolver,
                TokenizerResolver,
            )

            ConfigResolverChain(
                [
                    ArtifactDirResolver(),
                    TokenizerResolver(),
                    GpuMetricsResolver(),
                ]
            ).resolve_all(run)
            return

        # Linear workloads retain the compatibility resolver chain until the
        # strict authored protocol-v2 request replaces protocol v1.
        from aiperf.config.resolution.resolvers import build_default_resolver_chain

        build_default_resolver_chain().resolve_all(run)


def _parse_terminal(
    stdout: bytes,
    run: BenchmarkRun,
    *,
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
    detail = redact_string(
        str(terminal.get("error") or "native runner failed without an error message")
    )
    stderr = redact_string(completed.stderr.decode(errors="replace")).strip()
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
