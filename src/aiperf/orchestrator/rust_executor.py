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

from aiperf.common.redact import redact_string
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.native_report import (
    export_python_compatibility_reports,
    load_native_report,
    project_native_summary,
)
from aiperf.orchestrator.rust_wire import (
    RUNNER_PROTOCOL_VERSION,
    build_run_request,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun


logger = logging.getLogger(__name__)

__all__ = ["RustSubprocessExecutor"]

_RUNNER_ENV = "AIPERF_RUNNER_BIN"
_NATIVE_REPORT_SCHEMA_VERSION = "2.0"


class RustSubprocessExecutor(RunExecutor):
    """Execute each fully planned run through ``aiperf-runner`` over stdio."""

    def __init__(self, base_dir: Path, *, binary: Path | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.binary = _resolve_runner_binary(binary)
        self.capabilities = _load_capabilities(self.binary)

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
            _require_request_capabilities(self.capabilities, request)
            completed = subprocess.run(
                [str(self.binary)],
                input=orjson.dumps(request),
                capture_output=True,
                check=False,
            )
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


def _load_capabilities(binary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), "--capabilities"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace").strip()
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
    versions = capabilities.get("protocol_versions")
    if not isinstance(versions, list) or RUNNER_PROTOCOL_VERSION not in versions:
        raise RuntimeError(
            f"aiperf-runner does not support protocol {RUNNER_PROTOCOL_VERSION}: "
            f"advertised {versions!r}"
        )
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
            raise ValueError(
                f"aiperf-runner capability {field} must be an array of non-empty strings"
            )
    return capabilities


def _require_request_capabilities(
    capabilities: dict[str, Any], request: dict[str, Any]
) -> None:
    """Fail before launch when a resolved run exceeds the child contract."""
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
        _require_capability(
            capabilities,
            "run_features",
            "http_transport_policy",
        )
    _require_capability(capabilities, "dataset_types", dataset["type"])
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

    if "accuracy" in run:
        _require_capability(capabilities, "run_features", "python_accuracy_evaluator")
    if "outputs_path" in artifacts:
        _require_capability(capabilities, "run_features", "outputs_json")
    if "raw_path" in artifacts:
        _require_capability(capabilities, "run_features", "raw_records")
    gpu_telemetry = run.get("gpu_telemetry")
    if gpu_telemetry is not None:
        _require_capability(capabilities, "run_features", "gpu_telemetry")
        if not isinstance(gpu_telemetry, dict):
            raise ValueError("native run gpu_telemetry must be an object")
        sources = gpu_telemetry.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("native run GPU telemetry requires at least one source")
        for index, source in enumerate(sources):
            if not isinstance(source, dict) or not isinstance(source.get("type"), str):
                raise ValueError(f"native GPU telemetry source {index} omitted type")
            _require_capability(
                capabilities,
                "telemetry_source_types",
                source["type"],
            )
    if "network_latency" in run:
        network_latency = run["network_latency"]
        if not isinstance(network_latency, dict):
            raise ValueError("native run network_latency must be an object")
        _require_capability(capabilities, "run_features", "network_latency")
    server_metrics = run.get("server_metrics")
    if server_metrics is not None:
        _require_capability(capabilities, "run_features", "server_metrics")
        if not isinstance(server_metrics, dict):
            raise ValueError("native run server_metrics must be an object")
        formats = server_metrics.get("formats")
        if not isinstance(formats, list) or not formats:
            raise ValueError("native run server_metrics requires at least one format")
        for format_name in formats:
            if not isinstance(format_name, str):
                raise ValueError("native server metrics formats must be strings")
            _require_capability(
                capabilities,
                "server_metrics_formats",
                format_name,
            )
    live_streaming = run.get("live_streaming")
    if live_streaming is not None:
        if not isinstance(live_streaming, dict):
            raise ValueError("native run live_streaming must be an object")
        _require_capability(
            capabilities,
            "run_features",
            "python_live_streaming",
        )


def _require_capability(
    capabilities: dict[str, Any], field: str, required: str
) -> None:
    advertised = capabilities[field]
    if required not in advertised:
        raise RuntimeError(
            f"aiperf-runner does not support {field}.{required}; "
            f"advertised {advertised!r}"
        )


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
