# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local subprocess executor for MultiRunOrchestrator.

Runs each BenchmarkRun in a fresh subprocess of aiperf.orchestrator.subprocess_runner.
Body of MultiRunOrchestrator._execute_single_run prior to the executor-seam
refactor, lifted here so MultiRunOrchestrator can iterate variations x trials
through any executor (Task 8 changes the orchestrator side).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import orjson

from aiperf.common.redact import REDACTED_VALUE
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from aiperf.common.models.export_models import JsonMetricResult
    from aiperf.config.benchmark import BenchmarkPlan, BenchmarkRun


logger = logging.getLogger(__name__)

__all__ = ["LocalSubprocessExecutor"]


class LocalSubprocessExecutor(RunExecutor):
    """Run benchmarks via subprocess of aiperf.orchestrator.subprocess_runner."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def derive_id(self, plan: BenchmarkPlan, var_idx: int, trial: int) -> str:
        return uuid4().hex[:12]

    async def execute(self, run: BenchmarkRun) -> RunResult:
        """Run subprocess in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self._execute_sync, run)

    def _execute_sync(self, run: BenchmarkRun) -> RunResult:
        artifacts_path = run.artifact_dir
        artifacts_path.mkdir(parents=True, exist_ok=True)
        try:
            config_file = self._prepare_run_artifacts(run, artifacts_path)
            result = self._run_benchmark_subprocess(config_file)
            self._write_redacted_config(run, config_file)

            if result.returncode != 0:
                return self._failure_from_subprocess(result, run.label, artifacts_path)

            summary_metrics = self._extract_summary_metrics(artifacts_path)
            return self._build_result_from_metrics(
                summary_metrics, run.label, artifacts_path
            )
        except Exception as e:
            logger.exception(f"Error executing run {run.label}")
            return RunResult(
                label=run.label or f"run_{run.trial:04d}",
                success=False,
                error=str(e),
                artifacts_path=artifacts_path,
            )

    @staticmethod
    def _prepare_run_artifacts(run: BenchmarkRun, artifacts_path: Path) -> Path:
        """Serialize the run config (with secrets) for the subprocess to read.

        Overwritten with redacted copy after the subprocess returns.
        """
        config_file = artifacts_path / "run_config.json"
        with open(config_file, "wb") as f:
            f.write(
                orjson.dumps(
                    run.model_dump(mode="json", exclude_none=True),
                    option=orjson.OPT_INDENT_2,
                )
            )
        return config_file

    @staticmethod
    def _run_benchmark_subprocess(
        config_file: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run the benchmark subprocess runner and return its completed-process."""
        # No timeout - SystemController handles benchmark duration internally.
        # stdin/stdout pass through so Textual can detect TTY and render live dashboard.
        # -u forces unbuffered output for live dashboard rendering.
        return subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "aiperf.orchestrator.subprocess_runner",
                str(config_file),
            ],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def _write_redacted_config(run: BenchmarkRun, config_file: Path) -> None:
        """Overwrite the on-disk config file with a redacted copy so secrets don't persist."""
        redacted = run.model_dump(mode="json", exclude_none=True)
        if "cfg" in redacted and "endpoint" in redacted["cfg"]:
            endpoint = redacted["cfg"]["endpoint"]
            if "api_key" in endpoint and endpoint["api_key"] is not None:
                endpoint["api_key"] = REDACTED_VALUE
        with open(config_file, "wb") as f:
            f.write(orjson.dumps(redacted, option=orjson.OPT_INDENT_2))

    @staticmethod
    def _failure_from_subprocess(
        result: subprocess.CompletedProcess[str],
        label: str,
        artifacts_path: Path,
    ) -> RunResult:
        """Build a failed RunResult from a non-zero subprocess exit."""
        error_msg = f"Benchmark failed with exit code {result.returncode}"
        if result.stderr:
            error_msg += f"\nStderr: {result.stderr[-2000:]}"
        logger.error(error_msg)
        return RunResult(
            label=label,
            success=False,
            error=error_msg,
            artifacts_path=artifacts_path,
        )

    @staticmethod
    def _build_result_from_metrics(
        summary_metrics: dict[str, JsonMetricResult],
        label: str,
        artifacts_path: Path,
    ) -> RunResult:
        """Classify success/failure from extracted summary metrics."""
        if not summary_metrics:
            error_msg = (
                "No metrics found in artifacts - run may have failed to complete"
            )
            logger.error(error_msg)
            return RunResult(
                label=label,
                success=False,
                error=error_msg,
                artifacts_path=artifacts_path,
            )

        request_count_metric = summary_metrics.get("request_count")
        error_request_count_metric = summary_metrics.get("error_request_count")

        if not request_count_metric or request_count_metric.avg == 0:
            if error_request_count_metric and error_request_count_metric.avg > 0:
                error_msg = f"All {int(error_request_count_metric.avg)} requests failed"
            else:
                error_msg = "No requests completed"
            logger.error(error_msg)
            return RunResult(
                label=label,
                success=False,
                error=error_msg,
                artifacts_path=artifacts_path,
            )

        return RunResult(
            label=label,
            success=True,
            summary_metrics=summary_metrics,
            artifacts_path=artifacts_path,
        )

    def _extract_summary_metrics(
        self, artifacts_path: Path
    ) -> dict[str, JsonMetricResult]:
        """Extract run-level summary statistics from artifacts.

        Reads the profile_export_aiperf.json (or .zst variant) written by the
        SystemController and returns a dict of metric name -> JsonMetricResult.
        Returns empty dict if the file is missing or unparseable.
        """
        from aiperf.common.models.export_models import JsonMetricResult

        json_file = artifacts_path / "profile_export_aiperf.json"
        zst_file = artifacts_path / "profile_export_aiperf.json.zst"

        if zst_file.exists():
            json_file = zst_file
        elif not json_file.exists():
            logger.warning(f"Profile export file not found: {json_file}")
            return {}

        try:
            raw = json_file.read_bytes()
            if json_file.suffix == ".zst":
                import io

                import zstandard

                raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
            data = orjson.loads(raw)
            return JsonMetricResult.project_summary_dict(data)

        except (OSError, ValueError, orjson.JSONDecodeError) as e:
            logger.warning(f"Error extracting metrics from {json_file}: {e}")
            return {}
