# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local subprocess executor for MultiRunOrchestrator.

Runs each BenchmarkRun in a fresh subprocess of aiperf.orchestrator.subprocess_runner.
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

from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from aiperf.common.models.export_models import JsonMetricResult
    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun


logger = logging.getLogger(__name__)

__all__ = ["LocalSubprocessExecutor"]


class LocalSubprocessExecutor(RunExecutor):
    """Run benchmarks via subprocess of aiperf.orchestrator.subprocess_runner."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def derive_id(self, plan: BenchmarkPlan, var_idx: int, trial: int) -> str:
        return uuid4().hex

    async def execute(self, run: BenchmarkRun) -> RunResult:
        """Run subprocess in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self._execute_sync, run)

    def _execute_sync(self, run: BenchmarkRun) -> RunResult:
        artifacts_path = run.artifact_dir
        artifacts_path.mkdir(parents=True, exist_ok=True)
        try:
            config_file = self._prepare_run_artifacts(run, artifacts_path)
            result = self._run_benchmark_subprocess(config_file, run)

            if result.returncode != 0:
                return self._failure_from_subprocess(result, run.label, artifacts_path)

            summary_metrics = self._extract_summary_metrics(run)
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
        """Serialize the run config for the subprocess to read.

        ``run.cfg.endpoint`` has a JSON field_serializer that redacts
        ``api_key`` to ``<redacted>`` on every ``model_dump(mode="json")``
        call - so the file written here NEVER contains the plaintext key.
        The actual key is forwarded to the subprocess via the
        ``AIPERF_INJECTED_API_KEY`` environment variable in
        :meth:`_run_benchmark_subprocess`; the subprocess restores it
        onto the loaded config before running. This eliminates the
        plaintext-on-disk window entirely - no try/finally race, no
        partial-failure leak across sweep cells.
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
        run: BenchmarkRun,
    ) -> subprocess.CompletedProcess[str]:
        """Run the benchmark subprocess runner and return its completed-process.

        The api_key is forwarded via ``AIPERF_INJECTED_API_KEY`` rather
        than written into ``run_config.json`` (which would be redacted by
        the field_serializer anyway). ``subprocess_runner.main`` consumes
        and unsets the variable before invoking the benchmark, so neither
        the subprocess's own children nor any logging path sees it.
        """
        import os

        env = os.environ.copy()
        api_key = run.cfg.endpoint.api_key
        if api_key is not None:
            env["AIPERF_INJECTED_API_KEY"] = api_key
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
            env=env,
        )

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
        self, run: BenchmarkRun
    ) -> dict[str, JsonMetricResult]:
        """Extract run-level summary statistics from artifacts.

        Reads the summary JSON file (or its ``.zst`` variant) at the path
        computed by :attr:`ArtifactsConfig.profile_export_json_file`, which
        honors ``--profile-export-prefix`` and falls back to the historical
        ``profile_export_aiperf.json`` default when no prefix is set.

        Returns empty dict if the file is missing or unparsable.
        """
        from aiperf.common.models.export_models import JsonMetricResult

        artifacts_path = run.artifact_dir
        json_name = run.cfg.artifacts.profile_export_json_file.name
        json_file = artifacts_path / json_name
        zst_file = artifacts_path / f"{json_name}.zst"

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
            metrics = {}
            for field_name, field_value in data.items():
                if isinstance(field_value, dict) and "unit" in field_value:
                    metrics[field_name] = JsonMetricResult(**field_value)
            return metrics

        except (OSError, ValueError, orjson.JSONDecodeError) as e:
            logger.warning(f"Error extracting metrics from {json_file}: {e}")
            return {}
