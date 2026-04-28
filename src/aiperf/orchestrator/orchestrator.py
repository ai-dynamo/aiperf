# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-run orchestrator for AIPerf benchmarks."""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import (
    ExecutionStrategy,
    FixedTrialsStrategy,
    ParameterSweepStrategy,
    SweepConfidenceStrategy,
    SweepMode,
)

if TYPE_CHECKING:
    from aiperf.common.models.export_models import JsonMetricResult

logger = logging.getLogger(__name__)

__all__ = [
    "MultiRunOrchestrator",
]


class MultiRunOrchestrator:
    """Orchestrates execution of multiple benchmark runs using a strategy.

    The orchestrator is a thin coordinator. Strategy objects own:
    - Execution iteration (via execute() for custom loops, or should_continue/get_next_config for generic)
    - Aggregation logic (aggregate())
    - Export logic (export_aggregates())
    - Result tagging (tag_result())
    - Failure collection (collect_failed_values())

    The orchestrator provides:
    - Strategy resolution from config (_resolve_strategy)
    - A generic execution loop for simple strategies (_execute_loop)
    - Single-run subprocess execution (_execute_single_run)
    - Metrics extraction from artifacts (_extract_summary_metrics)
    """

    def __init__(
        self,
        base_dir: Path,
        service_config: ServiceConfig,
    ):
        """Initialize MultiRunOrchestrator.

        Args:
            base_dir: Base directory for all artifacts
            service_config: Service configuration for SystemController
        """
        self.base_dir = Path(base_dir)
        self.service_config = service_config

    def execute_and_export(
        self, base_config: UserConfig, strategy: ExecutionStrategy | None = None
    ) -> list[RunResult]:
        """Execute benchmark, aggregate results, and export aggregates.

        This is the main entry point that handles the complete workflow:
        1. Resolve strategy (if not provided)
        2. Execute runs (strategy may own its loop or use generic loop)
        3. Aggregate results via strategy
        4. Export aggregates via strategy

        Args:
            base_config: Base benchmark configuration
            strategy: Optional execution strategy. If None, auto-detected from config.

        Returns:
            List of RunResult, one per run executed
        """
        if strategy is None:
            strategy = self._resolve_strategy(base_config)

        results = self._execute(base_config, strategy)

        aggregate = strategy.aggregate(results, base_config)
        if aggregate is not None:
            strategy.export_aggregates(aggregate, self.base_dir)

        return results

    def execute(
        self, base_config: UserConfig, strategy: ExecutionStrategy | None = None
    ) -> list[RunResult]:
        """Execute benchmark runs without aggregation/export.

        Useful for testing or when callers want to handle aggregation themselves.

        Args:
            base_config: Base benchmark configuration
            strategy: Optional execution strategy. If None, auto-detected from config.

        Returns:
            List of RunResult, one per run executed
        """
        if strategy is None:
            strategy = self._resolve_strategy(base_config)

        return self._execute(base_config, strategy)

    def _resolve_strategy(self, config: UserConfig) -> ExecutionStrategy:
        """Detect mode from config and return the appropriate strategy.

        Only called from multi-run paths (sweep, confidence, or both).
        Single-run benchmarks go through _run_single_benchmark() in cli_runner
        and never reach the orchestrator.

        Args:
            config: User configuration

        Returns:
            Appropriate ExecutionStrategy

        Raises:
            ValueError: If no multi-run mode is detected
        """
        has_sweep = config.loadgen.get_sweep_parameter() is not None
        has_confidence = config.loadgen.num_profile_runs > 1

        if has_sweep and has_confidence:
            logger.info(
                f"Executing parameter sweep with confidence trials "
                f"(mode: {config.loadgen.parameter_sweep_mode})"
            )
            return SweepConfidenceStrategy(
                sweep=self._create_sweep_strategy(config),
                confidence=self._create_confidence_strategy(config),
                mode=SweepMode(config.loadgen.parameter_sweep_mode),
            )
        if has_sweep:
            logger.info("Executing parameter sweep (no confidence trials)")
            return self._create_sweep_strategy(config)
        if has_confidence:
            logger.info(
                f"Executing confidence trials (n={config.loadgen.num_profile_runs})"
            )
            return self._create_confidence_strategy(config)

        raise ValueError(
            "MultiRunOrchestrator requires sweep or confidence mode. "
            "Single-run benchmarks should use _run_single_benchmark() directly."
        )

    def _execute(
        self, config: UserConfig, strategy: ExecutionStrategy
    ) -> list[RunResult]:
        """Execute runs using the strategy.

        First checks if the strategy wants to own its loop (execute() returns
        a list). If not, falls back to the generic loop.

        Args:
            config: Base benchmark configuration
            strategy: Execution strategy

        Returns:
            List of run results
        """
        custom_results = strategy.execute(
            config, self._execute_single_run, self.base_dir
        )
        if custom_results is not None:
            return custom_results

        return self._execute_loop(config, strategy)

    def _execute_loop(
        self, config: UserConfig, strategy: ExecutionStrategy
    ) -> list[RunResult]:
        """Generic execution loop driven entirely by strategy.

        Used by FixedTrialsStrategy and ParameterSweepStrategy which don't
        need custom iteration.

        Args:
            config: Base benchmark configuration
            strategy: Execution strategy

        Returns:
            List of run results
        """
        results: list[RunResult] = []
        run_index = 0

        logger.info(
            f"Starting multi-run benchmark with strategy: {strategy.__class__.__name__}"
        )

        strategy.validate_config(config)

        while strategy.should_continue(results):
            run_config = strategy.get_next_config(config, results)
            label = strategy.get_run_label(run_index)
            artifact_path = strategy.get_run_path(self.base_dir, run_index)

            logger.info(f"[{run_index + 1}] Executing {label}...")

            result = self._execute_single_run(run_config, label, artifact_path)
            result = strategy.tag_result(result, run_index)
            results.append(result)

            if result.success:
                logger.info(f"[{run_index + 1}] {label} completed successfully")
            else:
                logger.error(f"[{run_index + 1}] {label} failed: {result.error}")

            run_index += 1

            if strategy.should_continue(results):
                cooldown = strategy.get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Applying cooldown: {cooldown}s")
                    time.sleep(cooldown)

        successful = sum(1 for r in results if r.success)
        logger.info(f"All runs complete: {successful}/{len(results)} successful")

        failed_values = strategy.collect_failed_values(results)
        if failed_values:
            logger.warning(
                f"Some sweep values failed: {[fv['value'] for fv in failed_values]}"
            )
            for fv in failed_values:
                logger.warning(f"  {fv['parameter_name']}={fv['value']}: {fv['error']}")

        return results

    def _create_sweep_strategy(self, config: UserConfig) -> ParameterSweepStrategy:
        """Create parameter sweep strategy from config.

        Args:
            config: User configuration with sweep parameters

        Returns:
            ParameterSweepStrategy configured from config

        Raises:
            ValueError: If no sweep parameter is detected in config
        """
        sweep_info = config.loadgen.get_sweep_parameter()
        if not sweep_info:
            raise ValueError(
                "No sweep parameter detected in configuration. "
                "To enable parameter sweep, provide a parameter as a comma-separated list. "
                "Example: --concurrency 10,20,30"
            )

        param_name, param_values = sweep_info

        return ParameterSweepStrategy(
            parameter_name=param_name,
            parameter_values=param_values,
            cooldown_seconds=config.loadgen.parameter_sweep_cooldown_seconds,
            same_seed=config.loadgen.parameter_sweep_same_seed,
            auto_set_seed=True,
        )

    def _create_confidence_strategy(self, config: UserConfig) -> FixedTrialsStrategy:
        """Create confidence/fixed trials strategy from config.

        Args:
            config: User configuration with confidence parameters

        Returns:
            FixedTrialsStrategy configured from config
        """
        return FixedTrialsStrategy(
            num_trials=config.loadgen.num_profile_runs,
            cooldown_seconds=config.loadgen.profile_run_cooldown_seconds,
            auto_set_seed=config.loadgen.set_consistent_seed,
            disable_warmup_after_first=config.loadgen.profile_run_disable_warmup_after_first,
        )

    def _execute_single_run(
        self, config: UserConfig, label: str, artifact_path: Path
    ) -> RunResult:
        """Execute a single benchmark run in a subprocess.

        Each run is executed in a separate subprocess to ensure complete isolation.
        This allows the SystemController to call os._exit() without affecting the orchestrator.

        Args:
            config: Benchmark configuration
            label: Label for this run (e.g., "run_0001", "concurrency_10")
            artifact_path: Path where artifacts should be stored

        Returns:
            RunResult with success status and metrics or error
        """
        try:
            # Ensure artifact directory exists
            artifact_path = Path(artifact_path)
            artifact_path.mkdir(parents=True, exist_ok=True)

            config = config.model_copy(deep=True)
            config.output.artifact_directory = artifact_path

            # Serialize configs to JSON
            # Use exclude_defaults=True to avoid serializing fields that weren't explicitly set
            # This prevents validation errors on deserialization for fields with conditional validators
            config_data = {
                "user_config": config.model_dump(
                    mode="json",
                    exclude_defaults=True,
                    exclude_none=True,
                    context={"include_secrets": True},
                ),
                "service_config": self.service_config.model_dump(
                    mode="json", exclude_defaults=True, exclude_none=True
                ),
            }

            # Write config with secrets for subprocess to read.
            # Overwritten with redacted version after the subprocess finishes.
            config_file = artifact_path / "run_config.json"
            with open(config_file, "wb") as f:
                f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2))

            # Run the benchmark in a subprocess using the dedicated runner module
            # stdin/stdout are passed through to terminal so Textual can detect TTY
            # -u flag forces unbuffered output so live dashboard updates are visible immediately
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",  # Unbuffered output - critical for live dashboard rendering
                    "-m",
                    "aiperf.orchestrator.subprocess_runner",
                    str(config_file),
                ],
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Overwrite config file with redacted version so secrets don't persist in artifacts
            redacted_config_data = {
                "user_config": config.model_dump(
                    mode="json", exclude_defaults=True, exclude_none=True
                ),
                "service_config": self.service_config.model_dump(
                    mode="json", exclude_defaults=True, exclude_none=True
                ),
            }
            with open(config_file, "wb") as f:
                f.write(orjson.dumps(redacted_config_data, option=orjson.OPT_INDENT_2))

            if result.returncode != 0:
                error_msg = f"Benchmark failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nStderr: {result.stderr[-2000:]}"
                logger.error(error_msg)
                return RunResult(
                    label=label,
                    success=False,
                    error=error_msg,
                    artifacts_path=artifact_path,
                )

            # Extract summary metrics from the artifacts
            summary_metrics = self._extract_summary_metrics(artifact_path)

            if not summary_metrics:
                error_msg = (
                    "No metrics found in artifacts - run may have failed to complete"
                )
                logger.error(error_msg)
                return RunResult(
                    label=label,
                    success=False,
                    error=error_msg,
                    artifacts_path=artifact_path,
                )

            # Check if any requests completed successfully
            request_count_metric = summary_metrics.get("request_count")
            error_request_count_metric = summary_metrics.get("error_request_count")

            if not request_count_metric or request_count_metric.avg == 0:
                if error_request_count_metric and error_request_count_metric.avg > 0:
                    error_msg = (
                        f"All {int(error_request_count_metric.avg)} requests failed"
                    )
                    logger.error(error_msg)
                    return RunResult(
                        label=label,
                        success=False,
                        error=error_msg,
                        artifacts_path=artifact_path,
                    )
                error_msg = "No requests completed"
                logger.error(error_msg)
                return RunResult(
                    label=label,
                    success=False,
                    error=error_msg,
                    artifacts_path=artifact_path,
                )

            return RunResult(
                label=label,
                success=True,
                summary_metrics=summary_metrics,
                artifacts_path=artifact_path,
            )
        except Exception as e:
            logger.exception(f"Error executing run {label}")
            return RunResult(
                label=label,
                success=False,
                error=str(e),
                artifacts_path=artifact_path,
            )

    def _extract_summary_metrics(
        self, artifacts_path: Path
    ) -> dict[str, "JsonMetricResult"]:
        """Extract run-level summary statistics from artifacts.

        Reads the profile_export_aiperf.json file written by the SystemController
        and extracts the summary metrics, preserving the full structure with units.

        Args:
            artifacts_path: Path to run artifacts directory

        Returns:
            Dict mapping metric name to JsonMetricResult
        """
        from aiperf.common.models.export_models import JsonMetricResult

        json_file = artifacts_path / "profile_export_aiperf.json"

        if not json_file.exists():
            logger.warning(f"Profile export file not found: {json_file}")
            return {}

        try:
            with open(json_file, "rb") as f:
                data = orjson.loads(f.read())

            metrics = {}
            for field_name, field_value in data.items():
                if isinstance(field_value, dict) and "unit" in field_value:
                    try:
                        metrics[field_name] = JsonMetricResult(**field_value)
                    except Exception as e:
                        logger.debug(f"Skipping field {field_name}: {e}")
                        continue

            return metrics

        except Exception:
            logger.exception(f"Error extracting metrics from {json_file}")
            return {}
