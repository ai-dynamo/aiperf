# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for aggregate exporters."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import aiofiles

from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.orchestrator.aggregation.base import AggregateResult


def __getattr__(name: str) -> float:
    """Module-level back-compat shim for ``CONTEXT_OVERFLOW_RATE_LIMIT``.

    The threshold now lives on ``Environment.AGENTX.CONTEXT_OVERFLOW_RATE_LIMIT``
    (env var ``AIPERF_AGENTX_CONTEXT_OVERFLOW_RATE_LIMIT``); this shim keeps
    existing imports working and resolves the value lazily so test-time env
    overrides take effect without re-importing the module.
    """
    if name == "CONTEXT_OVERFLOW_RATE_LIMIT":
        return Environment.AGENTX.CONTEXT_OVERFLOW_RATE_LIMIT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


CONTEXT_OVERFLOW_REASON = "context_overflow_rate_exceeded"
RUN_CANCELLED_REASON = "run_cancelled"


def _build_run_metadata_dict(
    *,
    scenario_name: str | None,
    submission_valid: bool | None,
    submission_invalid_reasons: list[str] | None = None,
) -> dict:
    """Build the run-metadata sub-dict for the aggregate export.

    Returns an empty dict when ``scenario_name`` is ``None`` so non-scenario
    runs are not polluted with submission-tracking fields. When
    ``scenario_name`` is set, returns the ``scenario`` name plus a coerced
    ``submission_valid`` bool, and includes ``submission_invalid_reasons``
    only when that list is non-empty.

    Args:
        scenario_name: Active scenario identifier, or ``None`` for a
            non-scenario run.
        submission_valid: Whether the run is a valid scenario submission.
            Coerced to ``bool`` (``None`` becomes ``False``) when emitted.
        submission_invalid_reasons: Optional list of machine-readable
            reason codes (e.g. ``"unsafe_override"``,
            ``"context_overflow_rate_exceeded"``).

    Returns:
        A dict suitable for merging into the top-level aggregate JSON output.
    """
    md: dict = {}
    if scenario_name is not None:
        md["scenario"] = scenario_name
        md["submission_valid"] = bool(submission_valid)
        if submission_invalid_reasons:
            md["submission_invalid_reasons"] = list(submission_invalid_reasons)
    return md


def compute_submission_outcome(
    *,
    scenario_name: str | None,
    validator_submission_valid: bool | None,
    validator_reasons: list[str] | None = None,
    total_responses: int = 0,
    context_overflow_count: int = 0,
    was_cancelled: bool = False,
    runtime_invalid_reasons: list[str] | None = None,
) -> tuple[bool | None, list[str]]:
    """Combine validator outcome with runtime threshold checks into a final verdict.

    The validator-side outcome covers static config violations (handled at
    UserConfig.model_post_init by ``validate_scenario``). This helper folds
    in runtime-only signals that are only knowable post-run -- the >1%
    context-overflow rate per spec §7, post-run validation failures, and early
    cancellation (Ctrl+C): a cancelled run produces partial metrics and is never
    a valid submission.

    Rate semantics: strictly greater than
    ``Environment.AGENTX.CONTEXT_OVERFLOW_RATE_LIMIT`` (default 0.01 per
    spec §7, override via ``AIPERF_AGENTX_CONTEXT_OVERFLOW_RATE_LIMIT``)
    flips ``submission_valid`` to False; equal-to is accepted (boundary
    behavior pinned by tests). When ``total_responses == 0`` the rate is
    treated as 0 (undefined / no successful responses), so the overflow
    rule does not flip submission validity in that case -- other failure
    signals surface a 0-success run.

    When ``scenario_name`` is None this is a no-scenario run and the
    function returns ``(None, [])`` -- callers should drop the
    ``submission_valid`` field from the output entirely.

    Args:
        scenario_name: Active scenario, or None for a non-scenario run.
        validator_submission_valid: Outcome from ``validate_scenario`` --
            True if the static lock was satisfied, False under
            ``--unsafe-override`` with violations, None for non-scenario.
        validator_reasons: Reason codes already collected by the validator
            (e.g. ``"unsafe_override"``).
        total_responses: Total responses received during the run
            (successes + overflow + other failures).
        context_overflow_count: Count of context-overflow responses
            during the run.
        was_cancelled: Whether the run was cancelled early (graceful
            Ctrl+C). True flips ``submission_valid`` to False with reason
            ``"run_cancelled"``.
        runtime_invalid_reasons: Runtime validation reason tags to merge into
            the final verdict. Any value makes the submission invalid.

    Returns:
        A ``(submission_valid, reasons)`` tuple suitable for feeding into
        ``_build_run_metadata_dict``. ``submission_valid`` is ``None``
        when ``scenario_name`` is None.
    """
    if scenario_name is None:
        return None, []

    reasons: list[str] = list(validator_reasons or [])
    valid: bool = (
        bool(validator_submission_valid)
        if validator_submission_valid is not None
        else True
    )

    if total_responses > 0:
        rate = context_overflow_count / total_responses
        if rate > Environment.AGENTX.CONTEXT_OVERFLOW_RATE_LIMIT:
            valid = False
            if CONTEXT_OVERFLOW_REASON not in reasons:
                reasons.append(CONTEXT_OVERFLOW_REASON)

    if was_cancelled:
        valid = False
        if RUN_CANCELLED_REASON not in reasons:
            reasons.append(RUN_CANCELLED_REASON)

    for reason in runtime_invalid_reasons or []:
        valid = False
        if reason not in reasons:
            reasons.append(reason)

    return valid, reasons


@dataclass(slots=True)
class AggregateExporterConfig:
    """Configuration for aggregate exporters.

    Simpler than ExporterConfig because aggregate exports don't need:
    - ProfileResults (single-run data)
    - TelemetryExportData (per-run telemetry)
    - ServerMetricsResults (per-run server metrics)
    - Full benchmark config (just need output directory)

    Attributes:
        result: AggregateResult to export
        output_dir: Directory where export file will be written
    """

    result: AggregateResult
    output_dir: Path


class AggregateBaseExporter(AIPerfLoggerMixin, ABC):
    """Base class for all aggregate exporters.

    Provides common functionality:
    - File writing logic
    - Directory creation
    - Error handling
    - Logging

    Subclasses implement:
    - _generate_content() - Format-specific content generation
    - get_file_name() - Output file name
    """

    def __init__(self, config: AggregateExporterConfig, **kwargs) -> None:
        """Initialize aggregate exporter.

        Args:
            config: Configuration for the exporter
            **kwargs: Additional arguments passed to AIPerfLoggerMixin
        """
        super().__init__(**kwargs)
        self._config = config
        self._result = config.result
        self._output_dir = Path(config.output_dir)

    @abstractmethod
    def get_file_name(self) -> str:
        """Return the output file name.

        Returns:
            str: File name (e.g., "profile_export_aiperf_aggregate.json")
        """
        pass

    @abstractmethod
    def _generate_content(self) -> str:
        """Generate export content string.

        Subclasses implement format-specific content generation.

        Returns:
            str: Complete content string ready to write to file
        """
        pass

    async def export(self) -> Path:
        """Export aggregate result to file.

        Creates output directory, generates content, and writes to file.

        Returns:
            Path: Path to written file

        Raises:
            Exception: If file writing fails
        """
        await asyncio.to_thread(self._output_dir.mkdir, parents=True, exist_ok=True)

        file_path = self._output_dir / self.get_file_name()

        self.debug(lambda: f"Exporting aggregate data to: {file_path}")

        try:
            content = self._generate_content()

            async with aiofiles.open(file_path, "w", newline="", encoding="utf-8") as f:
                await f.write(content)

            self.info(f"Exported aggregate data to: {file_path}")
            return file_path

        except Exception as e:
            self.error(f"Failed to export to {file_path}: {e}")
            raise
