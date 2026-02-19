# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for multi-run orchestration."""

from pathlib import Path
from typing import Any

from pydantic import Field

from aiperf.common.config import UserConfig
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.export_models import JsonMetricResult


class RunConfig(AIPerfBaseModel):
    """Configuration for a single benchmark run."""

    config: UserConfig = Field(description="The benchmark configuration to execute")
    label: str = Field(
        description="Human-readable label for this run (e.g., 'run_0001')"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about this run (e.g., trial number, parameter values)",
    )


class RunResult(AIPerfBaseModel):
    """Result from executing a single benchmark run."""

    label: str = Field(description="Label identifying this run")
    success: bool = Field(description="Whether the run completed successfully")
    summary_metrics: dict[str, JsonMetricResult] = Field(
        default_factory=dict,
        description="Run-level summary statistics (e.g., {'time_to_first_token': JsonMetricResult(unit='ms', avg=150, p99=195)})",
    )
    error: str | None = Field(default=None, description="Error message if run failed")
    artifacts_path: Path | None = Field(
        default=None, description="Path to run artifacts directory"
    )
