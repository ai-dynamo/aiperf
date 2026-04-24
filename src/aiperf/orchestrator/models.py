# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for multi-run orchestration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.export_models import JsonMetricResult


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
