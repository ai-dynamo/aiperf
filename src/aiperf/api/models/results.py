# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared response models for the ``/api/results`` endpoints.

These models are used by both the in-process results router
(``aiperf.api.routers.results``) and the standalone controller-side results
sidecar (``aiperf.kubernetes.results_sidecar``); defining them in one place
keeps the two HTTP surfaces contractually identical. The final-results payload
(``BenchmarkStatus`` / ``BenchmarkResultsResponse``) lives here too, next to
the file-listing schema it is served alongside.
"""

from __future__ import annotations

from pydantic import Field

from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.models import AIPerfBaseModel
from aiperf.common.models.record_models import ProcessRecordsResult


class ResultFileInfo(AIPerfBaseModel):
    """Metadata for a single result file."""

    name: str = Field(description="Filename of the result artifact")
    size: int = Field(description="File size in bytes", ge=0)


class ResultsListResponse(AIPerfBaseModel):
    """Response for listing available result files."""

    files: list[ResultFileInfo] = Field(
        default_factory=list, description="Available result files"
    )
    ready: bool = Field(
        default=False,
        description="Whether the controller has published the final ready marker.",
    )
    processing: bool = Field(
        default=False,
        description="Whether the controller has marked result export as in progress.",
    )


class BenchmarkStatus(CaseInsensitiveStrEnum):
    """Status of a benchmark run."""

    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


class BenchmarkResultsResponse(AIPerfBaseModel):
    """Final benchmark results response."""

    status: BenchmarkStatus = Field(
        description="Benchmark status: running, complete, or cancelled"
    )
    results: ProcessRecordsResult | None = Field(
        default=None, description="Final benchmark results if complete"
    )
