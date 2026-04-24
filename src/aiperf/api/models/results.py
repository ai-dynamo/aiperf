# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared response models for the ``/api/results`` endpoints.

These models are used by both the in-process results router
(``aiperf.api.routers.results``) and the standalone controller-side results
sidecar (``aiperf.kubernetes.results_sidecar``); defining them in one place
keeps the two HTTP surfaces contractually identical.
"""

from __future__ import annotations

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel


class ResultFileInfo(AIPerfBaseModel):
    """Metadata for a single result file."""

    name: str = Field(description="Filename of the result artifact")
    size: int = Field(description="File size in bytes")


class ResultsListResponse(AIPerfBaseModel):
    """Response for listing available result files."""

    files: list[ResultFileInfo] = Field(
        default_factory=list, description="Available result files"
    )
