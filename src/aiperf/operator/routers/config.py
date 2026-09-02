# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-side config endpoints for clients to discover current settings.

Exposes a read-only view of operator configuration that clients need to
replicate server-side policies locally. For example, ``aiperf kube results
list-runs --preview`` reads ``/api/v1/config/retention`` so it can mark
which runs the operator would reap under the current retention settings
without having to guess defaults.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field


class RetentionConfigResponse(BaseModel):
    """Current retention policy settings for the operator's results PVC."""

    retain_runs: int = Field(
        ge=0,
        description="Current AIPERF_RESULTS_RETAIN_RUNS setting. "
        "Count policy: keep N newest runs per job.",
    )
    retain_days: int = Field(
        ge=0,
        description="Current AIPERF_RESULTS_RETAIN_DAYS setting "
        "(0 = age policy disabled). Age policy: keep runs newer than D days. "
        "A run is deleted only when BOTH policies agree to reap it.",
    )


class FeaturesResponse(BaseModel):
    """Boot-time feature flags the SPA needs to gate top-nav entries."""

    dashboard_enabled: bool = Field(
        description="Whether the Plotly Dash sidecar is wired up. When true, "
        "the SPA shows the 'Plots ↗' top-nav link pointing at /dashboard/. "
        "Reflects AIPERF_DASHBOARD_PROXY_ENABLED on the results-server "
        "container, which Helm sets only when dashboard.enabled=true."
    )


def create_config_router() -> APIRouter:
    """Create the ``/api/v1/config/*`` router exposing operator settings.

    Kept as a factory so tests can mount the router on a fresh FastAPI
    app and patch ``OperatorEnvironment`` per-test without module-level
    state leaking between cases.
    """
    from aiperf.operator.environment import OperatorEnvironment

    router = APIRouter(prefix="/api/v1/config", tags=["config"])

    @router.get("/retention", response_model=RetentionConfigResponse)
    async def get_retention_config() -> RetentionConfigResponse:
        return RetentionConfigResponse(
            retain_runs=OperatorEnvironment.RESULTS.RETAIN_RUNS,
            retain_days=OperatorEnvironment.RESULTS.RETAIN_DAYS,
        )

    @router.get("/features", response_model=FeaturesResponse)
    async def get_features() -> FeaturesResponse:
        return FeaturesResponse(
            dashboard_enabled=OperatorEnvironment.DASHBOARD.PROXY_ENABLED,
        )

    return router
