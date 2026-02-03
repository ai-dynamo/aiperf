# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes-style health check mixin for AIPerf services.

Provides liveness and readiness probe support for all services:
- is_healthy(): Liveness check - is the service alive and not deadlocked?
- is_ready(): Readiness check - is the service ready to accept traffic?
- get_health_details(): Detailed health info for debugging
"""

from __future__ import annotations

from dataclasses import dataclass

from aiperf.common.enums import LifecycleState


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """Result of the health check."""

    service_id: str
    state: LifecycleState
    healthy: bool
    ready: bool


class HealthCheckMixin:
    """Kubernetes-style health checks for all services.

    This mixin provides the core health check logic that can be used by:
    - FastAPIService: Via FastAPI routes (/healthz, /readyz)
    - Other services: Via HealthServerMixin (lightweight asyncio HTTP server)

    The mixin expects the service to have a `state` property returning LifecycleState.
    """

    # These will be provided by AIPerfLifecycleMixin
    state: LifecycleState
    id: str

    def is_healthy(self) -> bool:
        """Liveness check: Is the service alive and not deadlocked?

        Returns True if the service is not in a FAILED state. This indicates
        the process is running and responsive - K8s should not restart it.

        Returns:
            True if healthy, False if the service should be restarted.
        """
        return self.state != LifecycleState.FAILED

    def is_ready(self) -> bool:
        """Readiness check: Is the service ready to accept traffic?

        Returns True only when the service is in RUNNING state. This indicates
        the service has completed initialization and can handle requests.

        Returns:
            True if ready to accept traffic, False otherwise.
        """
        return self.state == LifecycleState.RUNNING

    def get_health_details(self) -> HealthCheckResult:
        """Get detailed health information for debugging.

        Returns:
            HealthCheckResult with service health details.
        """
        return HealthCheckResult(
            service_id=getattr(self, "id", "unknown"),
            state=self.state,
            healthy=self.is_healthy(),
            ready=self.is_ready(),
        )
