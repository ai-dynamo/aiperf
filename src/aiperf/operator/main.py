# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes Operator.

Handles AIPerfJob CRD lifecycle with:
- Spec validation and endpoint health checks
- Kubernetes event emission
- Condition tracking (ConfigValid, EndpointReachable, ResourcesCreated, etc.)
- Metrics summary extraction
- Results storage with retry logic
- Job cancellation support
- Job timeout detection
- Pod restart monitoring
- Results TTL cleanup

Run: kopf run -m aiperf.operator.main --verbose

Handler categories dispatched below, in order: startup (configure),
lifecycle (on_create / on_delete / on_cancel / on_benchmark_complete),
and timers (monitor_progress, cleanup_old_results).

All kopf decorators live here so handler modules stay decorator-free
and are independently testable.
"""

from __future__ import annotations

import logging
from typing import Any

import kopf

from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_GROUP,
    AIPERF_PLURAL,
    AIPERF_VERSION,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers import cleanup, create, lifecycle, monitor

logger = logging.getLogger(__name__)


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_: Any) -> None:
    """Configure operator settings."""
    settings.persistence.finalizer = f"{AIPERF_GROUP}/finalizer"
    settings.posting.level = logging.INFO


@kopf.on.create(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL)
async def on_create(
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    patch: kopf.Patch,
    **_: Any,
) -> dict[str, Any]:
    """Create ConfigMap and JobSet for the benchmark job."""
    return await create.on_create(
        body=body, spec=spec, name=name, namespace=namespace, uid=uid, patch=patch
    )


@kopf.on.delete(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL)
async def on_delete(
    name: str, namespace: str, status: dict[str, Any], **_: Any
) -> None:
    """Clean up cached ProgressClient on CR deletion."""
    await lifecycle.on_delete(name=name, namespace=namespace, status=status)


@kopf.on.update(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL, field="spec.cancel")
async def on_cancel(
    body: dict[str, Any],
    spec: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle cancellation request via spec.cancel field."""
    await lifecycle.on_cancel(
        body=body, spec=spec, status=status, name=name, namespace=namespace, patch=patch
    )


@kopf.on.update(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    annotations={Annotations.BENCHMARK_COMPLETE: "true"},
)
async def on_benchmark_complete(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Handle benchmark completion signal from controller pod."""
    await lifecycle.on_benchmark_complete(
        body=body, status=status, name=name, namespace=namespace, patch=patch
    )


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    interval=OperatorEnvironment.MONITOR.INTERVAL,
    initial_delay=OperatorEnvironment.MONITOR.INITIAL_DELAY,
)
async def monitor_progress(
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Monitor job progress and update status."""
    await monitor.monitor_progress(
        body=body, status=status, spec=spec, name=name, namespace=namespace, patch=patch
    )


@kopf.timer(
    AIPERF_GROUP,
    AIPERF_VERSION,
    AIPERF_PLURAL,
    interval=86400.0,
    initial_delay=3600.0,
    idle=3600.0,
)
async def cleanup_old_results(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    **_: Any,
) -> None:
    """Clean up old results based on TTL."""
    await cleanup.cleanup_old_results(body=body, status=status, name=name)
