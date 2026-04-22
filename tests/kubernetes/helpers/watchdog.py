# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watchdog adapter for E2E tests.

Creates a K8sWatchdogSource from the test KubectlClient's connection
settings, so E2E tests use the exact same watchdog as production.
"""

from __future__ import annotations

from aiperf.kubernetes.watchdog import BenchmarkWatchdog, K8sWatchdogSource
from tests.kubernetes.helpers.kubectl import KubectlClient

__all__ = ["BenchmarkWatchdog", "K8sWatchdogSource", "make_watchdog_source"]


async def make_watchdog_source(kubectl: KubectlClient) -> K8sWatchdogSource:
    """Create a K8sWatchdogSource from a test KubectlClient's connection settings."""
    return await K8sWatchdogSource.create(
        kubeconfig=kubectl.kubeconfig,
        kube_context=kubectl.context,
    )
