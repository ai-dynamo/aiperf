# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watchdog adapter for E2E tests.

Creates a K8sWatchdogSource from the test KubectlClient's connection
settings, so E2E tests use the exact same watchdog as production.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.watchdog import BenchmarkWatchdog, K8sWatchdogSource
from tests.kubernetes.helpers.kubectl import KubectlClient

__all__ = ["BenchmarkWatchdog", "K8sWatchdogSource", "make_watchdog_source"]


@asynccontextmanager
async def make_watchdog_source(
    kubectl: KubectlClient,
) -> AsyncIterator[K8sWatchdogSource]:
    """Yield a watchdog source using the test client's cluster settings."""
    async with k8s_client(
        kubeconfig=kubectl.kubeconfig,
        context=kubectl.context,
        wait_for_credentials=False,
    ) as api:
        yield K8sWatchdogSource(api)
