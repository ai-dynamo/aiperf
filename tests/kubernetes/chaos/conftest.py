# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for chaos scenarios.

``chaos_injector`` is the single entry point for pod/CR chaos;
``toxiproxy_injector`` drives REST/API disruption tests via a
cluster-deployed toxiproxy; ``mock_server_injector`` drives
benchmark-runtime faults against the k8s harness mock server. Compose
with the package-level ``operator_ready`` and ``kubectl`` fixtures from
``tests/kubernetes/conftest.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos.mock_server_injector import MockServerInjector
from tests.kubernetes.chaos.toxiproxy import ToxiproxyInjector
from tests.kubernetes.helpers.kubectl import KubectlClient


@pytest.fixture
def chaos_injector(kubectl: KubectlClient) -> ChaosInjector:
    """Provide a ``ChaosInjector`` bound to the package-scoped cluster."""
    return ChaosInjector(kubectl=kubectl)


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def toxiproxy_injector(
    kubectl: KubectlClient,
) -> AsyncIterator[ToxiproxyInjector]:
    """Package-scoped toxiproxy fixture.

    Must share scope with ``kubectl`` (package-scoped in
    ``tests/kubernetes/conftest.py``). Applies ``fixtures/toxiproxy.yaml``,
    opens an admin port-forward, and tears the namespace down at package
    end. Individual tests must call ``await injector.reset()`` in their
    own ``finally`` to keep proxies/toxics from leaking across tests.
    """
    injector = ToxiproxyInjector()
    await injector.ensure_deployed(kubectl)
    try:
        yield injector
    finally:
        await injector.teardown(kubectl)


@pytest_asyncio.fixture
async def mock_server_injector(
    kubectl: KubectlClient,
) -> AsyncIterator[MockServerInjector]:
    """Function-scoped mock-server chaos injector.

    Auto-restores every mutation applied during the test by calling
    ``injector.restore()`` on teardown.
    """
    injector = MockServerInjector(kubectl=kubectl)
    try:
        yield injector
    finally:
        await injector.restore()
