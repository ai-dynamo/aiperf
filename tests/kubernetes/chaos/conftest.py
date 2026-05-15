# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for chaos scenarios.

``chaos_injector`` is the single entry point for pod/CR chaos;
``toxiproxy_injector`` drives REST/API disruption tests via a
cluster-deployed toxiproxy; ``mock_server_injector`` drives
benchmark-runtime faults against the k8s harness mock server;
``operator_ready_toxiproxy_routed`` redeploys the operator with its
controller-HTTP traffic pinned at the toxiproxy Service so a test can
inject faults on that link. Compose with the package-level
``operator_ready`` and ``kubectl`` fixtures from
``tests/kubernetes/conftest.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos.mock_server_injector import MockServerInjector
from tests.kubernetes.chaos.toxiproxy import (
    TOXIPROXY_APISERVER_PORT,
    TOXIPROXY_CONTROLLER_HTTP_PORT,
    TOXIPROXY_NAMESPACE,
    TOXIPROXY_SERVICE,
    ToxiproxyInjector,
)
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import OperatorDeployer


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


# URL the operator uses via AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE when
# routed through toxiproxy. See the fixture below.
#
# NOTE on shape: ``ProgressClient._base_url`` uses this as a bare URL
# (scheme+host+port, with any trailing slash stripped) and appends
# ``/api/progress`` / ``/api/workers`` / ``/health`` itself. Do NOT append
# ``/api`` here — that would double up the path and every call would 404.
CONTROLLER_HTTP_OVERRIDE_URL = (
    f"http://{TOXIPROXY_SERVICE}.{TOXIPROXY_NAMESPACE}.svc.cluster.local:"
    f"{TOXIPROXY_CONTROLLER_HTTP_PORT}"
)
APISERVER_SERVICE_HOST_OVERRIDE = (
    f"{TOXIPROXY_SERVICE}.{TOXIPROXY_NAMESPACE}.svc.cluster.local"
)
APISERVER_SERVICE_PORT_OVERRIDE = str(TOXIPROXY_APISERVER_PORT)


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def operator_ready_apiserver_toxiproxy_routed(
    kubectl: KubectlClient,
    project_root: Path,
    loaded_images,  # noqa: ANN001 - session-scoped helper, not typed in test surface
    jobset_controller: None,
    mock_server: None,
    k8s_settings,  # noqa: ANN001 - test-fixture dataclass
    operator_job_namespace: str,
    toxiproxy_injector: ToxiproxyInjector,  # noqa: ARG001 - establishes ordering: toxiproxy must exist before we pin the operator at its Service
) -> AsyncIterator[OperatorDeployer]:
    """Operator redeployed with apiserver traffic routed through toxiproxy.

    The default ``operator_ready`` fixture relies on the cluster-injected
    ``KUBERNETES_SERVICE_HOST`` / ``KUBERNETES_SERVICE_PORT`` env vars, so all
    operator -> apiserver traffic goes directly to ``kubernetes.default.svc``.
    For chaos scenario C15 we need that path to traverse toxiproxy instead.
    This fixture redeploys the operator once with those env vars pinned at the
    chaos-namespace toxiproxy Service and restores a plain operator at teardown
    so sibling package tests run with production-shaped routing.

    **Proxy lifecycle is the caller's responsibility.** The toxiproxy
    proxy itself (``add_proxy(name=..., listen=..., upstream=...)``) is
    NOT created here because the real upstream (per-CR controller pod IP)
    is unknown until after the CR's JobSet spawns. Tests should:

    1. Create an AIPerfJob via the yielded ``OperatorDeployer``.
    2. Poll for the controller pod IP via
       ``ChaosInjector.get_controller_pod_name`` +
       ``kubectl get pod ... -o jsonpath={.status.podIP}``.
    3. Call ``toxiproxy_injector.add_proxy(name="controller",
       listen=f"0.0.0.0:{TOXIPROXY_CONTROLLER_HTTP_PORT}",
       upstream=f"{pod_ip}:19090")`` to open the data path.
    4. Add toxics (``add_toxic("controller", "timeout", ...)``) to
       inject faults.
    5. ``await toxiproxy_injector.reset()`` in ``finally`` to wipe
       proxies+toxics for the next test.

    Until step 3 runs, the operator's initial controller-HTTP calls fail
    with a transport error against an unroutable proxy — the operator
    retries and proceeds normally once the proxy is wired up.

    Scope: ``package`` (matches every other chaos fixture that requires
    a living operator Deployment). Do NOT compose this fixture with the
    default ``operator_ready`` in the same test — they both install the
    operator, and the second install fights the first's labels.
    """
    deployer = OperatorDeployer(
        kubectl=kubectl,
        project_root=project_root,
        operator_image=k8s_settings.aiperf_image,
        default_job_namespace=operator_job_namespace,
        apiserver_service_host_override=APISERVER_SERVICE_HOST_OVERRIDE,
        apiserver_service_port_override=APISERVER_SERVICE_PORT_OVERRIDE,
    )
    await deployer.install_crd()
    await kubectl.run("create", "namespace", operator_job_namespace, check=False)
    await deployer.deploy_operator()
    try:
        yield deployer
    finally:
        if not k8s_settings.skip_cleanup:
            await deployer.cleanup_all()
            restore = OperatorDeployer(
                kubectl=kubectl,
                project_root=project_root,
                operator_image=k8s_settings.aiperf_image,
                default_job_namespace=operator_job_namespace,
                controller_http_url_override=None,
            )
            await restore.deploy_operator()
