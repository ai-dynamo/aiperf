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

from aiperf.kubernetes.environment import K8sEnvironment
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
CONTROLLER_HTTP_UPSTREAM_PORT = K8sEnvironment.PORTS.API_SERVICE
APISERVER_SERVICE_HOST_OVERRIDE = (
    f"{TOXIPROXY_SERVICE}.{TOXIPROXY_NAMESPACE}.svc.cluster.local"
)
APISERVER_SERVICE_PORT_OVERRIDE = str(TOXIPROXY_APISERVER_PORT)
APISERVER_TLS_SERVER_NAME_OVERRIDE = "kubernetes.default.svc"


def _missing_operator_env_vars(
    env_stdout: str,
    expected: dict[str, str],
) -> list[str]:
    """Return expected env names absent from ``kubectl set env --list`` output."""
    observed: dict[str, str] = {}
    for line in env_stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            observed[name.strip()] = value.strip()
    return [name for name, value in expected.items() if observed.get(name) != value]


async def _assert_live_operator_env(
    kubectl: KubectlClient,
    expected: dict[str, str],
) -> None:
    """Fail fast when an override fixture deployed an operator without its env."""
    env_check = await kubectl.run(
        "set",
        "env",
        "deployment/aiperf-operator",
        "--list",
        "-n",
        OperatorDeployer.OPERATOR_NAMESPACE,
        check=True,
    )
    missing = _missing_operator_env_vars(env_check.stdout, expected)
    assert not missing, (
        "operator chaos-route precondition failed: live deployment is missing "
        f"expected env vars {missing!r}; check the override fixture wiring"
    )


@pytest_asyncio.fixture
async def operator_ready_toxiproxy_routed(
    kubectl: KubectlClient,
    project_root: Path,
    loaded_images,  # noqa: ANN001 - session-scoped helper, not typed in test surface
    jobset_controller: None,
    mock_server: None,
    k8s_settings,  # noqa: ANN001 - test-fixture dataclass
    operator_job_namespace: str,
    toxiproxy_injector: ToxiproxyInjector,  # noqa: ARG001 - establishes ordering: toxiproxy must exist before we pin the operator at its Service
) -> AsyncIterator[OperatorDeployer]:
    """Operator redeployed with controller HTTP traffic routed through toxiproxy."""
    deployer = OperatorDeployer(
        kubectl=kubectl,
        project_root=project_root,
        operator_image=k8s_settings.aiperf_image,
        default_job_namespace=operator_job_namespace,
        controller_http_url_override=CONTROLLER_HTTP_OVERRIDE_URL,
    )
    await deployer.install_crd()
    await kubectl.run("create", "namespace", operator_job_namespace, check=False)
    await deployer.deploy_operator()
    await _assert_live_operator_env(
        kubectl,
        {"AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE": CONTROLLER_HTTP_OVERRIDE_URL},
    )
    try:
        yield deployer
    finally:
        if not k8s_settings.skip_cleanup:
            await deployer.cleanup_all()
        # Restoring a plain operator is NOT a cleanup nicety and must run even
        # under --k8s-skip-cleanup. The toxiproxy env overrides are shared
        # mutable cluster state, and `operator_ready` reuses any operator whose
        # Deployment merely reports readyReplicas -- it never inspects env. Left
        # in place, every later test in the session runs against an operator
        # still pinned at a toxiproxy listener whose upstream controller pod is
        # gone, so _apply_controller_progress_status (the only writer of
        # phase=Running / currentPhase=profiling) never runs and CRs go
        # Pending -> Initializing -> Completed. Completion still succeeds
        # because it is JobSet-watch driven and results use the non-overridden
        # results-sidecar port, which makes the leak silent.
        restore = OperatorDeployer(
            kubectl=kubectl,
            project_root=project_root,
            operator_image=k8s_settings.aiperf_image,
            default_job_namespace=operator_job_namespace,
            controller_http_url_override=None,
        )
        await restore.deploy_operator()


@pytest_asyncio.fixture
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
    chaos-namespace toxiproxy Service, while keeping TLS verification pinned to
    ``kubernetes.default.svc``, and restores a plain operator at teardown so
    sibling package tests run with production-shaped routing.

    The apiserver proxy must exist before the operator is deployed because
    kopf's login and watches use the overridden Kubernetes service env vars
    immediately at startup. Tests add toxics after the yielded deployer creates
    a CR and call ``await toxiproxy_injector.reset()`` in ``finally``.

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
        apiserver_tls_server_name_override=APISERVER_TLS_SERVER_NAME_OVERRIDE,
    )
    await deployer.install_crd()
    await kubectl.run("create", "namespace", operator_job_namespace, check=False)
    await toxiproxy_injector.add_proxy(
        name="apiserver",
        listen=f"0.0.0.0:{TOXIPROXY_APISERVER_PORT}",
        upstream="kubernetes.default.svc:443",
    )
    await deployer.deploy_operator()
    await _assert_live_operator_env(
        kubectl,
        {
            "KUBERNETES_SERVICE_HOST": APISERVER_SERVICE_HOST_OVERRIDE,
            "KUBERNETES_SERVICE_PORT": APISERVER_SERVICE_PORT_OVERRIDE,
            "AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE": APISERVER_TLS_SERVER_NAME_OVERRIDE,
        },
    )
    try:
        yield deployer
    finally:
        if not k8s_settings.skip_cleanup:
            await deployer.cleanup_all()
        # Restoring a plain operator is NOT a cleanup nicety and must run even
        # under --k8s-skip-cleanup. The toxiproxy env overrides are shared
        # mutable cluster state, and `operator_ready` reuses any operator whose
        # Deployment merely reports readyReplicas -- it never inspects env. Left
        # in place, every later test in the session runs against an operator
        # still pinned at a toxiproxy listener whose upstream controller pod is
        # gone, so _apply_controller_progress_status (the only writer of
        # phase=Running / currentPhase=profiling) never runs and CRs go
        # Pending -> Initializing -> Completed. Completion still succeeds
        # because it is JobSet-watch driven and results use the non-overridden
        # results-sidecar port, which makes the leak silent.
        restore = OperatorDeployer(
            kubectl=kubectl,
            project_root=project_root,
            operator_image=k8s_settings.aiperf_image,
            default_job_namespace=operator_job_namespace,
            controller_http_url_override=None,
        )
        await restore.deploy_operator()
