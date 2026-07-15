# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from tests.kubernetes.helpers.cluster import ClusterConfig, ClusterRuntime, LocalCluster
from tests.kubernetes.helpers.helm import HelmValues


@pytest.mark.parametrize(
    ("runtime", "expected_context"),
    [
        param(ClusterRuntime.KIND, "kind-aiperf-pytest", id="kind"),
        param(ClusterRuntime.MINIKUBE, "aiperf-pytest", id="minikube"),
    ],
)  # fmt: skip
def test_local_cluster_uses_cluster_config_runtime(
    runtime: ClusterRuntime,
    expected_context: str,
) -> None:
    cluster = LocalCluster(config=ClusterConfig(runtime=runtime))

    assert cluster.runtime is runtime
    assert cluster.context == expected_context


def test_helm_values_include_chaos_controller_http_override() -> None:
    values = HelmValues(controller_http_url_override="http://toxiproxy:20002")

    assert (
        "chaos.controllerHttpUrlOverride=http://toxiproxy:20002" in values.to_set_args()
    )


def test_helm_values_include_apiserver_tls_route_overrides() -> None:
    values = HelmValues(
        apiserver_service_host_override="toxiproxy.aiperf-chaos-toxiproxy.svc.cluster.local",
        apiserver_service_port_override="20000",
        apiserver_tls_server_name_override="kubernetes.default.svc",
    )

    args = values.to_set_args()

    assert (
        "chaos.apiserverServiceHostOverride=toxiproxy.aiperf-chaos-toxiproxy.svc.cluster.local"
        in args
    )
    assert "chaos.apiserverServicePortOverride=20000" in args
    assert "chaos.apiserverTlsServerNameOverride=kubernetes.default.svc" in args
