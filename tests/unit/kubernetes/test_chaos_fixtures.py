# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.kubernetes.environment import K8sEnvironment
from tests.kubernetes.chaos import conftest as chaos_conftest


def test_controller_http_toxiproxy_fixture_is_exported() -> None:
    assert hasattr(chaos_conftest, "operator_ready_toxiproxy_routed")


def test_controller_http_toxiproxy_upstream_uses_api_service_port() -> None:
    assert (
        chaos_conftest.CONTROLLER_HTTP_UPSTREAM_PORT == K8sEnvironment.PORTS.API_SERVICE
    )


def test_operator_override_fixtures_restore_after_each_test() -> None:
    assert (
        chaos_conftest.operator_ready_toxiproxy_routed._fixture_function_marker.scope
        == "function"
    )
    assert (
        chaos_conftest.operator_ready_apiserver_toxiproxy_routed._fixture_function_marker.scope
        == "function"
    )


def test_apiserver_toxiproxy_fixture_declares_tls_server_name_override() -> None:
    assert chaos_conftest.APISERVER_TLS_SERVER_NAME_OVERRIDE == "kubernetes.default.svc"


def test_operator_env_assertion_names_missing_env_var() -> None:
    missing = chaos_conftest._missing_operator_env_vars(
        env_stdout="AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE=http://proxy\n",
        expected={
            "AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE": "http://proxy",
            "KUBERNETES_SERVICE_HOST": "toxiproxy.aiperf-chaos-toxiproxy.svc.cluster.local",
        },
    )

    assert missing == ["KUBERNETES_SERVICE_HOST"]
