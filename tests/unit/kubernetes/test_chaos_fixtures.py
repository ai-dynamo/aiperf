# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests.kubernetes.chaos import conftest as chaos_conftest


def test_controller_http_toxiproxy_fixture_is_exported() -> None:
    assert hasattr(chaos_conftest, "operator_ready_toxiproxy_routed")


def test_operator_override_fixtures_restore_after_each_test() -> None:
    assert (
        chaos_conftest.operator_ready_toxiproxy_routed._fixture_function_marker.scope
        == "function"
    )
    assert (
        chaos_conftest.operator_ready_apiserver_toxiproxy_routed._fixture_function_marker.scope
        == "function"
    )
