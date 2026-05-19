# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from tests.kubernetes.helpers.cluster import ClusterConfig, ClusterRuntime, LocalCluster


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
